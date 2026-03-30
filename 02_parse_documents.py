# Databricks notebook source
# MAGIC %md
# MAGIC # Parse Documents using AI Parse Document
# MAGIC
# MAGIC This notebook parses SEC filing PDFs using `ai_parse_document` version 2.0:
# MAGIC
# MAGIC **Pipeline:**
# MAGIC 1. Parse documents and save raw results to intermediate table (`parsed_document_table`)
# MAGIC 2. Extract pages from intermediate table → `document_pages_table`
# MAGIC 3. Extract elements from intermediate table → `document_elements_table`
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Intermediate table with raw parsed document JSON
# MAGIC - Pages table with rendered page images
# MAGIC - Elements table with text, tables, and figures (with AI-generated descriptions)

# COMMAND ----------

# MAGIC %pip install pyyaml --quiet
# MAGIC %restart_python

# COMMAND ----------

import os
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    config_path = "config.yaml"
else:
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(notebook_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

catalog = config["global"]["catalog"]
schema = config["global"]["schema"]

# Bronze layer config
bronze_table = config["ingestion"]["bronze_layer"]["bronze_table"]

# Silver layer config
silver_config = config["ingestion"]["silver_layer"]
parsed_docs_table = silver_config["parsed_document_table"]
elements_table = silver_config["document_elements_table"]
pages_table = silver_config["document_pages_table"]
image_output_path = f"/Volumes/{catalog}/{schema}/{silver_config['image_output_path']}"
source_volume_path = f"/Volumes/{catalog}/{schema}/{config['ingestion']['source_volume_path']}"


print(f"Configuration loaded.")
print(f"  Bronze: {catalog}.{schema}.{bronze_table}")
print(f"  Silver parsed docs (intermediate): {catalog}.{schema}.{parsed_docs_table}")
print(f"  Silver pages: {catalog}.{schema}.{pages_table}")
print(f"  Silver elements: {catalog}.{schema}.{elements_table}")
print(f"  image_output_path: {image_output_path}")
print(f"  source_volume_path: {source_volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F

# Create image output volume if needed
if image_output_path.startswith("/Volumes/"):
    parts = image_output_path.split("/")
    if len(parts) >= 5:
        vol_catalog, vol_schema, vol_name = parts[2], parts[3], parts[4]
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {vol_catalog}.{vol_schema}.{vol_name}")

print("Setup complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Bronze Table and Parse Documents

# COMMAND ----------

bronze_table_path = f"{catalog}.{schema}.{bronze_table}"
parsed_docs_table_path = f"{catalog}.{schema}.{parsed_docs_table}"

bronze_df = spark.table(bronze_table_path)
bronze_df.createOrReplaceTempView("bronze_docs")

print("Parsing documents with ai_parse_document v2.0...")

# Parse documents using ai_parse_document
parse_query = f"""
SELECT
    b.file_name,
    b.file_path,
    b.document_type,
    ai_parse_document(
        content,
        map(
            'version', '2.0',
            'imageOutputPath', '{image_output_path}',
            'descriptionElementTypes', '*'
        )
    ) AS parsed_doc,
    current_timestamp() AS parsed_at
FROM bronze_docs b
JOIN READ_FILES('{f"{source_volume_path}"}', format => 'binaryFile') r
ON b.file_name = regexp_extract(r.path, '[^/]+$', 0)
"""

parsed_df = spark.sql(parse_query)

# Write to intermediate parsed documents table
parsed_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(parsed_docs_table_path)

print(f"Document parsing complete. Written to {parsed_docs_table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Pages Silver Table

# COMMAND ----------

parsed_docs_df = spark.table(parsed_docs_table_path)
display(parsed_docs_df.withColumn("parsed_doc", F.col("parsed_doc").cast("string")).limit(1))

# COMMAND ----------

print("Extracting pages from intermediate table...")

# Read from the intermediate parsed documents table
parsed_docs_df = spark.table(parsed_docs_table_path)
parsed_docs_df.createOrReplaceTempView("parsed_docs")

pages_query = """
SELECT
    file_name,
    document_type,
    page.id::INT AS page_id,
    page.image_uri::STRING AS image_uri,
    current_timestamp() AS extracted_at
FROM parsed_docs
LATERAL VIEW OUTER explode(from_json(
    to_json(parsed_doc:document:pages),
    'ARRAY<STRUCT<id: INT, image_uri: STRING>>'
)) AS page
"""

pages_df = spark.sql(pages_query)

# Write pages silver table
pages_table_path = f"{catalog}.{schema}.{pages_table}"
pages_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(pages_table_path)

print(f"Pages written to {pages_table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Elements Silver Table

# COMMAND ----------

display(table(pages_table_path))

# COMMAND ----------

print("Extracting elements from intermediate table...")

# Note: parsed_docs temp view already created in previous step from intermediate table

elements_query = """
SELECT
    file_name,
    document_type,
    element.id::INT AS element_id,
    element.type::STRING AS element_type,
    element.content::STRING AS content,
    element.description::STRING AS description,
    element.bbox[0].page_id::INT AS page_id,
    to_json(element.bbox) AS bbox_json,
    current_timestamp() AS extracted_at
FROM parsed_docs
LATERAL VIEW OUTER explode(from_json(
    to_json(parsed_doc:document:elements),
    'ARRAY<STRUCT<id: INT, type: STRING, content: STRING, description: STRING, bbox: ARRAY<STRUCT<coord: ARRAY<INT>, page_id: INT>>>>'
)) AS element
"""

elements_df = spark.sql(elements_query)

# Write elements silver table
elements_table_path = f"{catalog}.{schema}.{elements_table}"
elements_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(elements_table_path)

print(f"Elements written to {elements_table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

display(table(elements_table_path))

# COMMAND ----------

# Read silver tables back for summary stats
result_pages = spark.table(pages_table_path)
result_elements = spark.table(elements_table_path)

total_pages = result_pages.count()
total_elements = result_elements.count()
unique_docs = result_pages.select("file_name").distinct().count()

print("=" * 50)
print("PARSING SUMMARY")
print("=" * 50)
print(f"Documents processed: {unique_docs}")
print(f"Total pages: {total_pages}")
print(f"Total elements: {total_elements}")
print(f"Avg pages per document: {total_pages / unique_docs if unique_docs > 0 else 0:.1f}")
print(f"Avg elements per document: {total_elements / unique_docs if unique_docs > 0 else 0:.1f}")

# COMMAND ----------

# Element type distribution
print("\nElements by Type:")
element_types = result_elements.groupBy("element_type").count().orderBy(F.desc("count")).collect()
for row in element_types:
    print(f"  {row['element_type']}: {row['count']}")
