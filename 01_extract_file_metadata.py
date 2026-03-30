# Databricks notebook source
# MAGIC %md
# MAGIC # Extract File Metadata using AI Functions
# MAGIC
# MAGIC This notebook extracts metadata from SEC filing PDFs using Databricks AI functions:
# MAGIC - **ai_extract**: Extract the year from the file path
# MAGIC - **ai_classify**: Classify the company name based on file path using labels from the entities table
# MAGIC - **ai_classify**: Classify the document type as 10k, 8k, 10q, or Earnings Report

# COMMAND ----------

# DBTITLE 1,Install pyyaml
# MAGIC %pip install pyyaml --quiet
# MAGIC %restart_python

# COMMAND ----------

import os
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Setup

# COMMAND ----------

# Detect if running in Databricks
IN_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

# Define display function that works both locally and in Databricks
if IN_DATABRICKS:
    # In Databricks, display is already available globally
    pass
else:
    # Locally, define display as a wrapper around show()
    print("in else")
    def display(df, n=20, truncate=True):
        """Local replacement for Databricks display() function."""
        if hasattr(df, 'show'):
            df.show(n=n, truncate=truncate)
        elif hasattr(df, 'toPandas'):
            print(df.limit(n).toPandas().to_string())
        else:
            print(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Get the directory where this notebook is located
if IN_DATABRICKS:
    notebook_dir = os.getcwd()
    config_path = "config.yaml"
else:
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(notebook_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

catalog = config["global"]["catalog"]
schema = config["global"]["schema"]
source_volume_path = config["ingestion"]["source_volume_path"]
# entities_table = config["ingestion"]["entities_table"]
bronze_table = config["ingestion"]["bronze_layer"]["bronze_table"]
llm = config["ingestion"]["bronze_layer"]["metadata_extraction_llm"]

source_volume_path = f"/Volumes/{catalog}/{schema}/{source_volume_path}"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f" Source volume path: {source_volume_path}")
# print(f"Entities table: {catalog}.{schema}.{entities_table}")
print(f"Bronze table: {catalog}.{schema}.{bronze_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## List Files from Volume

# COMMAND ----------

from pyspark.sql import functions as F

# List all PDF files in the volume
files_df = spark.createDataFrame(
    [(f.name, f.path, f.size) for f in dbutils.fs.ls(source_volume_path) if f.name.endswith('.pdf')],
    ["file_name", "file_path", "file_size"]
)

print(f"Found {files_df.count()} PDF files")
display(files_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Company Labels from Entities Table

# COMMAND ----------

# # Get company names from entities table to use as classification labels
# entities_df = spark.table(f"{catalog}.{schema}.{entities_table}")
# display(entities_df)

# # Collect company names as a list for ai_query classification
# company_labels = [row.company_name for row in entities_df.select("company_name").collect()]
# print(f"Found {len(company_labels)} company labels")
# print(f"Sample labels: {company_labels[:10]}")

# # Format company labels as a comma-separated string for the prompt
# company_labels_str = ", ".join(company_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Temporary View for SQL AI Functions

# COMMAND ----------

# Register the files DataFrame as a temp view for SQL operations
files_df.createOrReplaceTempView("pdf_files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Year and Classify Company using AI Functions
# MAGIC
# MAGIC Using:
# MAGIC - `ai_extract(content, labels)` - Extracts entities from text, returns a STRUCT
# MAGIC - `ai_query(endpoint, prompt)` - Queries an LLM to classify company name (more flexible than ai_classify)
# MAGIC - `ai_classify(content, labels)` - Classifies document type into one of the provided labels

# COMMAND ----------

# Document type labels for classification (ai_classify supports 2-20 labels)
doc_type_labels = "('hr', 'finance', 'research', 'engineering','support')"

# # Build the company classification prompt
# doc_type_prompt = f"""Classify the document type from the given filename.
# Return ONLY the document type from this list, with no additional text or explanation:
# {doc_type_labels}

# Filename: """

# COMMAND ----------

# Extract metadata using AI functions
metadata_query = f"""
SELECT
    file_name,
    file_path,
    file_size,
    ai_classify(file_name, ARRAY{doc_type_labels}) AS document_type
FROM pdf_files
"""

print("Running AI extraction query...")
metadata_df = spark.sql(metadata_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Extracted Metadata

# COMMAND ----------

display(metadata_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Final Bronze Table Schema

# COMMAND ----------

# Select and rename columns for the bronze table
bronze_df = metadata_df.select(
    F.col("file_name"),
    F.col("file_path"),
    F.col("file_size"),
    F.col("document_type"),
    F.current_timestamp().alias("ingested_at")
)

display(bronze_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Bronze Table

# COMMAND ----------

# Write to bronze table
bronze_table_path = f"{catalog}.{schema}.{bronze_table}"

bronze_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(bronze_table_path)

print(f"Written {bronze_df.count()} records to {bronze_table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Bronze Table

# COMMAND ----------

# Verify the bronze table
result_df = spark.table(bronze_table_path)
print(f"Total records: {result_df.count()}")
display(result_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

print("Documents by Type:")
display(result_df.groupBy("document_type").count().orderBy(F.desc("count")))

# COMMAND ----------

