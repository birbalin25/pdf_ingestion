# Databricks notebook source
# MAGIC %md
# MAGIC # Aggregate Document Elements by Page
# MAGIC
# MAGIC This notebook aggregates document elements from the silver layer by page_id:
# MAGIC - Concatenates all element content per page with newlines
# MAGIC - Marks headers and titles with appropriate prefixes
# MAGIC - Creates a gold layer table with aggregated page content
# MAGIC
# MAGIC **Input:** `document_elements_table` (silver layer)
# MAGIC **Output:** `aggregated_pages_table` (gold layer)

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

# Silver layer config (input)
elements_table = config["ingestion"]["silver_layer"]["document_elements_table"]

# Gold layer config (output)
aggregated_pages_table = config["ingestion"]["gold_layer"]["aggregated_pages_table"]
quality_rater_llm = config["ingestion"]["gold_layer"]["page_quality_rater_llm"]
quality_rater_prompt = config["ingestion"]["gold_layer"]["page_quality_rater_prompt"]
quality_scored_table = "docs_gold_pages_with_quality_scores"

print(f"Configuration loaded.")
print(f"  Input: {catalog}.{schema}.{elements_table}")
print(f"  Output: {catalog}.{schema}.{aggregated_pages_table}")
print(f"  Quality scored output: {catalog}.{schema}.{quality_scored_table}")
print(f"  Quality rater LLM: {quality_rater_llm}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Spark Session

# COMMAND ----------

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Aggregation Logic

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# Read elements table
elements_table_path = f"{catalog}.{schema}.{elements_table}"
elements_df = spark.table(elements_table_path)

print(f"Loaded {elements_df.count()} elements from {elements_table_path}")

# Show element types for reference
print("\nElement types in table:")
elements_df.groupBy("element_type").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Format and Aggregate Elements by Page
# MAGIC
# MAGIC - Headers are prefixed with `## `
# MAGIC - Titles are prefixed with `# `
# MAGIC - Other elements are added as-is
# MAGIC - Elements are ordered by element_id within each page

# COMMAND ----------

# Format content based on element type
# Headers get "## " prefix, titles get "# " prefix
formatted_elements_df = elements_df.withColumn(
    "formatted_content",
    F.when(
        F.lower(F.col("element_type")) == "title",
        F.concat(F.lit("# "), F.coalesce(F.col("content"), F.lit("")))
    ).when(
        F.lower(F.col("element_type")).contains("header"),
        F.concat(F.lit("## "), F.coalesce(F.col("content"), F.lit("")))
    ).when(
        F.lower(F.col("element_type")).isin("table", "figure"),
        # For tables and figures, include description if available
        F.when(
            F.col("description").isNotNull() & (F.col("description") != ""),
            F.concat(
                F.lit("["),
                F.upper(F.col("element_type")),
                F.lit("]\n"),
                F.coalesce(F.col("content"), F.lit("")),
                F.lit("\n[Description: "),
                F.col("description"),
                F.lit("]")
            )
        ).otherwise(
            F.concat(
                F.lit("["),
                F.upper(F.col("element_type")),
                F.lit("]\n"),
                F.coalesce(F.col("content"), F.lit(""))
            )
        )
    ).otherwise(
        F.coalesce(F.col("content"), F.lit(""))
    )
)

# Filter out empty content
formatted_elements_df = formatted_elements_df.filter(
    F.col("formatted_content").isNotNull() & (F.trim(F.col("formatted_content")) != "")
)

# COMMAND ----------

display(formatted_elements_df)

# COMMAND ----------

# Aggregate elements by page, ordered by element_id
# Group by document identifiers + page_id
aggregated_df = formatted_elements_df.groupBy(
    "file_name",
    "document_type",
    "page_id"
).agg(
    # Collect content in order of element_id and join with newlines
    F.concat_ws(
        "\n\n",
        F.array_sort(
            F.collect_list(
                F.struct(
                    F.col("element_id"),
                    F.col("formatted_content")
                )
            )
        ).getField("formatted_content")
    ).alias("page_content"),
    # Also track element counts and types for metadata
    F.count("*").alias("element_count"),
    F.collect_set("element_type").alias("element_types"),
    F.max("extracted_at").alias("extracted_at")
).withColumn(
    "aggregated_at",
    F.current_timestamp()
)

# Convert element_types array to string for easier querying
aggregated_df = aggregated_df.withColumn(
    "element_types",
    F.array_join(F.col("element_types"), ", ")
)

# Order by document and page
aggregated_df = aggregated_df.orderBy("file_name", "page_id")

print(f"Aggregated to {aggregated_df.count()} pages")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview Aggregated Data

# COMMAND ----------

# Show sample of aggregated pages
display(aggregated_df.select(
    "file_name",
    "page_id",
    "element_count",
    "element_types",
    F.substring("page_content", 1, 500).alias("content_preview")
).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold Layer Table

# COMMAND ----------

aggregated_pages_table_path = f"{catalog}.{schema}.{aggregated_pages_table}"

aggregated_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(aggregated_pages_table_path)

print(f"Aggregated pages written to {aggregated_pages_table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

result_df = spark.table(aggregated_pages_table_path)

total_pages = result_df.count()
unique_docs = result_df.select("file_name").distinct().count()
avg_elements_per_page = result_df.agg(F.avg("element_count")).collect()[0][0]

print("=" * 50)
print("AGGREGATION SUMMARY")
print("=" * 50)
print(f"Documents processed: {unique_docs}")
print(f"Total pages: {total_pages}")
print(f"Avg elements per page: {avg_elements_per_page:.1f}")

# COMMAND ----------

# Content length statistics
content_stats = result_df.agg(
    F.avg(F.length("page_content")).alias("avg_content_length"),
    F.min(F.length("page_content")).alias("min_content_length"),
    F.max(F.length("page_content")).alias("max_content_length")
).collect()[0]

print(f"\nContent length statistics:")
print(f"  Average: {content_stats['avg_content_length']:.0f} characters")
print(f"  Min: {content_stats['min_content_length']} characters")
print(f"  Max: {content_stats['max_content_length']} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rate Page Quality with LLM
# MAGIC
# MAGIC Use `ai_query` to evaluate each page's helpfulness for financial analysis.
# MAGIC Pages are rated on a binary scale (0 = not helpful, 1 = helpful).

# COMMAND ----------

print(f"Rating page quality using {quality_rater_llm}...")

# Create temp view for SQL query
result_df.createOrReplaceTempView("aggregated_pages")

# Build the quality rating query using ai_query
# The prompt instructs the LLM to return only 0 or 1
quality_query = f"""
SELECT
    file_name,
    document_type,
    page_id,
    page_content,
    element_count,
    element_types,
    extracted_at,
    aggregated_at,
    ai_query(
        '{quality_rater_llm}',
        CONCAT('{quality_rater_prompt}', page_content)
    ) AS quality_score_raw,
    current_timestamp() AS quality_rated_at
FROM aggregated_pages
"""

quality_rated_df = spark.sql(quality_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse Quality Scores and Filter

# COMMAND ----------

# Parse the quality score (LLM returns string, convert to int)
# Handle potential edge cases where LLM might return extra text
quality_rated_df = quality_rated_df.withColumn(
    "quality_score",
    F.when(
        F.trim(F.col("quality_score_raw")).rlike("^1$|^1[^0-9]"),
        F.lit(1)
    ).when(
        F.trim(F.col("quality_score_raw")).rlike("^0$|^0[^0-9]"),
        F.lit(0)
    ).otherwise(
        # If response contains "1" anywhere, treat as helpful
        F.when(F.col("quality_score_raw").contains("1"), F.lit(1)).otherwise(F.lit(0))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Quality-Scored Gold Table

# COMMAND ----------

# Select final columns for the quality-scored table
final_df = quality_rated_df.select(
    "file_name",
    "company_name",
    "year",
    "document_type",
    "page_id",
    "page_content",
    "element_count",
    "element_types",
    "quality_score",
    "extracted_at",
    "aggregated_at",
    "quality_rated_at"
)

quality_scored_table_path = f"{catalog}.{schema}.{quality_scored_table}"

final_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(quality_scored_table_path)

print(f"Quality-scored pages written to {quality_scored_table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality-Scored Table Summary

# COMMAND ----------

quality_result_df = spark.table(quality_scored_table_path)

total_pages = quality_result_df.count()
quality_pages = quality_result_df.filter(F.col("quality_score") == 1).count()
filtered_count = total_pages - quality_pages

print("=" * 50)
print("QUALITY-SCORED TABLE SUMMARY")
print("=" * 50)

print(f"  Total pages rated: {total_pages}")
print(f"  Helpful pages (score=1): {quality_pages}")
print(f"  Filtered out (score=0): {filtered_count}")
print(f"  Retention rate: {quality_pages/total_pages*100:.1f}%")


# COMMAND ----------

print("\nPage aggregation and quality rating pipeline complete!")
