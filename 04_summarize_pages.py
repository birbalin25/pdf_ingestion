# Databricks notebook source
# MAGIC %md
# MAGIC # Summarize Page Contents
# MAGIC
# MAGIC This notebook summarizes the content of high-quality pages using an LLM:
# MAGIC 1. Filter pages with quality_score = 1 from the gold layer
# MAGIC 2. Concatenate metadata (company_name, document_type, year) to page content
# MAGIC 3. Summarize using ai_query() with configured LLM and prompt
# MAGIC 4. Write to platinum layer table
# MAGIC
# MAGIC **Input:** `sec_docs_gold_pages_with_quality_scores` (gold layer)
# MAGIC **Output:** `sec_docs_platinum` (platinum layer)

# COMMAND ----------

# MAGIC %pip install pyyaml --quiet
# MAGIC %restart_python

# COMMAND ----------

import os
import yaml
from pyspark.sql.functions import *

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

# Gold layer config (input)
quality_scored_table = "docs_gold_pages_with_quality_scores"
aggregated_pages_table = config["ingestion"]["gold_layer"]["aggregated_pages_table"]

# Platinum layer config (output)
platinum_config = config["ingestion"]["platinum_layer"]
summarization_llm = platinum_config["summarization_llm"]
summarization_prompt = platinum_config["summarization_prompt"]
platinum_table = platinum_config["platinum_table"]

print(f"Configuration loaded.")
print(f"  Input: {catalog}.{schema}.{quality_scored_table}")
print(f"  Output: {catalog}.{schema}.{platinum_table}")
print(f"  Summarization LLM: {summarization_llm}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Spark Session

# COMMAND ----------

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Source Table and Add UUIDs

# COMMAND ----------

from pyspark.sql import functions as F

# Read the quality-scored table
# quality_scored_table_path = f"{catalog}.{schema}.{quality_scored_table}" aggregated_pages_table
quality_scored_table_path = f"{catalog}.{schema}.{aggregated_pages_table}"
quality_df = spark.table(quality_scored_table_path)

# Add UUID to each row before any filtering or processing
high_quality_df = quality_df.withColumn("chunk_id", F.expr("uuid()"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter Quality Pages

# COMMAND ----------

# # Filter for high-quality pages only (quality_score = 1)
# high_quality_df = quality_df.filter(F.col("quality_score") == 1)

# high_quality_count = high_quality_df.count()

# print(f"High-quality pages (score=1): {high_quality_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Content with Metadata Context
# MAGIC
# MAGIC Concatenate company_name, document_type, and year to page_content
# MAGIC to provide context for the summarization.

# COMMAND ----------

# Create enriched content with metadata context
enriched_df = high_quality_df.withColumn(
    "enriched_content",
    F.concat(
        F.lit("\nDocument Type: "), F.col("document_type"),
        F.lit("\n\nPage Content:\n"), F.col("page_content")
    )
)

# Preview enriched content
print("Sample enriched content:")
sample = enriched_df.select("enriched_content").first()
if sample:
    print(sample["enriched_content"][:500] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summarize Pages with ai_query()

# COMMAND ----------

# print(f"Summarizing {high_quality_count} pages using {summarization_llm}...")

# Create temp view for SQL query
enriched_df.createOrReplaceTempView("enriched_pages")

# Escape single quotes in the prompt for SQL
escaped_prompt = summarization_prompt.replace("'", "''")

# Build the summarization query using ai_query
summarization_query = f"""
SELECT
    chunk_id,
    file_name,
    document_type,
    page_id,
    page_content,
    enriched_content,
    element_count,
    element_types,
    ai_query(
        '{summarization_llm}',
        CONCAT('{escaped_prompt}', '\\n\\n', enriched_content)
    ) AS page_summary,
    current_timestamp() AS summarized_at
FROM enriched_pages
"""

summarized_df = spark.sql(summarization_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Platinum Layer Table

# COMMAND ----------

# final_df = summarized_df.selectExpr(
#     "chunk_id",
#     "file_name",
#     "document_type",
#     "page_id",
#     "concat(page_content, page_summary) AS page_content_final",
#     "page_content",
#     "page_summary",
#     "element_count",
#     "element_types",
#     "summarized_at"
# )

# platinum_table_path = f"{catalog}.{schema}.{platinum_table}"

# final_df.write \
#     .mode("overwrite") \
#     .option("overwriteSchema", "true") \
#     .saveAsTable(platinum_table_path)

# print(f"Summarized pages written to {platinum_table_path}")

# COMMAND ----------

prompt= """

You are a text processing AI. Your task is to detect and mask all personally identifiable information (PII) in the text.

Instructions:

1. Only mask the PII types specified below. Other text and text format should remain unchanged.
2. Replace each detected PII with the mask format specified for that type.
3. Maintain the original text structure and punctuation.

PII types to mask and their mask format:
- Names: [[NAME]]
- Emails: <<mask all except domain.com>>
- Phone numbers: [[mask all except last 4 digits.Preserve the format]]
- Social Security Numbers (SSN): <<SSN>>
- Credit Card Numbers: [all all digits.Preserve the format]
- Addresses: [ADDRESS]

Output only the text with PII masked. Do not add any explanations.

Text to process:

"""

# COMMAND ----------

final_df = summarized_df.selectExpr(
    "chunk_id",
    "file_name",
    "document_type",
    "page_id",
    "concat(page_content, page_summary) AS page_content_final",
    "page_content",
    "page_summary",
    "element_count",
    "element_types",
    "summarized_at"
).withColumn(
    "page_content_final_masked",
    expr(f"""
      ai_query(
        'databricks-claude-opus-4-6',
        CONCAT('{prompt}', page_content_final)
      )
    """)
)

platinum_table_path = f"{catalog}.{schema}.{platinum_table}"

final_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(platinum_table_path)

print(f"Summarized pages written to {platinum_table_path}")

# COMMAND ----------

display(table(platinum_table_path).select("page_content_final","page_content_final_masked"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

result_df = spark.table(platinum_table_path)

total_summarized = result_df.count()
unique_docs = result_df.select("file_name").distinct().count()

print("=" * 50)
print("SUMMARIZATION SUMMARY")
print("=" * 50)
print(f"Documents processed: {unique_docs}")
print(f"Pages summarized: {total_summarized}")

# COMMAND ----------

# Summary length statistics
summary_stats = result_df.agg(
    F.avg(F.length("page_summary")).alias("avg_summary_length"),
    F.min(F.length("page_summary")).alias("min_summary_length"),
    F.max(F.length("page_summary")).alias("max_summary_length")
).collect()[0]

print(f"\nSummary length statistics:")
print(f"  Average: {summary_stats['avg_summary_length']:.0f} characters")
print(f"  Min: {summary_stats['min_summary_length']} characters")
print(f"  Max: {summary_stats['max_summary_length']} characters")

# COMMAND ----------

# Content compression ratio
compression_stats = result_df.agg(
    F.avg(F.length("page_content")).alias("avg_content_length"),
    F.avg(F.length("page_summary")).alias("avg_summary_length")
).collect()[0]

compression_ratio = compression_stats["avg_content_length"] / compression_stats["avg_summary_length"]
print(f"\nCompression ratio: {compression_ratio:.1f}x")
print(f"  (Average content: {compression_stats['avg_content_length']:.0f} chars -> Summary: {compression_stats['avg_summary_length']:.0f} chars)")
