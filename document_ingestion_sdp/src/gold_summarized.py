# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer: PII Masking → Delta Table (docs_gold)
# MAGIC
# MAGIC Runs as a standalone job task AFTER the SDP pipeline refresh.
# MAGIC Reads from `docs_silver_pages` (materialized view) and writes a regular
# MAGIC Delta table – not a streaming table, not a materialized view.

# COMMAND ----------

from pyspark.sql import functions as F

# ── Configuration ───────────────────────────────────────────────────────────
CATALOG = "bircatalog"
SCHEMA = "pdf2"
GOLD_TABLE = "docs_gold"
PII_MASKING_LLM = "databricks-claude-opus-4-6"
PII_MASKING_PROMPT = (
    "You are a text processing AI. Your task is to detect and mask all personally identifiable "
    "information (PII) in the text. \\n\\n"
    "Instructions:\\n\\n"
    "1. Only mask the PII types specified below. Other text and text format should remain unchanged.\\n"
    "2. Replace each detected PII with the mask format specified for that type.\\n"
    "3. Maintain the original text structure and punctuation.\\n\\n"
    "PII types to mask and their mask format:\\n"
    "- Names: [[NAME]]\\n"
    "- Emails: <<mask all except domain.com>>\\n"
    "- Phone numbers: [[mask all except last 4 digits.Preserve the format]]\\n"
    "- Social Security Numbers (SSN): <<SSN>>\\n"
    "- Credit Card Numbers: [all all digits.Preserve the format]\\n"
    "- Addresses: [ADDRESS]\\n\\n"
    "Output only the text with PII masked. Do not add any explanations.\\n\\n"
    "Text to process:\\n\\n"
)

source_table = f"{CATALOG}.{SCHEMA}.docs_silver_pages"
target_table = f"{CATALOG}.{SCHEMA}.{GOLD_TABLE}"

print(f"Source: {source_table}")
print(f"Target: {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Source Table

# COMMAND ----------

base_df = spark.table(source_table)

# Add UUID
base_df = base_df.withColumn("chunk_id", F.expr("uuid()"))

print(f"Read {base_df.count()} rows from {source_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply PII Masking

# COMMAND ----------

escaped_pii_prompt = PII_MASKING_PROMPT.replace("'", "''")
masked_df = base_df.withColumn(
    "page_content_masked",
    F.expr(
        f"ai_query('{PII_MASKING_LLM}', "
        f"concat('{escaped_pii_prompt}', page_content))"
    ),
)

result_df = masked_df.select(
    "chunk_id",
    "file_name",
    "document_type",
    "page_id",
    "page_content",
    "page_content_masked",
    "element_count",
    "element_types",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold Table

# COMMAND ----------

result_df.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(target_table)

print(f"Gold table written to {target_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

gold_df = spark.table(target_table)
print(f"Total rows: {gold_df.count()}")
print(f"Unique documents: {gold_df.select('file_name').distinct().count()}")
