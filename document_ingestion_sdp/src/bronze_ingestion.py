# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer: AutoLoader Ingestion + Parsing + Element Extraction
# MAGIC
# MAGIC Streaming ingestion of PDFs via AutoLoader, classification, parsing with
# MAGIC `ai_parse_document v2.0`, and element extraction via LATERAL VIEW explode.

# COMMAND ----------

import pyspark.pipelines as dp
from config import SOURCE_VOLUME_PATH, IMAGE_OUTPUT_PATH, DOC_TYPE_LABELS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Table: docs_bronze_parsed_docs_raw

# COMMAND ----------

@dp.table(
    name="docs_bronze_parsed_docs_raw",
    comment="Ingests PDFs via AutoLoader, classifies document type, and parses with ai_parse_document v2.0.",
)
def docs_silver_parsed_docs_raw():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", "*.pdf")
        .load(SOURCE_VOLUME_PATH)
        .selectExpr(
            "regexp_extract(path, '[^/]+$', 0) AS file_name",
            "path AS file_path",
            "length AS file_size",
            f"ai_classify(regexp_extract(path, '[^/]+$', 0), ARRAY{DOC_TYPE_LABELS}) AS document_type",
            f"ai_parse_document(content, map('version', '2.0', 'imageOutputPath', '{IMAGE_OUTPUT_PATH}', 'descriptionElementTypes', '*')) AS parsed_doc",
            "current_timestamp() AS ingested_at",
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Table: docs_bronze_elements

# COMMAND ----------

@dp.table(
    name="docs_bronze_elements",
    comment="Individual elements (text, tables, figures) extracted from parsed documents.",
)
def docs_bronze_elements():
    return spark.sql("""
        SELECT
            file_name,
            file_path,
            document_type,
            element.id::INT AS element_id,
            element.type::STRING AS element_type,
            element.content::STRING AS content,
            element.description::STRING AS description,
            element.bbox[0].page_id::INT AS page_id,
            to_json(element.bbox) AS bbox_json,
            current_timestamp() AS extracted_at
        FROM STREAM(docs_bronze_parsed_docs_raw)
        LATERAL VIEW OUTER explode(from_json(
            to_json(parsed_doc:document:elements),
            'ARRAY<STRUCT<id: INT, type: STRING, content: STRING, description: STRING, bbox: ARRAY<STRUCT<coord: ARRAY<INT>, page_id: INT>>>>'
        )) AS element
    """)
