# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer: Format Elements and Aggregate by Page
# MAGIC
# MAGIC Formats element content with markdown-style prefixes and aggregates
# MAGIC all elements per page into a single `page_content` column.

# COMMAND ----------

import pyspark.pipelines as dp
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Materialized View: docs_silver_pages

# COMMAND ----------

@dp.materialized_view(
    name="docs_silver_pages",
    comment="Silver layer: elements formatted and aggregated per page with markdown-style content.",
)
def docs_silver_pages():
    elements_df = spark.table("docs_bronze_elements")

    # Format content based on element type:
    #   title  → "# <content>"
    #   header → "## <content>"  (matches header, sectionheader, pageheader, etc.)
    #   table/figure → "[TYPE]\n<content>\n[Description: <desc>]"
    #   other  → content as-is
    formatted_df = elements_df.withColumn(
        "formatted_content",
        F.when(
            F.lower(F.trim(F.col("element_type"))) == "title",
            F.concat(F.lit("# "), F.coalesce(F.col("content"), F.lit(""))),
        )
        .when(
            F.lower(F.trim(F.col("element_type"))).contains("header"),
            F.concat(F.lit("## "), F.coalesce(F.col("content"), F.lit(""))),
        )
        .when(
            F.lower(F.trim(F.col("element_type"))).isin("table", "figure"),
            F.when(
                F.col("description").isNotNull()
                & (F.trim(F.col("description")) != ""),
                F.concat(
                    F.lit("["),
                    F.upper(F.trim(F.col("element_type"))),
                    F.lit("]\n"),
                    F.coalesce(F.col("content"), F.lit("")),
                    F.lit("\n[Description: "),
                    F.col("description"),
                    F.lit("]"),
                ),
            ).otherwise(
                F.concat(
                    F.lit("["),
                    F.upper(F.trim(F.col("element_type"))),
                    F.lit("]\n"),
                    F.coalesce(F.col("content"), F.lit("")),
                )
            ),
        )
        .otherwise(F.coalesce(F.col("content"), F.lit(""))),
    )

    # Filter out empty content
    formatted_df = formatted_df.filter(
        F.col("formatted_content").isNotNull()
        & (F.trim(F.col("formatted_content")) != "")
    )

    # Aggregate elements by page, ordered by element_id
    aggregated_df = (
        formatted_df.groupBy("file_name", "file_path", "document_type", "page_id")
        .agg(
            F.concat_ws(
                "\n\n",
                F.array_sort(
                    F.collect_list(
                        F.struct(F.col("element_id"), F.col("formatted_content"))
                    )
                ).getField("formatted_content"),
            ).alias("page_content"),
            F.count("*").alias("element_count"),
            F.collect_set("element_type").alias("element_types"),
            F.max("extracted_at").alias("extracted_at"),
        )
        .withColumn("aggregated_at", F.current_timestamp())
    )

    # Convert element_types array to comma-separated string
    aggregated_df = aggregated_df.withColumn(
        "element_types", F.array_join(F.col("element_types"), ", ")
    )

    return aggregated_df.orderBy("file_name", "page_id")
