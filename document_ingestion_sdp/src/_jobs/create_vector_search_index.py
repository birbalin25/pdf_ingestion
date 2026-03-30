# Databricks notebook source
# MAGIC %md
# MAGIC # Create / Sync Vector Search Index
# MAGIC
# MAGIC Runs as a standalone job task AFTER the gold PII masking step.
# MAGIC Logic preserved from 05_create_vector_search_index.py.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch --quiet
# MAGIC %restart_python

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

# ── Configuration ───────────────────────────────────────────────────────────
CATALOG = "bircatalog"
SCHEMA = "pdf2"
GOLD_TABLE = "docs_gold"
VS_ENDPOINT_NAME = "ka-f3925e58-vs-endpoint"
VS_INDEX_NAME = "docs_platinum_index_bir"
VS_PRIMARY_KEY = "chunk_id"
VS_EMBEDDING_SOURCE_COLUMN = "page_content_masked"
VS_EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"
VS_PIPELINE_TYPE = "TRIGGERED"

source_table_name = f"{CATALOG}.{SCHEMA}.{GOLD_TABLE}"
full_index_name = f"{CATALOG}.{SCHEMA}.{VS_INDEX_NAME}"

print(f"Source table: {source_table_name}")
print(f"Index: {full_index_name}")
print(f"Endpoint: {VS_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def endpoint_exists(client, endpoint_name):
    """Check if a Vector Search endpoint exists."""
    try:
        endpoints = client.list_endpoints()
        endpoint_names = [ep.get("name") for ep in endpoints.get("endpoints", [])]
        return endpoint_name in endpoint_names
    except Exception as e:
        print(f"Error checking endpoints: {e}")
        return False


def wait_for_endpoint_ready(client, endpoint_name, timeout=3600):
    """Wait for endpoint to be ready."""
    print(f"Waiting for endpoint '{endpoint_name}' to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            endpoint = client.get_endpoint(endpoint_name)
            status = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
            if status == "ONLINE":
                print(f"Endpoint '{endpoint_name}' is ready.")
                return True
            print(f"  Endpoint status: {status}")
        except Exception as e:
            print(f"  Error checking endpoint: {e}")
        time.sleep(30)
    raise TimeoutError(
        f"Endpoint '{endpoint_name}' did not become ready within {timeout} seconds."
    )


def index_exists(client, endpoint_name, index_name):
    """Check if a Vector Search index exists."""
    try:
        index = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
        return index is not None
    except Exception as e:
        if "NOT_FOUND" in str(e) or "does not exist" in str(e).lower():
            return False
        print(f"Error checking index: {e}")
        return False


def wait_for_index_ready(client, endpoint_name, index_name, timeout=1800):
    """Wait for index to be ready (up to 30 minutes by default)."""
    print(f"Waiting for index '{index_name}' to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            index = client.get_index(
                endpoint_name=endpoint_name, index_name=index_name
            )
            status = index.describe().get("status", {}).get("ready", False)
            detailed_state = (
                index.describe().get("status", {}).get("detailed_state", "UNKNOWN")
            )
            if status:
                print(f"Index '{index_name}' is ready!")
                return True
            print(f"  Index status: ready={status}, state={detailed_state}")
        except Exception as e:
            print(f"  Error checking index: {e}")
        time.sleep(60)

    print(
        f"Warning: Index did not become ready within {timeout} seconds. "
        "It may still be processing."
    )
    return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Change Data Feed

# COMMAND ----------

spark.sql(
    f"ALTER TABLE {source_table_name} "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)
print(f"Change Data Feed enabled on {source_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create / Verify Endpoint

# COMMAND ----------

client = VectorSearchClient()
print("Vector Search client initialized.")

if endpoint_exists(client, VS_ENDPOINT_NAME):
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists.")
else:
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}'...")
    client.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Endpoint '{VS_ENDPOINT_NAME}' creation initiated.")

wait_for_endpoint_ready(client, VS_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create / Sync Index

# COMMAND ----------

already_exists = index_exists(client, VS_ENDPOINT_NAME, full_index_name)
print(f"Index '{full_index_name}' exists: {already_exists}")

if already_exists:
    print(f"Index '{full_index_name}' already exists. Triggering sync...")
    index = client.get_index(
        endpoint_name=VS_ENDPOINT_NAME, index_name=full_index_name
    )
    sync_result = index.sync()
    print(f"Sync triggered successfully. Result: {sync_result}")
else:
    print(f"Creating new Delta Sync index '{full_index_name}'...")
    index = client.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        source_table_name=source_table_name,
        index_name=full_index_name,
        pipeline_type=VS_PIPELINE_TYPE,
        primary_key=VS_PRIMARY_KEY,
        embedding_source_column=VS_EMBEDDING_SOURCE_COLUMN,
        embedding_model_endpoint_name=VS_EMBEDDING_MODEL_ENDPOINT,
    )
    print(f"Index creation initiated. Details: {index}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Index Ready

# COMMAND ----------

wait_for_index_ready(client, VS_ENDPOINT_NAME, full_index_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

try:
    index = client.get_index(
        endpoint_name=VS_ENDPOINT_NAME, index_name=full_index_name
    )
    index_info = index.describe()
    print("=" * 60)
    print("VECTOR SEARCH INDEX SUMMARY")
    print("=" * 60)
    print(f"Index name: {full_index_name}")
    print(f"Endpoint: {VS_ENDPOINT_NAME}")
    print(f"Source table: {source_table_name}")
    print(f"Primary key: {VS_PRIMARY_KEY}")
    print(f"Embedding column: {VS_EMBEDDING_SOURCE_COLUMN}")
    print(f"Status: {index_info.get('status', {})}")
    num_rows = index_info.get("status", {}).get("num_rows", "N/A")
    print(f"Number of rows indexed: {num_rows}")
except Exception as e:
    print(f"Error getting index info: {e}")

print("\nVector Search index setup complete!")
print(f"You can now query the index using: {full_index_name}")
