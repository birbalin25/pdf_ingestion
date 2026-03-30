# Databricks notebook source
# MAGIC %md
# MAGIC # Create Vector Search Index
# MAGIC
# MAGIC This notebook creates or syncs a Vector Search index from the platinum layer table:
# MAGIC 1. Check if the Vector Search endpoint exists, create if not
# MAGIC 2. Check if the index exists
# MAGIC 3. If index exists, trigger a sync
# MAGIC 4. If index doesn't exist, create a new Delta Sync index with computed embeddings
# MAGIC
# MAGIC **Input:** `sec_docs_platinum` (platinum layer)
# MAGIC **Output:** Vector Search index for semantic search

# COMMAND ----------

# MAGIC %pip install pyyaml databricks-vectorsearch --quiet
# MAGIC %restart_python

# COMMAND ----------

import os
import yaml
import time

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

# Platinum layer config (source table)
platinum_table = config["ingestion"]["platinum_layer"]["platinum_table"]

# Vector search config
vs_config = config["ingestion"]["vector_search"]
endpoint_name = vs_config["endpoint_name"]
index_name = vs_config["index_name"]
primary_key = vs_config["primary_key"]
embedding_source_column = vs_config["embedding_source_column"]
embedding_model_endpoint = vs_config["embedding_model_endpoint"]
pipeline_type = vs_config.get("pipeline_type", "TRIGGERED")

# Full paths
source_table_name = f"{catalog}.{schema}.{platinum_table}"
full_index_name = f"{catalog}.{schema}.{index_name}"

print(f"Configuration loaded.")
print(f"  Source table: {source_table_name}")
print(f"  Vector Search endpoint: {endpoint_name}")
print(f"  Index name: {full_index_name}")
print(f"  Primary key: {primary_key}")
print(f"  Embedding source column: {embedding_source_column}")
print(f"  Embedding model: {embedding_model_endpoint}")
print(f"  Pipeline type: {pipeline_type}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Vector Search Client

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()
print("Vector Search client initialized.")

spark.sql(f"ALTER TABLE {source_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check/Create Vector Search Endpoint

# COMMAND ----------

def endpoint_exists(client, endpoint_name: str) -> bool:
    """Check if a Vector Search endpoint exists."""
    try:
        endpoints = client.list_endpoints()
        endpoint_names = [ep.get("name") for ep in endpoints.get("endpoints", [])]
        return endpoint_name in endpoint_names
    except Exception as e:
        print(f"Error checking endpoints: {e}")
        return False


def wait_for_endpoint_ready(client, endpoint_name: str, timeout: int = 3600):
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
    raise TimeoutError(f"Endpoint '{endpoint_name}' did not become ready within {timeout} seconds.")

# COMMAND ----------

# Check if endpoint exists, create if not
if endpoint_exists(client, endpoint_name):
    print(f"Endpoint '{endpoint_name}' already exists.")
else:
    print(f"Creating endpoint '{endpoint_name}'...")
    client.create_endpoint(
        name=endpoint_name,
        endpoint_type="STANDARD"
    )
    print(f"Endpoint '{endpoint_name}' creation initiated.")

# Wait for endpoint to be ready
wait_for_endpoint_ready(client, endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check if Index Exists

# COMMAND ----------

def index_exists(client, endpoint_name: str, index_name: str) -> bool:
    """Check if a Vector Search index exists."""
    try:
        index = client.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )
        return index is not None
    except Exception as e:
        # Index doesn't exist or other error
        if "NOT_FOUND" in str(e) or "does not exist" in str(e).lower():
            return False
        print(f"Error checking index: {e}")
        return False

# COMMAND ----------

index_already_exists = index_exists(client, endpoint_name, full_index_name)
print(f"Index '{full_index_name}' exists: {index_already_exists}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Sync Index

# COMMAND ----------

if index_already_exists:
    # Index exists - trigger a sync
    print(f"Index '{full_index_name}' already exists. Triggering sync...")

    index = client.get_index(
        endpoint_name=endpoint_name,
        index_name=full_index_name
    )

    # Trigger sync for TRIGGERED pipeline type
    sync_result = index.sync()
    print(f"Sync triggered successfully.")
    print(f"Sync result: {sync_result}")

else:
    # Index doesn't exist - create new Delta Sync index
    print(f"Creating new Delta Sync index '{full_index_name}'...")

    index = client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        source_table_name=source_table_name,
        index_name=full_index_name,
        pipeline_type=pipeline_type,
        primary_key=primary_key,
        embedding_source_column=embedding_source_column,
        embedding_model_endpoint_name=embedding_model_endpoint,
    )

    print(f"Index creation initiated.")
    print(f"Index details: {index}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Index to be Ready (Optional)

# COMMAND ----------

def wait_for_index_ready(client, endpoint_name: str, index_name: str, timeout: int = 1800):
    """Wait for index to be ready (up to 30 minutes by default)."""
    print(f"Waiting for index '{index_name}' to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            index = client.get_index(
                endpoint_name=endpoint_name,
                index_name=index_name
            )
            status = index.describe().get("status", {}).get("ready", False)
            detailed_state = index.describe().get("status", {}).get("detailed_state", "UNKNOWN")

            if status:
                print(f"Index '{index_name}' is ready!")
                return True
            print(f"  Index status: ready={status}, state={detailed_state}")
        except Exception as e:
            print(f"  Error checking index: {e}")
        time.sleep(60)

    print(f"Warning: Index did not become ready within {timeout} seconds. It may still be processing.")
    return False

# COMMAND ----------

# Wait for the index to be ready
index_ready = wait_for_index_ready(client, endpoint_name, full_index_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Index Summary

# COMMAND ----------

try:
    index = client.get_index(
        endpoint_name=endpoint_name,
        index_name=full_index_name
    )
    index_info = index.describe()

    print("=" * 60)
    print("VECTOR SEARCH INDEX SUMMARY")
    print("=" * 60)
    print(f"Index name: {full_index_name}")
    print(f"Endpoint: {endpoint_name}")
    print(f"Source table: {source_table_name}")
    print(f"Primary key: {primary_key}")
    print(f"Embedding column: {embedding_source_column}")
    print(f"Status: {index_info.get('status', {})}")

    # Get row count if available
    num_rows = index_info.get("status", {}).get("num_rows", "N/A")
    print(f"Number of rows indexed: {num_rows}")

except Exception as e:
    print(f"Error getting index info: {e}")

# COMMAND ----------

print("\nVector Search index setup complete!")
print(f"You can now query the index using: {full_index_name}")
