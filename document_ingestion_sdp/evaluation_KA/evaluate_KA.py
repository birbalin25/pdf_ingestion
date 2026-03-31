# Databricks notebook source
import mlflow
import json
from mlflow.genai.scorers import (
    Correctness,
    RetrievalSufficiency,
    RetrievalGroundedness,
    RelevanceToQuery,
    Guidelines,
    scorer,
)

# COMMAND ----------

dbutils.widgets.text("eval_csv_volume_path", "", "Eval CSV volume path")
dbutils.widgets.text("ka-endpoint", "", "KA Endpoint")


eval_csv_volume_path = dbutils.widgets.get("eval_csv_volume_path")
ka_endpoint = dbutils.widgets.get("ka-endpoint")

print(f"eval_csv_volume_path: {eval_csv_volume_path}")
print(f"ka-endpoint: {ka_endpoint}")

# COMMAND ----------

predict_fn = mlflow.genai.to_predict_fn(f"endpoints:/{ka_endpoint}")

# COMMAND ----------

df = spark.read.csv(eval_csv_volume_path, header=True)
data_dict = df.collect()
qa_pairs = [row.asDict() for row in data_dict]

print(json.dumps(qa_pairs, indent=2))

# COMMAND ----------

eval_data = [
    {
        "inputs": {
            "input": [
                {"role": "user", "content": qa["question"]}
            ]
        },
        "expectations": {
            "expected_response": qa["answer"],
        },
    }
    for qa in qa_pairs
]

print(json.dumps(eval_data, indent=2))

# COMMAND ----------

@scorer
def concise_response(inputs, outputs, expectations=None, traces=None) -> str:
    """
    Return 'yes' if response is reasonably concise, otherwise 'no'.
    Adjust key 'response' to match your predict_fn output schema.
    """
    text = ""
    if isinstance(outputs, dict):
        text = outputs.get("response") or outputs.get("output") or ""
    else:
        text = str(outputs)
    word_count = len(str(text).split())
    return "yes" if word_count <= 80 else "no"

scorers = [
  
    Correctness(),
    RetrievalSufficiency(),

    RetrievalGroundedness(),
    RelevanceToQuery(),

    Guidelines(
        name="global_style",
        guidelines=[
            "The response must be in English.",
            "The response must be helpful and not evasive.",
        ],
    ),

    concise_response,
]

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=scorers,
)


# COMMAND ----------


print("Aggregate metrics:", results.metrics)
eval_df = results.tables["eval_results"]
display(eval_df)

# COMMAND ----------

