# ---------------------------------------------------------------------------
# config.py  –  All pipeline constants (replaces config.yaml)
# ---------------------------------------------------------------------------

# ── Unity Catalog location ──────────────────────────────────────────────────
CATALOG = "bircatalog"
SCHEMA = "pdf2"

# ── Volume paths ─────────────────────────────────────────────────────────────
SOURCE_VOLUME = "vol1"
SOURCE_VOLUME_SUBPATH = "source"
SOURCE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{SOURCE_VOLUME}/{SOURCE_VOLUME_SUBPATH}"

IMAGE_OUTPUT_VOLUME = "vol1"
IMAGE_OUTPUT_SUBPATH = "parsed_doc_images"
IMAGE_OUTPUT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{IMAGE_OUTPUT_VOLUME}/{IMAGE_OUTPUT_SUBPATH}"

# ── Table names (unqualified – the pipeline catalog/schema supplies the prefix) ──
GOLD_TABLE = "docs_gold"

# ── LLM endpoints ───────────────────────────────────────────────────────────
PII_MASKING_LLM = "databricks-claude-opus-4-6"

# ── Classification labels ───────────────────────────────────────────────────
DOC_TYPE_LABELS = "('hr', 'finance', 'research', 'engineering', 'support')"

# ── Prompts ─────────────────────────────────────────────────────────────────
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

# ── Vector Search ────────────────────────────────────────────────────────────
VS_ENDPOINT_NAME = "ka-f3925e58-vs-endpoint"
VS_INDEX_NAME = "docs_gold_index"
VS_PRIMARY_KEY = "chunk_id"
VS_EMBEDDING_SOURCE_COLUMN = "page_content_masked"
VS_EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"
VS_PIPELINE_TYPE = "TRIGGERED"
