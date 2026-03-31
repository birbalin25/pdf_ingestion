# Databricks notebook source
# MAGIC %pip install pdfplumber PyPDF2 reportlab PyMuPDF --quiet
# MAGIC %restart_python

# COMMAND ----------

# =============================================================================
# PII MASKING PIPELINE FOR PDF DOCUMENTS - DATABRICKS NOTEBOOK
# =============================================================================
#
# Prerequisites:
#   1. Cluster libraries: pdfplumber, PyPDF2, reportlab, fitz (PyMuPDF)
#   2. Foundation Model endpoint accessible via ai_query()
#   3. Upload input PDF to DBFS or Unity Catalog Volume
#
# Root Cause of Text Override Issue:
#   - Masked text can be wider/narrower than original text
#   - reportlab drawString doesn't clip to bounding box
#   - Font mismatch between PDF embedded fonts and reportlab fonts
#
# Fix:
#   - Measure masked text width using reportlab's stringWidth
#   - Scale font size DOWN if masked text overflows the bounding box
#   - Clip drawing to the exact bounding box using canvas.clipPath
#   - Add padding buffer between adjacent redaction zones
# =============================================================================



# ── CONFIGURATION ────────────────────────────────────────────────────────────
INPUT_PDF  = "/Volumes/bircatalog/pdf2/vol1/source/pdf2_HR.pdf"      # Change this
OUTPUT_PDF = "/Volumes/bircatalog/pdf2/vol1/test/pdf2_HR_masked.pdf"  # Change this
MODEL_ENDPOINT = "databricks-claude-opus-4-6"  # Or your serving endpoint

PII_CATEGORIES = [
    "PERSON_NAME", "EMAIL", "PHONE_NUMBER", "SSN", "CREDIT_CARD",
    "ADDRESS", "DATE_OF_BIRTH", "DRIVER_LICENSE", "PASSPORT_NUMBER",
    "IP_ADDRESS", "BANK_ACCOUNT", "MEDICAL_RECORD_NUMBER", "AGE",
    "ORGANIZATION", "NATIONAL_ID"
]
# ─────────────────────────────────────────────────────────────────────────────

import json, re, io, logging, copy
import pdfplumber
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import white, black
from reportlab.pdfbase.pdfmetrics import stringWidth
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pii_masker")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 : EXTRACT TEXT + WORD POSITIONS FROM PDF
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf(pdf_path: str) -> list:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            words = page.extract_words(
                x_tolerance=2, y_tolerance=2,
                extra_attrs=["fontname", "size"]
            )
            word_list = []
            for w in words:
                word_list.append({
                    "text":     w["text"],
                    "x0":       float(w["x0"]),
                    "y0":       float(w["top"]),
                    "x1":       float(w["x1"]),
                    "y1":       float(w["bottom"]),
                    "fontname": w.get("fontname", "Helvetica"),
                    "fontsize": float(w.get("size", 10)),
                })
            pages.append({
                "page_number": idx,
                "width":       float(page.width),
                "height":      float(page.height),
                "words":       word_list,
                "full_text":   page.extract_text() or "",
            })
    logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
    return pages

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 : DETECT PII VIA DATABRICKS ai_query()
# ═══════════════════════════════════════════════════════════════════════════════

def _build_prompt(text: str) -> str:
    cats = ", ".join(PII_CATEGORIES)
    return f"""You are a strict PII-detection engine. You MUST find every piece of Personally Identifiable Information in the text.

CATEGORIES: {cats}

RETURN FORMAT — a pure JSON array, nothing else:
[
  {{"original": "<exact text as it appears>", "masked": "<masked version>", "category": "<CATEGORY>"}},
  ...
]

MASKING RULES — the masked value MUST be the EXACT same character length as original:
  • Replace each alphanumeric character with *
  • Keep ALL spaces, hyphens, slashes, dots, @, parentheses, and other punctuation in their exact positions
  • Replace only street number with * from address. The number of * should match the street number.
  • Mask all digits with * except for the last 4 digits of credit card, driver license, passport number, Phone Number
  • Mask email id till @. keep domain.com intact.

  • Examples:
      "John Smith"          → "**** *****"
      "john.doe@email.com"  → "****.***@email.com"
      "+1 (555) 123-4567"   → "+* (***) ***-4567"
      "123-45-6789"         → "***-**-6789"
      "03/15/1990"          → "**/**/****"
      "192.168.1.1"         → "***.***.*.* "
  • The length of "masked" MUST equal the length of "original". This is critical.

If NO PII is found return exactly: []

TEXT:
\"\"\"
{text}
\"\"\"

JSON:"""


def detect_pii(spark, pages_data: list) -> dict:
    rows = [(p["page_number"], p["full_text"], _build_prompt(p["full_text"]))
            for p in pages_data if p["full_text"].strip()]
    if not rows:
        return {}

    schema = StructType([
        StructField("page_number", IntegerType(), False),
        StructField("text", StringType(), False),
        StructField("prompt", StringType(), False),
    ])
    df = spark.createDataFrame(rows, schema)
    df.createOrReplaceTempView("_pii_input")

    result_df = spark.sql(f"""
        SELECT
            page_number,
            text,
            ai_query('{MODEL_ENDPOINT}', prompt) AS pii_response
        FROM _pii_input
    """)

    pii_map = {}
    for row in result_df.collect():
        entities = _parse_response(row["pii_response"])
        # Post-process: force masked length == original length
        for e in entities:
            e["masked"] = _force_same_length(e["original"], e["masked"])
        pii_map[row["page_number"]] = entities
        logger.info(f"Page {row['page_number']}: {len(entities)} PII entities detected")
    return pii_map


def _parse_response(response: str) -> list:
    if not response:
        return []
    try:
        m = re.search(r'\[.*\]', response, re.DOTALL)
        if m:
            items = json.loads(m.group())
            return [
                {"original": str(e["original"]),
                 "masked":   str(e["masked"]),
                 "category": e.get("category", "UNKNOWN")}
                for e in items
                if isinstance(e, dict) and "original" in e and "masked" in e
            ]
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse error: {exc}")
    return []


def _force_same_length(original: str, masked: str) -> str:
    """
    Guarantee masked string is EXACTLY the same length as original.
    Rebuild character-by-character from original, replacing alphanums with *.
    This is the safety net — even if the LLM gets it wrong, we fix it.
    """
    if len(masked) == len(original):
        return masked
    # Rebuild from scratch to guarantee correctness
    result = []
    for ch in original:
        if ch.isalnum():
            result.append("*")
        else:
            result.append(ch)
    return "".join(result)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 : MAP PII ENTITIES → WORD BOUNDING BOXES (PRECISE)
# ═══════════════════════════════════════════════════════════════════════════════

def map_pii_to_words(page_data: dict, entities: list) -> list:
    """
    Match each PII entity to word-level bounding boxes.
    Split the masked text proportionally across matched words so each
    word-level redaction box contains only its share of the masked text.
    This prevents ANY overflow into adjacent content.
    """
    words = page_data["words"]
    redactions = []
    used_indices = set()  # prevent double-redaction of same word

    for ent in entities:
        pii_tokens = ent["original"].split()
        masked_full = ent["masked"]
        if not pii_tokens:
            continue

        for i in range(len(words)):
            # Skip if starting word already used
            if i in used_indices:
                continue

            matched = []
            ok = True
            for j, pt in enumerate(pii_tokens):
                idx = i + j
                if idx >= len(words) or idx in used_indices:
                    ok = False
                    break
                w_clean = re.sub(r'[^\w@.\-]', '', words[idx]["text"])
                p_clean = re.sub(r'[^\w@.\-]', '', pt)
                if w_clean.lower() != p_clean.lower():
                    ok = False
                    break
                matched.append((idx, words[idx]))

            if ok and matched:
                # Mark indices as used
                for idx, _ in matched:
                    used_indices.add(idx)

                # Split masked text into per-word chunks
                masked_tokens = masked_full.split()

                # If token counts don't match, distribute evenly
                if len(masked_tokens) != len(matched):
                    # Fallback: assign proportional substrings
                    masked_tokens = _split_masked_proportional(
                        masked_full, [w["text"] for _, w in matched]
                    )

                for k, (idx, word_data) in enumerate(matched):
                    masked_word = masked_tokens[k] if k < len(masked_tokens) else "*" * len(word_data["text"])
                    redactions.append({
                        "x0":       word_data["x0"],
                        "y0":       word_data["y0"],
                        "x1":       word_data["x1"],
                        "y1":       word_data["y1"],
                        "fontname": word_data["fontname"],
                        "fontsize": word_data["fontsize"],
                        "category": ent["category"],
                        "masked":   masked_word,
                        "original_word": word_data["text"],
                    })
                break  # Move to next entity after first match

    logger.info(f"Page {page_data['page_number']}: {len(redactions)} redaction boxes mapped")
    return redactions


def _split_masked_proportional(masked_full: str, original_words: list) -> list:
    """Split masked string into chunks proportional to original word lengths."""
    total_len = sum(len(w) for w in original_words)
    if total_len == 0:
        return ["*" * len(w) for w in original_words]

    # Remove spaces from masked_full, then distribute
    masked_chars = masked_full.replace(" ", "")
    result = []
    pos = 0
    for i, w in enumerate(original_words):
        share = len(w)
        chunk = masked_chars[pos:pos + share]
        # Pad or trim to exact word length
        if len(chunk) < share:
            chunk += "*" * (share - len(chunk))
        elif len(chunk) > share:
            chunk = chunk[:share]
        result.append(chunk)
        pos += share
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 : BUILD OUTPUT PDF — PRECISE CLIPPED REDACTION
# ═══════════════════════════════════════════════════════════════════════════════

REPORTLAB_FONT = "Helvetica"  # Safe cross-platform font

def _fit_text_in_box(masked_text: str, box_width: float, box_height: float,
                     target_fontsize: float) -> float:
    """
    Calculate the largest font size <= target_fontsize that fits
    masked_text within box_width. Also cap at box_height.
    """
    fs = min(target_fontsize, box_height - 1)
    if fs <= 0:
        fs = target_fontsize

    # Reduce font size until text fits in box width
    while fs > 1:
        tw = stringWidth(masked_text, REPORTLAB_FONT, fs)
        if tw <= box_width:
            return fs
        fs -= 0.25

    return max(fs, 1)


def redact_pdf(input_path: str, redactions_by_page: dict, output_path: str):
    """
    For each redaction:
      1. Save canvas state
      2. Clip to the EXACT bounding box of the original word
      3. Draw white rectangle (erase original)
      4. Draw masked text fitted within the box
      5. Restore canvas state (clip is removed)

    This guarantees masked text NEVER bleeds outside the word boundary.
    """
    reader = PdfReader(input_path)
    writer = PdfWriter()

    for pg_num in range(len(reader.pages)):
        orig_page = reader.pages[pg_num]
        pw = float(orig_page.mediabox.width)
        ph = float(orig_page.mediabox.height)

        rects = redactions_by_page.get(pg_num, [])
        if not rects:
            writer.add_page(orig_page)
            continue

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(pw, ph))

        for r in rects:
            # ── Convert coordinates: pdfplumber (top-left) → reportlab (bottom-left)
            box_x0     = r["x0"]
            box_y_bot  = ph - r["y1"]          # bottom edge in reportlab coords
            box_w      = r["x1"] - r["x0"]
            box_h      = r["y1"] - r["y0"]

            if box_w <= 0 or box_h <= 0:
                continue

            masked_text = r["masked"]
            orig_fs     = r["fontsize"]

            # Calculate font size that fits within the box
            fitted_fs = _fit_text_in_box(masked_text, box_w, box_h, orig_fs)

            # ── SAVE STATE → CLIP → DRAW → RESTORE ──
            c.saveState()

            # Define clipping rectangle — nothing drawn outside this box
            clip_path = c.beginPath()
            clip_path.rect(box_x0, box_y_bot, box_w, box_h)
            c.clipPath(clip_path, stroke=0, fill=0)

            # 1) White-out original text
            c.setFillColorRGB(1, 1, 1)
            c.setStrokeColorRGB(1, 1, 1)
            c.rect(box_x0, box_y_bot, box_w, box_h, fill=True, stroke=False)

            # 2) Draw masked text — vertically centered, left-aligned
            c.setFillColorRGB(0, 0, 0)
            c.setFont(REPORTLAB_FONT, fitted_fs)
            text_y = box_y_bot + (box_h - fitted_fs) / 2.0 + fitted_fs * 0.15
            c.drawString(box_x0, text_y, masked_text)

            c.restoreState()  # Clip boundary is now removed

        c.save()
        buf.seek(0)

        overlay_page = PdfReader(buf).pages[0]
        orig_page.merge_page(overlay_page)
        writer.add_page(orig_page)

    with open(output_path, "wb") as f:
        writer.write(f)
    logger.info(f"Masked PDF saved → {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 : VALIDATION — VERIFY NO TEXT LEAKS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_output(input_path: str, output_path: str, pii_map: dict):
    """
    Re-extract text from the output PDF and verify that none of the
    original PII strings appear in the masked document.
    """
    issues = []
    with pdfplumber.open(output_path) as pdf:
        for page in pdf.pages:
            pg_num = page.page_number - 1  # pdfplumber is 1-indexed
            text = (page.extract_text() or "").lower()
            entities = pii_map.get(pg_num, [])
            for ent in entities:
                orig = ent["original"].lower().strip()
                if len(orig) > 2 and orig in text:
                    issues.append(f"Page {pg_num}: PII still visible → '{ent['original']}'")

    if issues:
        logger.warning(f"⚠️  Validation found {len(issues)} potential leaks:")
        for issue in issues:
            logger.warning(f"   {issue}")
    else:
        logger.info("✅ Validation passed — no original PII text found in output PDF")
    return len(issues) == 0

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(input_pdf: str, output_pdf: str):
    spark = SparkSession.builder.getOrCreate()

    # 1. Extract
    logger.info("━━━ STEP 1/5: Extracting text & positions from PDF ━━━")
    pages = extract_pdf(input_pdf)

    # 2. Detect PII
    logger.info("━━━ STEP 2/5: Detecting PII via ai_query() ━━━")
    pii_map = detect_pii(spark, pages)

    if not pii_map or all(len(v) == 0 for v in pii_map.values()):
        logger.info("No PII detected. Output PDF is identical to input.")
        import shutil
        shutil.copy2(input_pdf, output_pdf)
        return output_pdf

    # 3. Map PII → bounding boxes (word-level, per-word split)
    logger.info("━━━ STEP 3/5: Mapping PII to word bounding boxes ━━━")
    redactions_by_page = {}
    for page in pages:
        pn = page["page_number"]
        if pn in pii_map and pii_map[pn]:
            redactions_by_page[pn] = map_pii_to_words(page, pii_map[pn])

    total = sum(len(v) for v in redactions_by_page.values())
    logger.info(f"Total redaction boxes: {total}")

    # 4. Produce masked PDF
    logger.info("━━━ STEP 4/5: Generating masked PDF ━━━")
    redact_pdf(input_pdf, redactions_by_page, output_pdf)

    # 5. Validate
    logger.info("━━━ STEP 5/5: Validating output ━━━")
    validate_output(input_pdf, output_pdf, pii_map)

    logger.info("Pipeline complete.")
    return output_pdf


# ── Execute ──────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
output = run_pipeline(INPUT_PDF, OUTPUT_PDF)
print(f"\nMasked PDF written to: {output}")