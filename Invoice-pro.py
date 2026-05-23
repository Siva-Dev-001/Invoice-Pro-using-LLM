import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image
import os, io, json, re
import fitz, random
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('Google_API_KEY')
# api_key = st.secrets['Google_API_KEY']

# Initialize Gemini AI client
client = genai.Client()

#Function to load Gemini Pro
models = [
    # "gemini-2.5-flash",
    # "gemini-2.5-pro",
    # "gemini-2-flash-lite",
    "gemini-3.5-flash"
]
LLM_MODEL = random.choice(models)

DEFAULT_FIELDS = [
    "invoice_number",
    "invoice_date",
    "invoice_amount",
    "total_amount",
    "total_amount_in_words",
    "gst",
    "subtotal",
    "invoiced_from",
    "invoiced_to",
    "items",
]
 
 
# ── JSON helpers ─────────────────────────────────────────────────────────────
 
def clean_json_response(text: str) -> str:
    """Strip markdown fences and control characters."""
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    return text.strip()
 
 
def parse_json_safely(text: str):
    cleaned = clean_json_response(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        st.warning(f"JSON parse error: {e}")
        return None
 
 
def repair_json(text: str) -> str:
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r"'", '"', text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text
 
 
def safe_json_loads(text: str):
    repaired = repair_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        st.error(f"JSON repair failed: {e}")
        return None
 
 
# ── Currency helpers ──────────────────────────────────────────────────────────
 
def standardize_inr(value) -> str | None:
    try:
        if value in (None, "", "null"):
            return None
        cleaned = (
            str(value)
            .replace(",", "")
            .replace("₹", "")
            .replace("INR", "")
            .strip()
        )
        return f"INR {float(cleaned):,.2f}"
    except (ValueError, TypeError):
        return value
 
 
def standardize_currency_fields(data):
    currency_keywords = ["amount", "price", "total", "subtotal", "gst", "tax", "cost", "rate"]
    if isinstance(data, dict):
        for key, value in data.items():
            if any(k in key.lower() for k in currency_keywords):
                data[key] = standardize_inr(value)
            elif isinstance(value, (dict, list)):
                standardize_currency_fields(value)
    elif isinstance(data, list):
        for item in data:
            standardize_currency_fields(item)
    return data
 
 
# ── Extraction ────────────────────────────────────────────────────────────────
 
def extract_invoice_data(
    file,
    file_extension: str,
    fields: list[str] | None = None,
    system: str | None = None,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> dict:
    """Extract structured invoice data using Gemini. Returns a result dict."""
 
    if fields is None:
        fields = DEFAULT_FIELDS
 
    try:
        file_type = file_extension.lower()
 
        if file_type in ("png", "jpg", "jpeg"):
            image = Image.open(file)
 
        elif file_type == "pdf":
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            pix = pdf[0].get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes("png")))
 
        else:
            return {"success": False, "error": f"Unsupported file type: {file_type}"}
 
        prompt = f"""
Extract invoice data from this image.
 
Fields to extract:
{", ".join(fields)}
 
STRICT RULES:
- Return ONLY valid JSON — no markdown, no backticks, no comments
- Use double quotes for all keys and string values
- No trailing commas
- Missing values must be null (not empty string)
- For "items", return a JSON array of objects with keys: description, quantity, unit_price, amount
"""
 
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            response_mime_type="application/json",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
 
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=[prompt, image],
            config=config,
        )
 
        text_response = response.text.strip()
        # Belt-and-suspenders cleanup even though we asked for application/json
        text_response = text_response.replace("```json", "").replace("```", "")
 
        extracted_data = parse_json_safely(text_response)
 
        if extracted_data is None:
            return {
                "success": False,
                "error": "Invalid JSON returned by model",
                "raw_response": text_response,   # kept for debugging
            }
 
        # ✅ BUG FIX 1: return extracted_data on the success path
        return {"success": True, "data": extracted_data}
 
    except Exception as e:
        return {"success": False, "error": str(e)}
 
 
# ── Chart helpers ─────────────────────────────────────────────────────────────
 
def _inr_to_float(val) -> float | None:
    """Convert 'INR 1,234.56' or raw number string to float."""
    if val is None:
        return None
    try:
        cleaned = re.sub(r"[^\d.]", "", str(val))
        return float(cleaned) if cleaned else None
    except ValueError:
        return None
 
 
def render_realtime_charts(records: list[dict]) -> None:
    """Render live charts that update as invoices are added."""
 
    st.subheader("📊 Real-time Invoice Analytics")
 
    df = pd.DataFrame(records)
 
    # ── 1. Total Amount per Invoice ──────────────────────────────────────────
    if "total_amount" in df.columns and "invoice_number" in df.columns:
        amounts = df[["invoice_number", "total_amount"]].copy()
        amounts["amount_num"] = amounts["total_amount"].apply(_inr_to_float)
        amounts = amounts.dropna(subset=["amount_num"])
 
        if not amounts.empty:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            bars = ax.barh(
                amounts["invoice_number"].astype(str),
                amounts["amount_num"],
                color="#4F8EF7",
                edgecolor="none",
            )
            ax.bar_label(bars, fmt="₹{:,.0f}", padding=4, fontsize=9)
            ax.set_xlabel("Amount (INR)")
            ax.set_title("Total Amount per Invoice")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
 
    # ── 2. GST vs Subtotal vs Total breakdown ────────────────────────────────
    numeric_cols = {}
    for col in ("subtotal", "gst", "total_amount"):
        if col in df.columns:
            vals = df[col].apply(_inr_to_float)
            if vals.notna().any():
                numeric_cols[col] = vals.fillna(0).tolist()
 
    if len(numeric_cols) >= 2 and "invoice_number" in df.columns:
        inv_labels = df["invoice_number"].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        x = range(len(inv_labels))
        width = 0.25
        colors = {"subtotal": "#A8D8A8", "gst": "#FFD580", "total_amount": "#4F8EF7"}
        for i, (col, vals) in enumerate(numeric_cols.items()):
            ax.bar([xi + i * width for xi in x], vals, width, label=col.replace("_", " ").title(), color=colors.get(col, "#ccc"))
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(inv_labels, rotation=20, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}"))
        ax.set_title("Subtotal / GST / Total Breakdown")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
 
    # ── 3. Item-level cost breakdown (first invoice with items) ──────────────
    for rec in records:
        items = rec.get("items")
        if isinstance(items, list) and items:
            try:
                item_df = pd.DataFrame(items)
                if "description" in item_df.columns and "amount" in item_df.columns:
                    item_df["amount_num"] = item_df["amount"].apply(_inr_to_float)
                    item_df = item_df.dropna(subset=["amount_num"])
                    if not item_df.empty:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.pie(
                            item_df["amount_num"],
                            labels=item_df["description"].str[:20],
                            autopct="%1.1f%%",
                            startangle=140,
                            colors=plt.cm.Pastel1.colors,
                        )
                        ax.set_title("Line-Item Cost Split (first invoice)")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
            except Exception:
                pass
            break  # only first invoice with items
 
 
# ── Main app ──────────────────────────────────────────────────────────────────
 
def main():
    st.set_page_config(page_title="Invoice Extractor", layout="wide")
    st.title("🧾 Multi-Language Invoice Extractor")
    st.caption("Powered by Gemini · Supports PDF, PNG, JPG")
 
    uploaded_files = st.file_uploader(
        "Upload one or more invoice files",
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg"],
    )
 
    if not uploaded_files:
        st.info("Upload at least one invoice to get started.")
        return
 
    records: list[dict] = []
    failed: list[str] = []
 
    progress = st.progress(0, text="Extracting invoices…")
 
    for idx, file in enumerate(uploaded_files):
        progress.progress((idx + 1) / len(uploaded_files), text=f"Processing {file.name}…")
 
        file_extension = file.name.rsplit(".", 1)[-1]
        result = extract_invoice_data(file, file_extension)
 
        if result.get("success"):
            # ✅ BUG FIX 2: read from result["data"], not result["raw_response"]
            data = result["data"]
            data = standardize_currency_fields(data)
            data["_source_file"] = file.name       # handy provenance column
            records.append(data)
        else:
            failed.append(f"**{file.name}**: {result.get('error', 'unknown error')}")
            st.warning(f"⚠️ {file.name} — {result.get('error')}")
 
    progress.empty()
 
    if failed:
        with st.expander("❌ Failed extractions"):
            for msg in failed:
                st.markdown(msg)
 
    if not records:
        st.error("No invoices could be extracted.")
        return
 
    # Flatten nested dicts for display (items stay as stringified JSON)
    flat_records = []
    for rec in records:
        flat = {}
        for k, v in rec.items():
            flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
        flat_records.append(flat)
 
    df = pd.DataFrame(flat_records)
    st.success(f"✅ Extracted {len(records)} invoice(s)")
    st.dataframe(df, width='stretch')
 
    # ── Real-time charts ─────────────────────────────────────────────────────
    render_realtime_charts(records)
 
    # ── Export ───────────────────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns(2)
 
    with col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name="invoice_data.csv",
            mime="text/csv",
        )
 
    with col2:
        # ✅ BUG FIX 3: write to a BytesIO buffer, not to disk, so we have bytes to serve
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Invoices")
        excel_bytes = excel_buffer.getvalue()
 
        st.download_button(
            "⬇️ Download Excel",
            data=excel_bytes,
            file_name="invoice_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
 
 
if __name__ == "__main__":
    main()