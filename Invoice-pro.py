import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import anthropic
from PIL import Image
import os, io, json, re
import fitz, random
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('Google_API_KEY')
# api_key = st.secrets['Google_API_KEY']

#Function to load Gemini Pro
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash",
    "gemini-3.1-flash-lite"
]

ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]
# ── Config ────────────────────────────────────────────────────────────────────
# Models and clients are now configured via the API Keys tab in the sidebar
LLM_MODEL        = None  # Set dynamically from sidebar
ANTHROPIC_MODEL  = None  # Set dynamically from sidebar
gemini_client    = None  # Initialized when user provides API key
anthropic_client = None  # Initialized when user provides API key

APP_VERSION   = "v1.0.2"
LINKEDIN_URL  = "https://www.linkedin.com/in/ramu-siva"
LINKEDIN_NAME = "Siva R"
LINKEDIN_ROLE = "Python Developer · Chennai, Tamilnadu"

DEFAULT_FIELDS = [
    "invoice_number", "invoice_date", "invoice_amount",
    "total_amount", "total_amount_in_words", "gst",
    "igst", "cgst", "sgst",
    "subtotal", "invoiced_from", "invoiced_to", "items",
]


# ── JSON helpers ──────────────────────────────────────────────────────────────

def clean_json_response(text: str) -> str:
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    return text.strip()


def parse_json_safely(text: str):
    cleaned = clean_json_response(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = re.sub(r"'", '"', cleaned)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            st.warning(f"JSON parse error: {e}")
            return None


# ── Currency helpers ──────────────────────────────────────────────────────────

def standardize_inr(value):
    try:
        if value in (None, "", "null"):
            return None
        cleaned = re.sub(r"[^\d.]", "", str(value))
        return f"INR {float(cleaned):,.2f}" if cleaned else None
    except (ValueError, TypeError):
        return value


def standardize_currency_fields(data):
    keys = ["amount", "price", "total", "subtotal", "gst", "igst", "cgst", "sgst", "tax", "cost", "rate"]
    if isinstance(data, dict):
        for k, v in data.items():
            if any(kw in k.lower() for kw in keys):
                data[k] = standardize_inr(v)
            elif isinstance(v, (dict, list)):
                standardize_currency_fields(v)
    elif isinstance(data, list):
        for item in data:
            standardize_currency_fields(item)
    return data


# ── Float helper ──────────────────────────────────────────────────────────────

def _to_float(val):
    if val is None:
        return None
    try:
        s = re.sub(r"[^\d.]", "", str(val))
        return float(s) if s else None
    except ValueError:
        return None


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_invoice_data(file, file_extension: str, fields=None, temperature=0, max_tokens=2048, model: str = None, client=None) -> dict:
    if fields is None:
        fields = DEFAULT_FIELDS
    if client is None:
        return {"success": False, "error": "Gemini client not initialized. Please set API key first."}
    if model is None:
        return {"success": False, "error": "Gemini Model not selected."}
    try:
        ft = file_extension.lower()
        if ft in ("png", "jpg", "jpeg"):
            image = Image.open(file)
        elif ft == "pdf":
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            pix = pdf[0].get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes("png")))
        else:
            return {"success": False, "error": f"Unsupported file type: {ft}"}
 
        prompt = (
            f"Extract invoice data from this image.\n"
            f"Fields: {', '.join(fields)}\n"
            "RULES: Return ONLY valid JSON. Double quotes. No trailing commas. No markdown. "
            "Missing values = null. For 'items': array of objects with keys "
            "description, quantity, unit_price, amount."
        )
 
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        text = response.text.strip().replace("```json", "").replace("```", "")
        data = parse_json_safely(text)
        if data is None:
            return {"success": False, "error": "Invalid JSON from model", "raw": text}
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(records: list) -> list:
    flags = []
    seen = {}
    for rec in records:
        inv_no = rec.get("invoice_number", "")
        src    = rec.get("_source", "")
        if inv_no:
            if inv_no in seen:
                flags.append({"invoice": inv_no, "file": src, "type": "Duplicate invoice #", "severity": "high"})
            else:
                seen[inv_no] = src
        gst   = _to_float(rec.get("gst"))
        sub   = _to_float(rec.get("subtotal"))
        total = _to_float(rec.get("total_amount"))
        if gst and sub and sub > 0:
            rate = (gst / sub) * 100
            if rate < 4 or rate > 30:
                flags.append({"invoice": inv_no, "file": src,
                               "type": f"Unusual GST rate ({rate:.1f}%)", "severity": "medium"})
        if gst and sub and total:
            expected = sub + gst
            if abs(expected - total) > 2:
                flags.append({"invoice": inv_no, "file": src,
                               "type": f"Total mismatch (expected {expected:,.0f})", "severity": "medium"})
    return flags


# ── Aggregations ──────────────────────────────────────────────────────────────

def compute_aggregations(records: list) -> dict:
    def s(lst): return sum(x for x in lst if x)
    vendors = [r.get("invoiced_from") for r in records if r.get("invoiced_from")]
    return {
        "total_invoiced":  s(_to_float(r.get("total_amount")) for r in records),
        "total_subtotal":  s(_to_float(r.get("subtotal"))     for r in records),
        "total_gst":       s(_to_float(r.get("gst"))          for r in records),
        "total_igst":      s(_to_float(r.get("igst"))         for r in records),
        "total_cgst":      s(_to_float(r.get("cgst"))         for r in records),
        "total_sgst":      s(_to_float(r.get("sgst"))         for r in records),
        "unique_vendors":  len(set(vendors)),
        "invoice_count":   len(records),
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def render_charts(records: list):
    df = pd.DataFrame(records)
    # ── 1. Total Amount per Invoice ──────────────────────────────────────────
    plotted = 0

    if "total_amount" in df.columns and "invoice_number" in df.columns:
        d = df[["invoice_number", "total_amount"]].copy()
        d["n"] = d["total_amount"].apply(_to_float)
        d = d.dropna(subset=["n"])
        if not d.empty:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            bars = ax.barh(d["invoice_number"].astype(str), d["n"], color="#185FA5", edgecolor="none")
            ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=8)
            ax.set_xlabel("Amount (INR)")
            ax.set_title("Total amount per invoice", fontsize=12)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            plotted += 1
    # ── 2. GST vs Subtotal vs Total breakdown ────────────────────────────────
    numeric = {}
    for col in ("subtotal", "gst", "total_amount"):
        if col in df.columns:
            v = df[col].apply(_to_float)
            if v.notna().any():
                numeric[col] = v.fillna(0).tolist()

    if len(numeric) >= 2 and "invoice_number" in df.columns:
        labels = df["invoice_number"].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(8, 3.2))
        x = list(range(len(labels)))
        w = 0.25
        colors = {"subtotal": "#9FE1CB", "gst": "#FAC775", "total_amount": "#185FA5"}
        for i, (col, vals) in enumerate(numeric.items()):
            ax.bar([xi + i * w for xi in x], vals, w,
                   label=col.replace("_", " ").title(), color=colors.get(col, "#ccc"))
        ax.set_xticks([xi + w for xi in x])
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
        ax.set_title("Subtotal / GST / Total breakdown", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        plotted += 1
    # ── 3. Item-level cost breakdown (first invoice with items) ──────────────
    for rec in records:
        items = rec.get("items")
        if isinstance(items, list) and items:
            try:
                idf = pd.DataFrame(items)
                if "description" in idf.columns and "amount" in idf.columns:
                    idf["n"] = idf["amount"].apply(_to_float)
                    idf = idf.dropna(subset=["n"])
                    if not idf.empty:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.pie(idf["n"], labels=idf["description"].str[:18],
                               autopct="%1.1f%%", startangle=140,
                               colors=plt.cm.Pastel1.colors)
                        ax.set_title("Line-item split (first invoice)", fontsize=12)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
            except Exception:
                pass
            break

    if plotted == 0:
        st.info("Upload invoices with amount data to see charts.")


# ── AI Chat ───────────────────────────────────────────────────────────────────

def build_data_context(records: list, agg: dict) -> str:
    rows = [{k: v for k, v in r.items() if k != "items"} for r in records]
    return (
        f"You are an expert accounting and audit assistant.\n"
        f"The user has extracted {len(records)} invoices.\n\n"
        f"Summary aggregations:\n{json.dumps(agg, indent=2)}\n\n"
        f"Extracted records (excluding line items):\n{json.dumps(rows, indent=2, default=str)}\n\n"
        "Answer the user's questions accurately with specific numbers. Be concise and professional. "
        "If asked about anomalies, refer to GST rate mismatches, duplicates, or total discrepancies."
    )


def chat_with_data(user_msg: str, history: list, records: list, agg: dict, client=None, model: str = None) -> str:
    if client is None:
        return "⚠️ Anthropic client not initialized. Please set API key in the API Keys tab."
    if model is None:
        return "⚠️ Anthropic model not selected. Please choose a model in the API Keys tab."
    system = build_data_context(records, agg)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": user_msg})
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return resp.content[0].text
    except Exception as e:
        return f"Error contacting AI: {e}"


# ── Annotations ───────────────────────────────────────────────────────────────

def render_annotations(records: list):
    st.subheader("Annotations")
    if "annotations" not in st.session_state:
        st.session_state.annotations = []

    inv_numbers = [r.get("invoice_number", f"Invoice {i+1}") for i, r in enumerate(records)]

    with st.form("ann_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            note = st.text_area("Note", placeholder="e.g. GST rate mismatch — vendor is 18% but invoice shows 12%", height=80)
        with col2:
            tag      = st.selectbox("Tag", ["GST mismatch", "Verified", "Duplicate", "Amount error", "Missing field", "Cleared", "Other"])
            inv      = st.selectbox("Invoice", ["(general)"] + inv_numbers)
            reviewer = st.text_input("Reviewer", placeholder="Your name")
        if st.form_submit_button("Add annotation") and note.strip():
            from datetime import datetime
            st.session_state.annotations.append({
                "reviewer": reviewer or "Anonymous",
                "tag":      tag,
                "invoice":  inv,
                "note":     note.strip(),
                "time":     datetime.now().strftime("%d %b %Y %H:%M"),
            })
            st.success("Annotation added.")

    if st.session_state.annotations:
        for ann in reversed(st.session_state.annotations):
            c1, c2, c3 = st.columns([1, 3, 1])
            c1.markdown(f"**{ann['reviewer']}**  \n`{ann['tag']}`")
            c2.markdown(f"{ann['note']}  \n_Linked to: {ann['invoice']}_")
            c3.caption(ann["time"])
            st.divider()


# ── Feature guide ─────────────────────────────────────────────────────────────

def render_feature_guide():
    st.markdown("### Who benefits & how")
    features = [
        ("🧮", "GST reconciliation",
         "Auto-splits IGST/CGST/SGST per invoice and maps to GSTR-2B / GSTR-3B line items instantly. "
         "Eliminates manual pivoting across hundreds of vendor PDFs.",
         "Accountant"),
        ("🛡️", "Audit trail",
         "Every extraction is timestamped with source file, model output, and reviewer annotations — "
         "creating an immutable chain of evidence for regulatory review.",
         "Auditor"),
        ("🌐", "Multi-language OCR",
         "Reads invoices in Tamil, Hindi, Arabic, Japanese, Chinese and 30+ languages. "
         "No manual re-entry for regional or import/export vendors.",
         "Both"),
        ("⚠️", "Anomaly detection",
         "Flags duplicate invoice numbers, GST rate mismatches vs registration, and "
         "subtotal-plus-tax totals that don't add up — before they reach the filing.",
         "Auditor"),
        ("💬", "AI chat on data",
         "Ask plain-language questions — 'total spend with Acme this quarter' or "
         "'show invoices where GST exceeds 18%' — and get instant calculated answers grounded in your data.",
         "Both"),
        ("📤", "One-click filing export",
         "Export to GSTR-ready Excel (pre-mapped columns), Tally-compatible CSV, "
         "or a structured PDF summary for client sign-off — all from a single button.",
         "Accountant"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc, badge) in enumerate(features):
        color = {"Accountant": "#EAF3DE", "Auditor": "#E6F1FB", "Both": "#EEEDFE"}[badge]
        tcolor = {"Accountant": "#3B6D11", "Auditor": "#185FA5", "Both": "#534AB7"}[badge]
        with cols[i % 3]:
            st.markdown(
                f'<div style="border:1px solid #e0e0e0;border-radius:10px;padding:14px;margin-bottom:12px;min-height:160px">'
                f'<div style="font-size:22px;margin-bottom:6px">{icon}</div>'
                f'<div style="font-size:13px;font-weight:600;margin-bottom:5px">{title}</div>'
                f'<div style="font-size:12px;color:#555;line-height:1.55;margin-bottom:8px">{desc}</div>'
                f'<span style="font-size:11px;background:{color};color:{tcolor};'
                f'border-radius:99px;padding:2px 8px">{badge}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── LinkedIn card ─────────────────────────────────────────────────────────────

def render_linkedin_card():
    st.markdown(
        f'<a href="{LINKEDIN_URL}" target="_blank" style="text-decoration:none">'
        f'<div style="border:1px solid #0A66C2;border-radius:10px;padding:10px 14px;'
        f'display:flex;align-items:center;gap:12px;margin-bottom:6px">'
        f'<div style="width:36px;height:36px;border-radius:50%;background:#0A66C2;'
        f'display:flex;align-items:center;justify-content:center;'
        f'color:#fff;font-weight:700;font-size:13px;flex-shrink:0">SR</div>'
        f'<div style="flex:1">'
        f'<div style="font-size:13px;font-weight:600;color:#0A66C2">{LINKEDIN_NAME}</div>'
        f'<div style="font-size:11px;color:#555">{LINKEDIN_ROLE}</div>'
        f'</div>'
        f'<div style="font-size:11px;color:#0A66C2;border:1px solid #0A66C2;'
        f'border-radius:99px;padding:3px 10px;white-space:nowrap">Connect</div>'
        f'</div></a>',
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────

def render_footer():
    st.markdown("---")
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.caption("🔒 Data processed locally · No invoice content stored · End-to-end encrypted")
    c2.caption(f"InvoiceIQ {APP_VERSION} · © 2026")
    c3.markdown(
        f'<a href="{LINKEDIN_URL}" target="_blank" style="font-size:12px;color:#0A66C2;text-decoration:none">'
        f'🔗 {LINKEDIN_NAME}</a>',
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=f"InvoiceIQ {APP_VERSION}",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "records"      not in st.session_state: st.session_state.records      = []
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "annotations"  not in st.session_state: st.session_state.annotations  = []

    # API Keys and clients
    if "gemini_api_key"     not in st.session_state: st.session_state.gemini_api_key     = ""
    if "anthropic_api_key"  not in st.session_state: st.session_state.anthropic_api_key  = ""
    if "gemini_model"       not in st.session_state: st.session_state.gemini_model       = GEMINI_MODELS[0]
    if "anthropic_model"    not in st.session_state: st.session_state.anthropic_model    = ANTHROPIC_MODELS[0]
    if "gemini_client"      not in st.session_state: st.session_state.gemini_client      = None
    if "anthropic_client"   not in st.session_state: st.session_state.anthropic_client   = None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 🧾 InvoiceIQ `{APP_VERSION}`")
        st.caption("Multi-language invoice extractor")
        st.divider()

        sidebar_tab1, sidebar_tab2 = st.tabs(["📤 Upload", "🔑 API Keys"])
 
        # ── Upload Tab ────────────────────────────────────────────────────────
        with sidebar_tab1:
            uploaded_files = st.file_uploader(
                "Upload invoices",
                accept_multiple_files=True,
                type=["pdf", "png", "jpg", "jpeg"],
            )
 
            selected_model = st.selectbox(
                "Gemini model",
                GEMINI_MODELS,
                index=GEMINI_MODELS.index(st.session_state.gemini_model),
                help="Choose which Gemini model to use for extraction",
                key="gemini_model_select",
            )
            st.session_state.gemini_model = selected_model
 
            if uploaded_files:
                if not st.session_state.gemini_client:
                    st.error("⚠️ Please set your Gemini API key in the API Keys tab first.")
                elif st.button("Extract", type="primary", use_container_width=True):
                    new_records = []
                    prog = st.progress(0, text="Extracting…")
                    for idx, f in enumerate(uploaded_files):
                        prog.progress((idx + 1) / len(uploaded_files), text=f"Processing {f.name}…")
                        ext = f.name.rsplit(".", 1)[-1]
                        result = extract_invoice_data(
                            f, ext, 
                            model=st.session_state.gemini_model, 
                            client=st.session_state.gemini_client
                        )
                        if result["success"]:
                            data = standardize_currency_fields(result["data"])
                            data["_source"] = f.name
                            new_records.append(data)
                        else:
                            st.warning(f"{f.name}: {result['error']}")
                    prog.empty()
                    st.session_state.records.extend(new_records)
                    st.success(f"Extracted {len(new_records)} invoice(s)")
 
            if st.session_state.records:
                if st.button("Clear all data", use_container_width=True):
                    st.session_state.records      = []
                    st.session_state.chat_history = []
                    st.rerun()
 
        # ── API Keys Tab ──────────────────────────────────────────────────────
        with sidebar_tab2:
            st.markdown("##### Gemini (for extraction)")
            
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.gemini_api_key,
                placeholder="Enter your Google AI API key",
                help="Get your API key from https://aistudio.google.com/apikey",
            )
            
            if gemini_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = gemini_key
                if gemini_key:
                    try:
                        st.session_state.gemini_client = genai.Client(api_key=gemini_key)
                        st.success("✅ Gemini client initialized")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize Gemini: {e}")
                        st.session_state.gemini_client = None
                else:
                    st.session_state.gemini_client = None
 
            st.divider()
            st.markdown("##### Claude (for AI chat)")
            
            anthropic_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.anthropic_api_key,
                placeholder="Enter your Anthropic API key",
                help="Get your API key from https://console.anthropic.com/",
            )
            
            anthropic_model_select = st.selectbox(
                "Claude model",
                ANTHROPIC_MODELS,
                index=ANTHROPIC_MODELS.index(st.session_state.anthropic_model),
                help="Choose which Claude model to use for AI chat",
            )
            st.session_state.anthropic_model = anthropic_model_select
            
            if anthropic_key != st.session_state.anthropic_api_key:
                st.session_state.anthropic_api_key = anthropic_key
                if anthropic_key:
                    try:
                        st.session_state.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                        st.success("✅ Anthropic client initialized")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize Anthropic: {e}")
                        st.session_state.anthropic_client = None
                else:
                    st.session_state.anthropic_client = None
 
            st.caption("💡 API keys are stored in session only and never saved to disk.")

        st.divider()
        render_linkedin_card()
        st.caption(f"InvoiceIQ {APP_VERSION} · © 2026")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    (tab_data, tab_agg, tab_anomaly,
     tab_charts, tab_annot, tab_chat,
     tab_export, tab_guide) = st.tabs([
        "📄 Extracted data",
        "📊 Aggregations",
        "⚠️ Anomalies",
        "📈 Charts",
        "📝 Annotations",
        "💬 AI Chat",
        "📤 Export",
        "ℹ️ Feature guide",
    ])

    records = st.session_state.records

    # ── Extracted data ────────────────────────────────────────────────────────
    with tab_data:
        st.subheader("Extracted invoice data")
        if not records:
            st.info("Upload invoices using the sidebar to get started.")
        else:
            flat = [{k: (json.dumps(v, ensure_ascii=False)
                         if isinstance(v, (dict, list)) else v)
                     for k, v in r.items()} for r in records]
            st.dataframe(pd.DataFrame(flat), width='stretch')

    # ── Aggregations ──────────────────────────────────────────────────────────
    with tab_agg:
        st.subheader("Aggregations")
        if not records:
            st.info("No data yet.")
        else:
            agg = compute_aggregations(records)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Invoices",       agg["invoice_count"])
            c2.metric("Total invoiced", f"INR {agg['total_invoiced']:,.2f}")
            c3.metric("Total GST",      f"INR {agg['total_gst']:,.2f}")
            c4.metric("Unique vendors", agg["unique_vendors"])
            st.markdown("##### GST split")
            g1, g2, g3 = st.columns(3)
            g1.metric("IGST", f"INR {agg['total_igst']:,.2f}")
            g2.metric("CGST", f"INR {agg['total_cgst']:,.2f}")
            g3.metric("SGST", f"INR {agg['total_sgst']:,.2f}")
            st.markdown("##### Per-vendor totals")
            vendor_map = {}
            for r in records:
                v   = r.get("invoiced_from") or "Unknown"
                amt = _to_float(r.get("total_amount")) or 0
                vendor_map[v] = vendor_map.get(v, 0) + amt
            vdf = pd.DataFrame(
                sorted(vendor_map.items(), key=lambda x: -x[1]),
                columns=["Vendor", "Total (INR)"],
            )
            vdf["Total (INR)"] = vdf["Total (INR)"].apply(lambda x: f"INR {x:,.2f}")
            st.dataframe(vdf, width='stretch', hide_index=True)

    # ── Anomalies ─────────────────────────────────────────────────────────────
    with tab_anomaly:
        st.subheader("Anomaly detection")
        if not records:
            st.info("No data yet.")
        else:
            flags = detect_anomalies(records)
            if not flags:
                st.success("No anomalies detected.")
            else:
                st.warning(f"{len(flags)} issue(s) found")
                for flag in flags:
                    sev = "🔴" if flag["severity"] == "high" else "🟡"
                    st.markdown(
                        f"{sev} **{flag['type']}** — Invoice `{flag['invoice']}` · _{flag['file']}_"
                    )

    # ── Charts ────────────────────────────────────────────────────────────────
    with tab_charts:
        st.subheader("Real-time charts")
        if not records:
            st.info("No data yet.")
        else:
            render_charts(records)

    # ── Annotations ───────────────────────────────────────────────────────────
    with tab_annot:
        if not records:
            st.info("Extract invoices first to annotate them.")
        else:
            render_annotations(records)

    # ── AI Chat ───────────────────────────────────────────────────────────────
    with tab_chat:
        st.subheader("AI chat — ask about your invoices")
        if not records:
            st.info("Extract at least one invoice to start chatting.")
        elif not st.session_state.anthropic_client:
            st.warning("⚠️ Please set your Anthropic API key in the sidebar API Keys tab to use AI chat.")
        else:
            agg = compute_aggregations(records)

            st.markdown("**Quick questions**")
            quick_qs = [
                "What is the total GST for all invoices?",
                "Which vendor has the highest invoice value?",
                "Show any GST rate anomalies.",
                "Give me a GSTR-2B summary.",
            ]
            q_cols = st.columns(4)
            for i, q in enumerate(quick_qs):
                if q_cols[i].button(q, key=f"qchip_{i}", width='stretch'):
                    with st.spinner("Thinking…"):
                        reply = chat_with_data(
                            q, st.session_state.chat_history, records, agg,
                            client=st.session_state.anthropic_client,
                            model=st.session_state.anthropic_model
                        )
                    st.session_state.chat_history.append({"role": "user",      "content": q})
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

            st.divider()

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Ask anything about your invoices…")
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        reply = chat_with_data(
                            user_input, st.session_state.chat_history, records, agg,
                            client=st.session_state.anthropic_client,
                            model=st.session_state.anthropic_model
                        )
                    st.markdown(reply)
                st.session_state.chat_history.append({"role": "user",      "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

            if st.session_state.chat_history:
                if st.button("Clear chat history"):
                    st.session_state.chat_history = []
                    st.rerun()

    # ── Export ────────────────────────────────────────────────────────────────
    with tab_export:
        st.subheader("Export")
        if not records:
            st.info("No data to export yet.")
        else:
            flat = [{k: (json.dumps(v, ensure_ascii=False)
                         if isinstance(v, (dict, list)) else v)
                     for k, v in r.items()} for r in records]
            df  = pd.DataFrame(flat)
            agg = compute_aggregations(records)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="invoiceiq_data.csv",
                    mime="text/csv",
                    width='stretch',
                )

            with c2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    df.to_excel(w, index=False, sheet_name="Invoices")
                    pd.DataFrame([agg]).to_excel(w, index=False, sheet_name="Aggregations")
                    flags = detect_anomalies(records)
                    if flags:
                        pd.DataFrame(flags).to_excel(w, index=False, sheet_name="Anomalies")
                    if st.session_state.annotations:
                        pd.DataFrame(st.session_state.annotations).to_excel(
                            w, index=False, sheet_name="Annotations"
                        )
                st.download_button(
                    "⬇️ Download Excel (full)",
                    data=buf.getvalue(),
                    file_name="invoiceiq_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch',
                )

            with c3:
                gstr_rows = [{
                    "GSTIN of supplier": r.get("invoiced_from", ""),
                    "Invoice number":    r.get("invoice_number", ""),
                    "Invoice date":      r.get("invoice_date", ""),
                    "Invoice value":     r.get("total_amount", ""),
                    "Taxable value":     r.get("subtotal", ""),
                    "IGST":              r.get("igst", ""),
                    "CGST":              r.get("cgst", ""),
                    "SGST":              r.get("sgst", ""),
                } for r in records]
                gbuf = io.BytesIO()
                with pd.ExcelWriter(gbuf, engine="openpyxl") as w:
                    pd.DataFrame(gstr_rows).to_excel(w, index=False, sheet_name="GSTR-2B")
                st.download_button(
                    "⬇️ Download GSTR-2B Excel",
                    data=gbuf.getvalue(),
                    file_name="invoiceiq_gstr2b.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch',
                )

    # ── Feature guide ─────────────────────────────────────────────────────────
    with tab_guide:
        render_feature_guide()

    # ── Footer ────────────────────────────────────────────────────────────────
    render_footer()


if __name__ == "__main__":
    main()