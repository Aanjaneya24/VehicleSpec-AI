"""Streamlit interface for VehicleSpec AI — Vehicle Specification Extraction System."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import settings
from src.extractor import ExtractionError, extract_specifications, response_csv, response_json
from src.indexing import build_index
from src.retrieval import RetrievalError, index_status, retrieve


@st.cache_resource(show_spinner="Loading manual index…")
def _cached_retrieve(query: str, top_k: int):
    """Load the embedder and FAISS index once; cache them for the session."""
    return retrieve(query, top_k)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_index_status():
    return index_status()


st.set_page_config(
    page_title="VehicleSpec AI — Vehicle Specification Extraction System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom light-theme CSS styling for automotive/diagnostics engineering application
st.markdown(
    """
<style>
    /* Global Base */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Compact Application Header */
    .app-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .app-header-left {
        max-width: 75%;
    }
    .app-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 4px 0;
        letter-spacing: -0.01em;
    }
    .app-subtitle {
        color: #475569;
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.4;
    }
    .header-status-badge {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
    }
    .header-status-badge.ready {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .header-status-badge.not-ready {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Technical Specification Card */
    .spec-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        border-left: 3px solid #1e40af;
    }
    .spec-header-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid #f1f5f9;
    }
    .spec-type-tag {
        font-size: 0.72rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .spec-component-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }
    .spec-value-box {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e40af;
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 4px 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    
    /* Citation Metadata */
    .citation-group {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .citation-item {
        font-size: 0.8rem;
        color: #334155;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 3px 8px;
    }
    .citation-label {
        font-weight: 600;
        color: #64748b;
        margin-right: 4px;
    }

    /* Verbatim Evidence Box */
    .evidence-section {
        margin-top: 10px;
    }
    .evidence-heading {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .evidence-content {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #64748b;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 0.82rem;
        color: #1e293b;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1.5;
        word-break: break-word;
    }

    /* Restrained Banners */
    .status-banner-passed {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 16px;
        color: #166534;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .abstention-card {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 3px solid #d97706;
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 18px;
    }
    .abstention-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #92400e;
        margin-bottom: 6px;
    }
    .abstention-body {
        color: #78350f;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .abstention-meta {
        font-size: 0.8rem;
        color: #b45309;
        border-top: 1px solid #fde68a;
        padding-top: 6px;
    }

    /* Retrieved Passage Card */
    .passage-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .passage-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #475569;
        margin-bottom: 4px;
    }
    .passage-score {
        font-weight: 600;
        color: #1e40af;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 4px;
        padding: 1px 6px;
    }

    /* Form & Action Buttons */
    div[data-testid="stFormSubmitButton"] > button {
        background: #1e40af !important;
        color: #ffffff !important;
        border: 1px solid #1e3a8a !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        font-size: 0.9rem !important;
        transition: background-color 0.15s ease !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1d4ed8 !important;
    }
    
    /* Quick query buttons */
    .stButton > button {
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #334155 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 6px 10px !important;
    }
    .stButton > button:hover {
        border-color: #1e40af !important;
        color: #1e40af !important;
        background: #f8fafc !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar: PDF Indexing & System Status
with st.sidebar:
    st.markdown("### Manual Indexing")
    st.caption("Manage and inspect the indexed workshop manual.")

    available, status_message = _cached_index_status()
    if available:
        st.markdown(
            f"""
        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:10px 12px; margin-bottom:16px;">
            <div style="font-weight:600; color:#166534; font-size:0.85rem;">Index Status: Ready</div>
            <div style="color:#15803d; font-size:0.78rem; margin-top:2px;">{html.escape(status_message)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:6px; padding:10px 12px; margin-bottom:16px;">
            <div style="font-weight:600; color:#991b1b; font-size:0.85rem;">Index Status: Not Ready</div>
            <div style="color:#b91c1c; font-size:0.78rem; margin-top:2px;">{html.escape(status_message)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("**Index a Workshop Manual**")
    uploaded = st.file_uploader(
        "Upload manual (PDF)",
        type=["pdf"],
        help="Select a text-based workshop manual PDF.",
    )
    custom_path = st.text_input(
        "Local PDF file path",
        value=str(settings.default_pdf),
        help="Absolute or relative path to the manual PDF file.",
    )

    if st.button("Build / Rebuild Index", type="primary", use_container_width=True):
        if uploaded is None and not custom_path.strip():
            st.warning("Please specify a PDF file path or upload a PDF file.")
        elif uploaded is not None and uploaded.size == 0:
            st.error("The uploaded file is empty (0 bytes). Please upload a valid text-based PDF.")
        else:
            try:
                if uploaded is not None:
                    settings.data_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = Path(uploaded.name).name
                    pdf_path = settings.data_dir / safe_name
                    pdf_path.write_bytes(uploaded.getbuffer())
                else:
                    pdf_path = Path(custom_path.strip())

                progress_bar = st.progress(0, text="Initializing index build…")
                pages, chunks = build_index(
                    pdf_path,
                    progress=lambda value, text: progress_bar.progress(value, text=text),
                )
                _cached_retrieve.clear()
                _cached_index_status.clear()
                st.success(f"Index successfully built: {pages} text-bearing pages, {chunks} chunks.")
                st.rerun()
            except Exception as error:
                st.error(f"Indexing failed: {error}")

    st.divider()
    with st.expander("System Configuration & Pipeline"):
        st.markdown(
            f"""
        - **Embedding Model:** `{settings.embedding_model}`
        - **Groq LLM:** `{settings.groq_model}`
        - **Chunk Size / Overlap:** `{settings.chunk_size}` / `{settings.chunk_overlap}` chars
        - **Retrieval:** Hybrid (85% Semantic + 15% Lexical)
        - **Grounding Validation:** Verbatim evidence verification against cited manual passage
        """
        )

# Main Application Header
status_class = "ready" if available else "not-ready"
status_text = "Index: Ready" if available else "Index: Unavailable"
st.markdown(
    f"""
<div class="app-header">
    <div class="app-header-left">
        <h1 class="app-title">VehicleSpec AI</h1>
        <p class="app-subtitle">
            Automotive workshop manual specification extraction with source citations and verbatim evidence grounding.
        </p>
    </div>
    <div>
        <span class="header-status-badge {status_class}">{status_text}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Quick-Select Example Queries
st.markdown("<p style='font-size:0.82rem; font-weight:600; color:#475569; margin-bottom:6px;'>Quick-test queries:</p>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

if c1.button("Caliper Pin Torque", use_container_width=True, help="Test front brake caliper guide pin bolt torque extraction"):
    st.session_state["query_input"] = "What is the torque specification for the front brake caliper guide pin bolts?"
    st.rerun()

if c2.button("Brake Fluid Capacity", use_container_width=True, help="Test front disc brake fluid capacity extraction"):
    st.session_state["query_input"] = "What is the specified brake fluid capacity for the front disc brake?"
    st.rerun()

if c3.button("TPMS Part Number", use_container_width=True, help="Test TPMS module part number extraction"):
    st.session_state["query_input"] = "What is the part number for the tire pressure monitoring system module?"
    st.rerun()

if c4.button("Flux Capacitor (Abstain)", use_container_width=True, help="Test abstention behavior on unsupported specification query"):
    st.session_state["query_input"] = "What is the torque specification for the flux capacitor?"
    st.rerun()

# Query Form
default_question = st.session_state.get("query_input", "")

with st.form("extraction_form"):
    question = st.text_input(
        "Specification query",
        value=default_question,
        placeholder="e.g. What is the torque specification for the front brake caliper guide pin bolts?",
        help="Enter a natural language inquiry about component specifications in the manual.",
    )
    col_k, col_submit = st.columns([1, 2])
    with col_k:
        top_k = st.slider(
            "Retrieved passage count",
            min_value=3,
            max_value=10,
            value=min(max(settings.default_top_k, 3), 10),
            help="Number of candidate passages retrieved from the manual index.",
        )
    with col_submit:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Extract Specification", use_container_width=True, type="primary")

# Execution and Results Display
if submitted and not question.strip():
    st.warning("Please enter a specification query before extracting.")

if submitted and question.strip():
    try:
        with st.status("Executing extraction pipeline...", expanded=True) as status_box:
            status_box.update(label="Retrieving relevant manual passages...", state="running")
            passages = _cached_retrieve(question, top_k)
            status_box.update(label=f"Retrieved {len(passages)} passages. Executing extraction and evidence validation...", state="running")
            response = extract_specifications(question, passages, settings.groq_model)
            status_box.update(label="Extraction pipeline complete.", state="complete", expanded=False)

        # 1. Unsupported Query State (Abstention)
        if response.not_found:
            st.markdown(
                f"""
            <div class="abstention-card">
                <div class="abstention-title">No supporting evidence found</div>
                <div class="abstention-body">
                    {html.escape(response.note or "No supported specification was found in the retrieved manual passages.")}
                </div>
                <div class="abstention-meta">
                    <strong>Evidence validation:</strong> Abstained. The indexed manual passages contain no verifiable match for this specification.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # 2. Supported Grounded State
        else:
            result_count = len(response.results)
            st.markdown(
                f"""
            <div class="status-banner-passed">
                <strong>Evidence validation: passed</strong> &mdash; Found {result_count} specification{'s' if result_count > 1 else ''} supported by manual citations.
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Specification Result Cards
            for result in response.results:
                unit_display = f" {html.escape(result.unit)}" if result.unit else ""
                st.markdown(
                    f"""
                <div class="spec-card">
                    <div class="spec-header-row">
                        <div>
                            <div class="spec-type-tag">{html.escape(result.spec_type)}</div>
                            <div class="spec-component-title">{html.escape(result.component)}</div>
                        </div>
                        <div>
                            <span class="spec-value-box">{html.escape(result.value)}{unit_display}</span>
                        </div>
                    </div>
                    <div class="citation-group">
                        <span class="citation-item"><span class="citation-label">Source:</span>{html.escape(result.source)}</span>
                        <span class="citation-item"><span class="citation-label">Page:</span>{result.page}</span>
                        <span class="citation-item"><span class="citation-label">Section:</span>{html.escape(result.section or 'General')}</span>
                    </div>
                    <div class="evidence-section">
                        <div class="evidence-heading">Supporting Evidence (Verbatim)</div>
                        <div class="evidence-content">"{html.escape(result.evidence)}"</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Multi-view inspection tabs (Table / Raw JSON / Evidence)
        view_tabs = st.tabs(["Table View", "Raw JSON Payload", "Retrieved Evidence"])

        with view_tabs[0]:
            if response.results:
                table_data = [item.model_dump() for item in response.results]
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            else:
                st.info("No structured tabular records available for an abstained query.")

        with view_tabs[1]:
            st.json(response.model_dump())

        with view_tabs[2]:
            st.caption(f"Top {len(passages)} candidate passages retrieved from the manual index:")
            for rank, passage in enumerate(passages, start=1):
                st.markdown(
                    f"""
                <div class="passage-card">
                    <div class="passage-meta-row">
                        <span>
                            <strong>#{rank} &bull; {html.escape(passage.source)}</strong> &nbsp;
                            <span>(Page {passage.page})</span>
                        </span>
                        <span class="passage-score">
                            Hybrid Score: {passage.score:.3f}
                        </span>
                    </div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:2px;">
                        Section: {html.escape(passage.section or 'General')} &nbsp;|&nbsp;
                        Semantic: {passage.semantic_score:.3f} &nbsp;|&nbsp; Lexical: {passage.lexical_score:.3f}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.code(passage.text, language=None)

        # Download Buttons
        st.markdown("<p style='font-size:0.82rem; font-weight:600; color:#475569; margin-top:16px; margin-bottom:8px;'>Export Results:</p>", unsafe_allow_html=True)
        col_json, col_csv = st.columns(2)
        with col_json:
            st.download_button(
                "Download JSON",
                data=response_json(response),
                file_name="vehicle_specs.json",
                mime="application/json",
                use_container_width=True,
                help="Download structured JSON response including citations and metadata.",
            )
        with col_csv:
            st.download_button(
                "Download CSV",
                data=response_csv(response),
                file_name="vehicle_specs.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download tabular CSV export with component, value, unit, page, and evidence.",
            )

    except (RetrievalError, ExtractionError) as error:
        st.error(f"Extraction error: {error}")
    except Exception as error:
        st.error(f"Unexpected error: {error}")
