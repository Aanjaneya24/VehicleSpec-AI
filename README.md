# VehicleSpec AI — Vehicle Specification Extraction System

VehicleSpec AI is a traceable Retrieval-Augmented Generation (RAG) system engineered to extract documented automotive specifications from workshop service manuals. The system pairs hybrid dense/sparse passage retrieval with LLM extraction and deterministic grounding validation. To prevent hallucinations, the system strictly abstains when the indexed manual lacks verifiable evidence for a requested specification.

---

## 1. Project Overview & Problem Statement

Automotive service manuals are dense, multi-hundred-page technical documents containing critical torque ratings, fluid capacities, clearances, and part numbers. Standard LLMs hallucinate automotive specifications when queried directly, posing mechanical failure and safety risks. 

Developed for the Predii Technical Assessment, VehicleSpec AI solves this by:
* Indexing workshop manual PDFs with exact page-level and section-level provenance.
* Employing hybrid retrieval (dense vector embeddings combined with lexical term overlap) to retrieve relevant manual excerpts.
* Restricting extraction to retrieved passages using Groq-hosted LLMs with a strict schema.
* Enforcing algorithmic post-extraction grounding validation that verifies exact verbatim evidence quotes against the cited source chunk before presenting results.
* Abstaining cleanly whenever retrieved passages do not provide verifiable evidence.

---

## 2. Key Features & Supported Workflows

* **PDF Ingestion & Section Detection:** Extracts text from text-based PDFs via PyMuPDF, preserving 1-based page indices and hierarchical section headers (e.g., `SECTION 206-03: Front Disc Brake`).
* **Deterministic Chunking:** Employs layout-aware chunking with character-level boundaries and overlaps, ensuring multi-line specification tables are not arbitrarily truncated.
* **Hybrid Retrieval (85% Semantic + 15% Lexical):** Combines dense cosine similarity via FAISS inner-product indexing with sparse keyword overlap scoring to ensure acronyms, part numbers, and component names are ranked accurately.
* **Deterministic Evidence Grounding:** Validates that extracted values, units, and components are supported by verbatim quotes found in the retrieved manual chunks. Rejects fabricated or misattributed values.
* **Explicit Abstention:** Returns `not_found: true` with a clear explanation whenever no supporting evidence exists in the manual.
* **Dual Interface (CLI & Web UI):** Provides a command-line indexing tool (`build_index.py`) and a Streamlit engineering interface (`app.py`).
* **Multi-Format Export:** Generates standardized JSON payloads and CSV exports for both supported extractions and abstained queries.

---

## 3. Architecture & Data Flow

```
Workshop Manual PDF (e.g., data/manual.pdf)
     │
     ▼
[PyMuPDF Text Ingestion] ──► Extracts text, preserves 1-based page numbers
     │
     ▼
[Deterministic Chunking] ──► 1,500-char chunks, 220-char overlap, section tagging
     │
     ▼
[Sentence-Transformers] ──► all-MiniLM-L6-v2 normalized 384-d vectors
     │
     ▼
[FAISS + Metadata Store] ──► artifacts/manual.faiss & artifacts/chunks.jsonl
     │
     ▼
[Hybrid Retrieval Engine] ──► 0.85 * Semantic (FAISS IP) + 0.15 * Lexical Overlap
     │
     ▼  Top-K Candidate Passages
[Groq LLM Inference] ──► Temperature 0.0, JSON object mode (openai/gpt-oss-120b)
     │
     ▼
[Grounding Validation] ──► Verifies verbatim quote in cited chunk & validates values
     │
     ├─► Passed: Return SpecificationResult(s) with page/section citations
     └─► Failed / Missing: Abstain (not_found = True)
     │
     ▼
[Presentation & Exports] ──► Technical UI, Table View, Raw JSON, CSV Download
```

### Artifact Files
When an index is built, artifacts are stored in `artifacts/`:
* `artifacts/manual.faiss`: Flat inner-product FAISS vector index.
* `artifacts/chunks.jsonl`: JSON Lines file with chunk text, 1-based page numbers, source filename, and section metadata.
* `artifacts/index_manifest.json`: Index manifest storing source PDF metadata, SHA-256 hash, chunk count, and embedding model configuration.


---

## 4. Technology Stack

* **Language:** Python 3.10 – 3.13 (Python 3.11+ recommended)
* **PDF Extraction:** PyMuPDF (`fitz` >= 1.24)
* **Embedding Model:** Sentence-Transformers (`all-MiniLM-L6-v2`, 384 dimensions)
* **Vector Index:** FAISS-CPU (`faiss-cpu` >= 1.8)
* **LLM Inference:** Groq Python SDK (`groq` >= 0.11) running `openai/gpt-oss-120b` (or configured model)
* **Data Validation:** Pydantic (>= 2.7)
* **Data Processing:** Pandas (>= 2.2)
* **Web UI:** Streamlit (>= 1.39)
* **Testing:** Pytest (>= 8.0)

---

## 5. Prerequisites & Setup (macOS / Linux)

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13 installed.
* A text-based workshop manual PDF (e.g., the 2014 Ford F-150 Workshop Manual placed at `data/manual.pdf`). Scanned image PDFs without an OCR text layer are not supported.
* A valid Groq API key (free tier available at [console.groq.com](https://console.groq.com)).

### Installation
Clone the repository and initialize the virtual environment:

```bash
git clone <repository-url>
cd "Vehicle Specification Extraction System"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Environment Configuration

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:

```dotenv
# Required: Your Groq API key (never commit this key to version control)
GROQ_API_KEY=gsk_your_actual_key_here

# Optional: Pipeline defaults (shown with default fallback values)
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1500
CHUNK_OVERLAP=220
DEFAULT_TOP_K=6
```

> **Security Note:** `.env` is ignored by `.gitignore`. The application automatically redacts API keys from error messages, UI exceptions, and logs.

---

## 7. Running the Application

### Step 1: Build the Vector Index
Ensure your manual PDF is placed in `data/manual.pdf` (or provide a custom path):

```bash
# Build index using default path (data/manual.pdf)
python build_index.py

# Or specify a custom PDF path
python build_index.py /path/to/workshop_manual.pdf
```
*Expected output:* `Indexed 852 text-bearing pages into 1054 chunks.`

### Step 2: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### Index Management via Web UI
You can also upload a PDF directly or rebuild the index at runtime via the **Manual Indexing** sidebar control.

---

## 8. Supported & Unsupported Query Examples

The system is tested and benchmarked against the 2014 Ford F-150 Workshop Manual:

| Query Type | Query Text | Expected Behavior | Grounded Extraction |
| :--- | :--- | :--- | :--- |
| **Torque Specification** | *What is the torque specification for the front brake caliper guide pin bolts?* | **Supported** | Component: `Brake caliper guide pin bolts`<br>Value: `37 Nm` (27 lb-ft)<br>Citation: Page 636, Section 206-03 |
| **Fluid Capacity** | *What is the specified brake fluid capacity for the front disc brake?* | **Supported** | Component: `Front disc brake fluid`<br>Value: `1,420 ml` (3 pt)<br>Citation: Page 636, Section 206-03 |
| **Part Number** | *What is the part number for the tire pressure monitoring system module?* | **Supported** | Component: `TPMS module`<br>Value: `15K602`<br>Citation: Page 194, Section 204-04 |
| **Unsupported Query** | *What is the torque specification for the flux capacitor?* | **Abstained** | Returns `not_found: true` with notice: `"Flux capacitor torque specification not found in provided passages."` |

---

## 9. Citation Format & Export Formats

### JSON Response Schema (`vehicle_specs.json`)
The system outputs a strictly validated Pydantic model:

```json
{
  "results": [
    {
      "component": "Brake caliper guide pin bolts",
      "spec_type": "torque",
      "value": "37",
      "unit": "Nm",
      "source": "manual.pdf",
      "page": 636,
      "section": "SECTION 206-03: Front Disc Brake",
      "evidence": "Brake caliper guide pin bolts 37 27 —"
    }
  ],
  "not_found": false,
  "note": null
}
```

When abstaining, `results` is empty and `not_found` is `true`:
```json
{
  "results": [],
  "not_found": true,
  "note": "Flux capacitor torque specification not found in provided passages."
}
```

### CSV Export Format (`vehicle_specs.csv`)
CSV exports contain standard column headers matching the extraction schema:
```csv
component,spec_type,value,unit,source,page,section,evidence
Brake caliper guide pin bolts,torque,37,Nm,manual.pdf,636,SECTION 206-03: Front Disc Brake,"Brake caliper guide pin bolts 37 27 —"
```
For abstained queries, the CSV export retains valid column headers with no data rows, ensuring downstream pipeline compatibility.

---

## 10. Testing & Retrieval Evaluation

### Running Unit Tests
The test suite operates offline without requiring an active Groq API key or the full PDF manual (Groq API calls and embeddings are mocked via fixtures):

```bash
.venv/bin/python -m pytest -q
```
*Current test suite result:* **24 passed in ~2.8s**.

Test coverage includes:
* PyMuPDF text ingestion and 1-based page tracking (`test_pdf_ingest.py`).
* Section-aware deterministic chunking and overlap (`test_chunking.py`).
* FAISS indexing, metadata synchronization, and index invalidation checks (`test_indexing_retrieval.py`).
* Verbatim grounding validation, schema fallback, API key redaction, and CSV/JSON export serialization (`test_extractor.py`).

### Evaluating Retrieval Metrics
Evaluate Hit@K and Mean Reciprocal Rank (MRR) against manually transcribed, verified ground-truth labels (`evaluation/manual_verified_cases.json`):

```bash
.venv/bin/python evaluate_retrieval.py --top-k 6
```

**Benchmark Results on 2014 Ford F-150 Workshop Manual (852 pages, 1,054 chunks):**
* **Verified retrieval cases:** 3
* **Hit@6:** `1.000` (100%)
* **MRR@6:** `1.000`
* **Verified unsupported cases:** 1 (correctly abstained)

---

## 11. Known Limitations & Troubleshooting

### Known Limitations
* **Text-Based PDFs Only:** Ingestion uses PyMuPDF text extraction. Scanned PDFs without an embedded OCR text layer will yield 0 text-bearing pages.
* **Complex Multi-Column Tables:** Highly nested or non-standard table layouts where headers and cells are separated across multiple pages can require tuning `CHUNK_SIZE` and `CHUNK_OVERLAP`.
* **Single-Document Scope:** The current FAISS index holds one active document at a time. Uploading or building a new index replaces the active vector index.
* **Diagrams and Schematics:** Specifications embedded purely as raster images or vector diagrams without text callouts are not indexed.

### Troubleshooting
* **`GROQ_API_KEY is not set`**: Ensure `.env` exists in the project root and contains a valid key. The application will raise an explicit `ExtractionError`.
* **`Index out of date or incompatible`**: Occurs if `artifacts/manual.faiss` does not match the source document or embedding model recorded in `artifacts/index_manifest.json`. Rebuild the index using `python build_index.py`.
* **`The uploaded file is empty (0 bytes)`**: Verify that the uploaded PDF is not empty or corrupted.
* **Embedding Model Download Warnings**: On first run, Sentence-Transformers downloads model weights from Hugging Face. Subsequent runs load from local cache.

---

## 12. Repository Structure

```
Vehicle Specification Extraction System/
├── .env.example               # Template for environment configuration
├── .gitignore                 # Excludes secrets, virtualenvs, PDFs, and FAISS artifacts
├── README.md                  # Technical documentation and setup guide
├── app.py                     # Streamlit engineering web application
├── build_index.py             # CLI script to ingest PDF and generate FAISS index
├── evaluate_retrieval.py      # CLI script to evaluate Hit@K and MRR retrieval metrics
├── requirements.txt           # Production and test Python dependencies
├── evaluation/
│   └── manual_verified_cases.json  # Ground-truth benchmark cases from 2014 Ford manual
├── src/
│   ├── __init__.py
│   ├── chunking.py            # Section-aware deterministic chunking
│   ├── config.py              # Pydantic environment configuration
│   ├── extractor.py           # Groq extraction, schema parsing, and grounding validation
│   ├── indexing.py            # Sentence-Transformers embedding and FAISS index builder
│   ├── pdf_ingest.py          # PyMuPDF text and 1-based page extraction
│   └── retrieval.py           # Hybrid (dense + lexical) passage retrieval
└── tests/
    ├── conftest.py            # Pytest fixtures and mock classes
    ├── test_chunking.py       # Chunk boundary and section detection tests
    ├── test_extractor.py      # LLM response parsing, grounding, and security tests
    ├── test_indexing_retrieval.py # FAISS persistence, reload, and validation tests
    └── test_pdf_ingest.py     # PDF parsing and page tracking tests
```
