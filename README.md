# Multi‑Modal Medical Assistant (Unofficial)

You can follow the details of app here from this video
https://drive.google.com/file/d/18nAMFV3y-Kr0JLwqpgWGhJYANnNMrz9S/view?usp=sharing


A Streamlit app that turns **documents + images** into **chat‑ready context** using OCR, semantic/fixed chunking, and vector search. It combines:

- **OCR / Layout extraction**: Azure Document Intelligence (prebuilt‑layout), Google Document AI (optional), EasyOCR (fallback for images)
- **Chunking**: fixed character chunks + semantic chunks (paragraphs & tables)
- **Embeddings & Vector DBs**: FAISS (local), Azure AI Search (vector), Pinecone (optional)
- **LLM**: OpenAI (GPT‑4o) for post‑processing OCR + image understanding + chat
- **UI**: Streamlit with a sidebar for model/temperature/history and file upload controls

> ⚕️ The UI and examples are phrased around *medical* documents/images, but the pipeline is generic and works for any domain.

---

## Key Features

1. **Upload & Analyze** PDF (text/scan) and an optional accompanying **medical image**.
2. **OCR & Layout** via Azure Document Intelligence → paragraphs & tables extracted; outputs also re‑chunked as fixed character windows.
3. **Image Understanding** via GPT‑4o Vision (base64 data URL), prompted by your sidebar text prompt.
4. **Hybrid Chunking** → semantic (paragraphs/tables) **plus** fixed (RecursiveCharacterTextSplitter) to mitigate each method’s weaknesses.
5. **Vectorization** to **FAISS** (local) or **Azure AI Search** (cloud) (Pinecone optional).
6. **RAG‑style Chat**: retrieves top‑K chunks and injects them into the system message; OCR text can be **post‑processed (denoised)** by an LLM before use.
7. **Session History** persisted to `chat_history.json` (simple JSON) with a *Clear Chat* button.
8. **Vector DB lifecycle**: create, save locally, load, delete (applies to FAISS; Azure/Pinecone create indexes and upsert on the fly).

---

## Architecture

```
[PDF/Image Upload]
     │
     ├── Azure Document Intelligence (prebuilt-layout)  ─┐
     │                                                   │   → semantic chunks (paragraphs + tables)
     ├── Google Document AI (optional, alternate OCR) ───┤
     │                                                   └─→ fixed chunks from full content
     └── EasyOCR (images fallback)  → text                

[Chunking]  semantic + fixed → [Embeddings: OpenAI] → [Vector DB: FAISS | Azure AI Search | Pinecone]

[Chat] user prompt → retrieve top‑K similar chunks → compose system message → OpenAI chat → stream response
```

---

## Why Hybrid Chunking?

- **Fixed chunks** (e.g., 1000 chars with 10% overlap) are robust but may cut sentences and lose structure.
- **Semantic chunks** preserve logical boundaries (paragraphs/tables) but may produce **small or uneven granularity** and sometimes underperform for tightly scoped queries.
- **Hybrid = both**: you get recall from fixed windows + precision from semantic units.

---

## Prerequisites

- Python **3.10+** (recommended)
- Accounts/credentials where applicable:
  - **OpenAI** API key (access to `gpt‑4o` for image understanding and `chat.completions`)
  - **Azure Document Intelligence** endpoint & key
  - **Azure AI Search** endpoint & admin key (if you’ll use cloud vector search)
  - **Pinecone** API key (optional)
  - **Google Document AI** service account (optional alternative OCR)

> 💡 GPU is **not required**, but `easyocr` pulls in `torch`; CPU wheels are fine for most docs. For large‑scale OCR, prefer Azure DI / Google Document AI.

---

## Project Layout (suggested)

```
multi_modal_assistant/
├─ app.py                # your provided Streamlit script
├─ requirements.txt      # see bottom of this README
├─ chat_history.json     # created at runtime (optional)
└─ .streamlit/
   └─ secrets.toml       # Streamlit secrets (recommended)
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If `faiss-cpu` fails on Apple Silicon, ensure you’re on Python 3.10/3.11 and try `pip install faiss-cpu --no-cache-dir`.

---

## Configuration

This app loads credentials from **Streamlit Secrets** (`st.secrets[...]`) and optionally from `.env`.

### 1) Streamlit secrets (recommended)

Create `.streamlit/secrets.toml`:

```toml
# OpenAI
OPENAI_API_KEY = "sk-..."

# Azure Document Intelligence
AZURE_DI_ENDPOINT = "https://<your-di-resource>.cognitiveservices.azure.com/"
AZURE_DI_KEY = "<your-di-key>"

# Azure AI Search (Vector Search)
AI_SEARCH_END = "https://<your-search>.search.windows.net"
AI_SEARCH_API = "<your-search-admin-key>"

# Pinecone (optional)
PINECONE_APIKEY = "<your-pinecone-key>"
```

> The code expects these exact secret keys. Adjust your secrets file accordingly.

### 2) Optional `.env`

A plain `.env` file in the project root will be loaded (though **secrets** are preferred for Streamlit Cloud):

```env
OPENAI_API_KEY=sk-...
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json
```

### 3) Google Document AI (optional)

If you intend to use the `document_layout_analysis_google(...)` helper:

- Create a **service account** with *Document AI API* access.
- Download its JSON credentials and set `GOOGLE_APPLICATION_CREDENTIALS` to the absolute path.
- You’ll need: `project_id`, `processor_id`, and (optionally) `location` (e.g., `us`).

### 4) Azure AI Search (vector index)

If you use `create_vector_database_azure(...)`, configure an index (the code uses `medical-assistant-index`).

- Enable **Vector Search** on your Search service.
- Make sure your vector field’s **dimensions** match your embedding model (OpenAI’s defaults are typically 1536/3072 depending on model).
- The sample code upserts texts via LangChain’s `AzureSearch` vectorstore wrapper.

### 5) Pinecone (optional)

Create an index named `medical-assistant-index` compatible with your embedding dimension. The helper shows how to upsert via LangChain’s Pinecone vectorstore wrapper.

---

## Running the app

```bash
streamlit run app.py
```

Open the local URL shown in your terminal.

---

## How to Use

1. **Sidebar controls**

   - **LLMs Parameter Changer**
     - *temperature*: slider (0–1)
     - *model\_name*: select box (default options include `gpt-4o`, `gpt-4.1`, Gemini/Llama/Mistral/Claude placeholders; the code ultimately calls OpenAI’s chat endpoint)
     - *max history length*: trims chat memory to N most recent turns
   - **Uploads**
     - *medical\_perscribtion*: PDF (your primary document)
     - *medical\_image*: PNG/JPEG/WEBP image (optional)
     - *prompt*: A textual instruction for the **image** (e.g., “Describe key findings in this chest X‑ray.”)
   - **Actions**
     - *Show The Comment*: Runs image → GPT‑4o vision to produce a description; caches it in session
     - *Create a VectorDatabase*: Runs OCR + chunking → builds vector index (Azure AI Search in the sample) and also indexes cached image description (if any)
     - *Save / Load / Delete VectorDatabase*: FAISS local persistence helpers

2. **Main area**

   - *Clear chat* button wipes history.
   - Chat with the assistant. If a vector DB is available, your query retrieves top‑K chunks → injected into a system message → streamed response.

> Tip: You can inspect the **“Last Query For relevant context”** sidebar text area to see which chunks were used.

---

## Important Implementation Notes

- **OCR Post‑processing**: `post_processors_LLMs` sends the raw OCR to GPT‑4o with the instruction *“Improve readability; do NOT summarize.”* This helps correct noisy OCR and makes RAG more effective.
- **Retrieval**: `retrieve_relevant_context(query, vec_db, k=10)` does ANN search (KNN) and returns `Document` chunks for context.
- **FAISS vs Cloud Vectors**: The demo primarily uses **Azure AI Search** in `create_vector_database_azure` but also includes FAISS create/save/load/delete. Pinecone is a stub example.
- **Session State**: `st.session_state.crtdata` holds the current vector store; `st.session_state.desc` caches the last image description; `st.session_state.messages` holds chat history.

---

## Known Limitations & TODOs

- **Import compatibility**: The script mixes newer LangChain modules (`langchain_core`) with older import paths (`from langchain.embeddings import OpenAIEmbeddings`). If you see import errors, either:

  1. Pin LangChain packages to versions that still re‑export `OpenAIEmbeddings`, **or**
  2. Update the import in code to `from langchain_openai import OpenAIEmbeddings` and ensure `langchain-openai` is installed.

- **Azure Search schema**: The index must exist and be vector‑enabled with a field matching your embedding **dimension**. The helper doesn’t create the schema.

- **Pinecone helper** is illustrative and assumes an index named `medical-assistant-index` exists.

- **Image data URL**: In `get_image_description`, the MIME prefix should be `data:image/jpeg;base64,` (semicolon after `jpeg`). The code currently uses a colon—update if you encounter 400s from the API.

- `` is imported but not wired into the UI.

- **Error handling** is intentionally minimal for clarity; add guards and user‑friendly messages in production.

---

## Troubleshooting

- `` → install `langchain-openai` and change the import, or pin LangChain to a version that re‑exports it.
- ``** install issues** → use Python 3.10/3.11, ensure `pip>=23`, and try `--no-cache-dir`.
- **Azure DI 401/403** → verify endpoint region matches key; check you used `AZURE_DI_ENDPOINT` base URL (ending with `.cognitiveservices.azure.com`).
- **Azure Search 400 on upsert** → your index schema likely mismatches the embedding vector dimension.
- **OpenAI 404/429/401** → confirm model access (`gpt‑4o`), quotas, and billing.
- **EasyOCR slow on CPU** → prefer Azure DI / Google Document AI for large PDFs; use EasyOCR only for small images or as a fallback.

---

## Security & Cost Notes

- **Never hard‑code keys**; keep them in `secrets.toml` / environment variables.
- **OCR & LLM calls incur cost**. Consider size limits and chunk‑before‑embed to reduce token usage.
- **PII/PHI**: For medical documents, ensure you comply with local regulations and your cloud providers’ compliance programs.

---

## License

No license specified. Add one if you intend to distribute.

---

## requirements.txt

Paste the following into `requirements.txt`:

```txt
# Core UI & env
streamlit>=1.34.0
python-dotenv>=1.0.1

# OpenAI (new SDK)
openai>=1.30.0

# OCR & CV
easyocr>=1.7.1
torch>=2.3.0
opencv-python-headless>=4.10.0.84

# PDF & text
PyPDF2>=3.0.1

# LangChain ecosystem (see README for compatibility notes)
langchain>=0.2.6
langchain-community>=0.2.6
langchain-core>=0.2.6
langchain-openai>=0.1.7

# Vector DBs
faiss-cpu>=1.7.4
azure-search-documents>=11.6.0
pinecone>=4.0.0

# Azure Document Intelligence
azure-ai-documentintelligence>=1.0.0
azure-core>=1.30.0

# Google Document AI (optional)
google-cloud-documentai>=2.24.0
google-api-core>=2.17.0

# Misc
pydantic>=2.7.0
typing-extensions>=4.11.0
streamlit-feedback>=0.1.3
```

> If your environment can’t satisfy some pins (e.g., corporate mirrors), relax the `>=` constraints. For Apple Silicon, you may prefer `opencv-python` over `opencv-python-headless` if you need GUI windows locally.
