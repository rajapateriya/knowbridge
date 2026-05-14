# KnowBridge — RAG-Powered Knowledge Assistant

KnowBridge is a lightweight **Retrieval-Augmented Generation (RAG)** app that turns a set of documents into a grounded, searchable knowledge assistant.

It provides:
- A **Gradio Web UI** to upload and index `.md`, `.docx`, and `.doc` files
- **Persistent ChromaDB** vector storage (no re-embedding every run)
- **Semantic retrieval + grounded answering** using a Groq-hosted LLM
- **Session-based chat persistence** (SQLite) with rolling summarization for long chats

## Project scope

KnowBridge is designed to work with user-supplied document collections across domains rather than being tied to a single business function.

Supported usage:
- Upload `.md`, `.docx`, or `.doc` files
- Ask factual, procedural, or follow-up questions answerable from those documents
- Receive grounded answers with a `SOURCES:` line for traceability

Out of scope:
- Questions requiring external knowledge not present in the uploaded documents
- Live data lookups from external systems
- Formal benchmark-grade retrieval evaluation in the current version

## How it works (brief)
1. Upload `.md`, `.docx`, or `.doc` files in the **Knowledge Base** tab.
2. The app **hashes** each file and only re-indexes files that changed.
3. Documents are **chunked**, embedded with a sentence-transformer model, and stored in **ChromaDB**.
4. In the **Chat** tab, your question is embedded, relevant chunks are retrieved, and the LLM answers using only retrieved context.
5. The assistant appends a `SOURCES:` line for traceability and stores chat history in SQLite.

## Requirements

### Software
- Python 3.11 recommended

Note: Some dependencies (ChromaDB / PyTorch / sentence-transformers) may not have wheels for very new Python versions yet (e.g. 3.14). If your `python3` is too new, use `python3.11`.

### Python libraries
Install all dependencies from this folder’s requirements file:

```bash
pip install -r requirements.txt
```

Key packages include: `langchain`, `langchain_groq`, `chromadb`, `sentence-transformers`, `gradio`, `python-dotenv`, `pyyaml`, `python-docx`.

### API keys
This app requires a Groq API key.

Create a `.env` file in the **project root** (this folder) with:

```env
GROQ_API_KEY=your-groq-api-key
```

Get a key from: https://console.groq.com/

## How to run

From the `rag-knowledge-assistant/` directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python code/app.py
```

Then open the local Gradio URL printed in the terminal (typically `http://127.0.0.1:7860`).

## Configuration

- App settings (LLM model, retrieval params): `code/config/config.yaml`
- Prompt rules (grounding, `SOURCES:` formatting): `code/config/prompt_config.yaml`

Common knobs:
- `vectordb.n_results` (top-k retrieval)
- `vectordb.threshold` (distance threshold)
- `llm` (Groq model name)

## Data and persistence

- Sample knowledge docs: `data/` (sample `.md` files included)
- Vector DB persisted to: `outputs/vector_db/`
- Chat history persisted to: `outputs/chat_history.db`

## Notes
- The UI supports indexing `.md`, `.docx`, and `.doc`.
- For `.doc`, the app will try to extract text from Confluence/Atlassian export formats (MIME-wrapped HTML) first, then fall back to macOS `textutil` or LibreOffice (`soffice`) conversion.
- If retrieval returns no chunks under the threshold, the pipeline falls back to returning the top-k results to avoid empty context.

## License & attribution
This project is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

- See `LICENSE` for the full text.
- **Attribution:** If you share, reuse, or adapt this project, please provide attribution.
	- Credit: **Raja** (this KnowBridge implementation) and retain the original license notice.
