import os
import logging
from dotenv import load_dotenv
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from document_indexer import get_db_collection, embed_documents, initialize_db

logger = logging.getLogger()


def setup_logging(log_level: str = "INFO"):
    # Accepts a log level string: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # Console output is gated at this level; the log file always captures DEBUG+.
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Clear any handlers pre-attached by third-party libraries (HuggingFace, ChromaDB etc.)
    # to prevent duplicate or uncontrolled console output.
    logger.handlers.clear()

    # Ensure outputs directory exists before attaching a file handler.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Keep root logger at DEBUG so the file handler can always capture full detail.
    # Console verbosity is controlled by the console handler level.
    logger.setLevel(logging.DEBUG)

    # File handler — always write full detail to the log file
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.DEBUG)

    # Console handler — controlled by log_level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


load_dotenv()

# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use initialize_db instead of get_db_collection so the collection is created
# if it doesn't exist yet (e.g. on first run before any documents are uploaded).
collection = initialize_db(collection_name="knowledge_base", delete_existing=False)


def retrieve_relevant_documents(
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> tuple[list[str], list[str]]:
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        tuple: (documents, sources) where documents is a list of text chunks and
               sources is the deduplicated list of source filenames referenced
    """
    logging.debug(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
        "sources": [],
    }
    # Embed the query using the same model used for documents
    logging.debug("Embedding query...")
    query_embedding = embed_documents([query])[0]  # Get the first (and only) embedding

    logging.debug("Querying collection...")
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    # Log all distances before filtering so we can diagnose threshold issues
    logging.debug("All retrieved results (before threshold filter):")
    for i, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
        source = (results["metadatas"][0][i] or {}).get("source", "unknown")
        status = "KEEP" if distance < threshold else "FILTER"
        logging.debug(f"  [{i}] {doc_id} | source={source} | distance={distance:.4f} | {status}")

    logging.debug("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            source = (results["metadatas"][0][i] or {}).get("source", "")
            doc_with_source = f"[Source: {source}]\n{results['documents'][0][i]}" if source else results["documents"][0][i]
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(doc_with_source)
            relevant_results["distances"].append(results["distances"][0][i])
            relevant_results["sources"].append(source)

    # Fallback: if nothing passed the threshold, return all top-N results so the LLM
    # still has context rather than receiving an empty list and hallucinating
    if not relevant_results["documents"] and results["ids"][0]:
        best_distance = min(results["distances"][0]) if results.get("distances") else None
        logging.warning(
            f"No documents passed threshold {threshold}. "
            "Returning top results as fallback so the LLM has context. "
            + (f"(best_distance={best_distance:.4f}; consider raising threshold)" if best_distance is not None else "")
        )
        for i in range(len(results["ids"][0])):
            source = (results["metadatas"][0][i] or {}).get("source", "")
            doc_with_source = f"[Source: {source}]\n{results['documents'][0][i]}" if source else results["documents"][0][i]
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(doc_with_source)
            relevant_results["distances"].append(results["distances"][0][i])
            relevant_results["sources"].append(source)

    # Deduplicate sources while preserving order
    seen = set()
    unique_sources = [s for s in relevant_results["sources"] if s and not (s in seen or seen.add(s))]

    return relevant_results["documents"], unique_sources


def respond_to_query(
    prompt_config: dict,
    query: str,
    llm: str,
    n_results: int = 5,
    threshold: float = 0.3,
    chat_history: list[dict] | None = None,
    history_summary: str | None = None,
    assistant_type: str = "PMO",
) -> tuple[str, list[str]]:
    """
    Respond to a query using the ChromaDB database.

    Args:
        chat_history: Recent turns as dicts with 'role' and 'content' (trimmed window).
        history_summary: Optional rolling summary of older messages beyond the trim
                         window. Injected into the system prompt so the LLM has
                         context of the full conversation even when history is trimmed.
        assistant_type: The domain/type of assistant (e.g. "PMO", "HR"). Used to
                        substitute {assistant_type} placeholders in the prompt config.

    Returns:
        tuple: (response_text, source_files)
    """

    relevant_documents, source_files = retrieve_relevant_documents(
        query, n_results=n_results, threshold=threshold
    )

    logging.info(
        "RAG retrieval: %s chunks from %s sources (top_k=%s, threshold=%s)",
        len(relevant_documents),
        len(source_files),
        n_results,
        threshold,
    )

    logging.debug("-" * 100)
    logging.debug("Relevant documents: \n")
    for doc in relevant_documents:
        logging.debug(doc)
        logging.debug("-" * 100)
    logging.debug("")

    logging.debug("User's question:")
    logging.debug(query)
    logging.debug("")
    logging.debug("-" * 100)
    logging.debug("")
    input_data = (
        f"Relevant documents:\n\n{relevant_documents}\n\nUser's question:\n\n{query}"
    )

    # Substitute {assistant_type} placeholder in the prompt config before building
    resolved_config = {
        k: v.replace("{assistant_type}", assistant_type) if isinstance(v, str)
        else [item.replace("{assistant_type}", assistant_type) if isinstance(item, str) else item for item in v]
        if isinstance(v, list) else v
        for k, v in prompt_config.items()
    }

    rag_assistant_prompt = build_prompt_from_config(
        resolved_config, input_data=input_data
    )

    logging.debug(f"RAG assistant prompt: {rag_assistant_prompt}")
    logging.debug("")

    llm = ChatGroq(model=llm)

    # Build a messages list: system prompt → prior history → current user query.
    # If a rolling summary exists, append it to the system prompt so the LLM
    # understands the full arc of the conversation even with a trimmed window.
    system_content = rag_assistant_prompt
    if history_summary:
        system_content += f"\n\nSummary of earlier conversation:\n{history_summary}"

    messages = [SystemMessage(content=system_content)]
    for turn in (chat_history or []):
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))

    raw_response = llm.invoke(messages).content

    # Parse the SOURCES line the LLM was instructed to append, then strip it
    # from the visible response so the answer stays clean.
    actual_sources = []
    response_lines = raw_response.strip().splitlines()
    clean_lines = []
    for line in response_lines:
        stripped = line.strip()
        if stripped.upper().startswith("SOURCES:"):
            sources_part = stripped[len("SOURCES:"):].strip()
            if sources_part.lower() != "none":
                seen = set()
                actual_sources = [
                    s.strip() for s in sources_part.split(",")
                    if s.strip() and not (s.strip() in seen or seen.add(s.strip()))
                ]
        else:
            clean_lines.append(line)

    clean_response = "\n".join(clean_lines).strip()
    return clean_response, actual_sources


def summarize_history(
    messages: list[dict],
    llm_model: str,
    existing_summary: str | None = None,
) -> str:
    """
    Summarize a list of chat messages into a concise rolling context string.
    If an existing_summary is provided, the new messages are incorporated into it.

    Args:
        messages: List of {"role": ..., "content": ...} dicts to summarize.
        llm_model: The LLM model name to use for summarization.
        existing_summary: Previous rolling summary to build upon.

    Returns:
        Updated summary string.
    """
    history_text = ""
    if existing_summary:
        history_text += f"Previous summary:\n{existing_summary}\n\n"
    history_text += "New messages to incorporate:\n"
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    summary_messages = [
        SystemMessage(
            content=(
                "You are a conversation summarizer. Create a concise summary of the "
                "conversation history provided. Capture key topics discussed, questions "
                "asked, and answers given. The summary will be used as context for future "
                "conversation turns. Be concise but preserve important details. "
                "Output only the summary text, no preamble."
            )
        ),
        HumanMessage(content=history_text),
    ]
    llm = ChatGroq(model=llm_model)
    return llm.invoke(summary_messages).content


if __name__ == "__main__":
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    setup_logging(log_level=app_config.get("log_level", "INFO"))
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

    rag_assistant_prompt = prompt_config["ai_assistant_system_prompt_advanced"]

    vectordb_params = app_config["vectordb"]
    llm = app_config["llm"]

    exit_app = False
    while not exit_app:
        query = input(
            "Enter a question, 'config' to change the parameters, or 'exit' to quit: "
        )
        if query == "exit":
            exit_app = True
            exit()

        elif query == "config":
            threshold = float(input("Enter the retrieval threshold: "))
            n_results = int(input("Enter the Top K value: "))
            vectordb_params = {
                "threshold": threshold,
                "n_results": n_results,
            }
            continue

        response, source_files = respond_to_query(
            prompt_config=rag_assistant_prompt,
            query=query,
            llm=llm,
            **vectordb_params,
        )
        logging.debug("-" * 100)
        print("LLM response:")
        print(response + "\n")
        if source_files:
            # Back-compat: older indexed markdown sources were stored without extension.
            def _display_filename(source: str) -> str:
                _, ext = os.path.splitext(source)
                return source if ext else f"{source}.md"

            sources_display = ", ".join(_display_filename(s) for s in source_files)
            print(f"Referenced documents: {sources_display}\n")
