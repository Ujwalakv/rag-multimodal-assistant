"""
rag_pipeline.py  -  Enhanced RAG Pipeline (hardened)

Key changes vs. the previous version
  * Groq model IDs are configurable (env / Streamlit secrets) and have fallbacks,
    so a decommissioned model no longer crashes the app.
  * One central `llm_invoke()` used everywhere.
  * query_rag() never raises - errors are returned as a readable answer.
  * Secrets are read lazily; missing keys give a clear message, not a KeyError.
  * Dedupe bug fixed (chunks sharing the same contextual prefix were dropped).
  * Reranking uses the ORIGINAL question; retrieval uses the expanded one.
  * Image extraction converts every image to PNG and is size-capped.
"""

from __future__ import annotations

import os
import time
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # older environments
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import CrossEncoder

load_dotenv()


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
def _get_secret(name: str, default: str = "") -> str:
    """Read from env vars first, then Streamlit secrets (if available)."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


PINECONE_API_KEY = _get_secret("PINECONE_API_KEY")
GROQ_API_KEY     = _get_secret("GROQ_API_KEY")
INDEX_NAME       = _get_secret("PINECONE_INDEX_NAME", "rag-index-v2")

EMBEDDING_MODEL  = _get_secret("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIM    = int(_get_secret("EMBEDDING_DIM", "1024"))   # bge-small => 384

# Groq models - override with env vars / secrets if Groq retires one again.
LLM_MODEL       = _get_secret("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
VISION_MODEL    = _get_secret("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
WHISPER_MODEL   = _get_secret("GROQ_WHISPER_MODEL", "whisper-large-v3")

CHUNK_SIZE       = 1500
CHUNK_OVERLAP    = 200
TOP_K_RETRIEVE   = 20
TOP_K_BM25       = 10
TOP_K_RERANK     = 5
CONTEXT_CHARS    = 1500          # per chunk sent to the LLM
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_IMAGES_PER_PDF = 10
MAX_AUDIO_BYTES    = 25 * 1024 * 1024   # Groq Whisper upload limit

ENABLE_CONTEXTUAL_CHUNKING = True
ENABLE_IMAGE_EXTRACTION    = True
ENABLE_GRAPH_BUILDING      = True

PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the context below.
The context may come from PDFs, Word documents, audio transcriptions, or image descriptions.

Rules:
1. If the question asks for a list, return EVERY item - do not stop early.
2. Read the ENTIRE context carefully before answering.
3. Do NOT make up information not in the context.
4. If the answer is not in the context, say:
   "I could not find this information in the uploaded documents."

Context:
{context}

Question: {question}

Thorough answer:"""

QA_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# ─────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────
_embeddings   = None
_pc_index     = None
_vector_store = None
_reranker     = None
_graph        = nx.Graph()
_all_chunks: List[Document] = []
_bm25_cache: Dict[str, Any] = {"n": -1, "retriever": None}


def missing_config() -> List[str]:
    """Names of required secrets that are not set."""
    return [n for n, v in (("PINECONE_API_KEY", PINECONE_API_KEY),
                           ("GROQ_API_KEY", GROQ_API_KEY)) if not v]


# ─────────────────────────────────────────────
#  LLM HELPER (single place that talks to Groq chat)
# ─────────────────────────────────────────────
def llm_invoke(prompt: str, max_tokens: int = 1024) -> str:
    """Call Groq; if a model is rejected/decommissioned/rate-limited, try the next one."""
    last_err: Optional[Exception] = None
    for model in dict.fromkeys([LLM_MODEL] + FALLBACK_MODELS):
        try:
            llm = ChatGroq(model=model, api_key=GROQ_API_KEY,
                           temperature=0, max_tokens=max_tokens)
            return str(llm.invoke(prompt).content)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"All Groq models failed. Last error: {last_err}")


# ─────────────────────────────────────────────
#  LAZY RESOURCES
# ─────────────────────────────────────────────
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_pinecone_index():
    global _pc_index
    if _pc_index is not None:
        return _pc_index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        dim = pc.describe_index(INDEX_NAME).dimension
        if dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Pinecone index '{INDEX_NAME}' has dimension {dim} but the embedding "
                f"model needs {EMBEDDING_DIM}. Set PINECONE_INDEX_NAME to a new name."
            )
    _pc_index = pc.Index(INDEX_NAME)
    return _pc_index


def get_total_vectors() -> int:
    stats = get_pinecone_index().describe_index_stats()
    try:
        return int(stats["total_vector_count"])
    except Exception:
        return int(getattr(stats, "total_vector_count", 0) or 0)


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = PineconeVectorStore(
            index=get_pinecone_index(),
            embedding=get_embeddings(),
            text_key="text",
        )
    return _vector_store


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def reset_all() -> None:
    """Delete the Pinecone index and clear all in-memory state."""
    global _pc_index, _vector_store
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME in [i.name for i in pc.list_indexes()]:
        pc.delete_index(INDEX_NAME)
    _pc_index = None
    _vector_store = None
    _all_chunks.clear()
    _graph.clear()
    _bm25_cache.update({"n": -1, "retriever": None})


# ─────────────────────────────────────────────
#  INGESTION HELPERS
# ─────────────────────────────────────────────
def add_context_to_chunks(chunks: List[Document], doc_summary: str) -> List[Document]:
    for chunk in chunks:
        chunk.page_content = f"[Document context: {doc_summary}]\n\n{chunk.page_content}"
    return chunks


def summarize_document(text: str) -> str:
    try:
        return llm_invoke(
            f"In ONE sentence, describe what this document is about:\n\n{text[:1500]}",
            max_tokens=100,
        ).strip()
    except Exception:
        return "Unknown document"


def extract_images_from_pdf(pdf_path: str, filename: str) -> List[Document]:
    import fitz
    import base64
    from groq import Groq as GroqClient

    client = GroqClient(api_key=GROQ_API_KEY)
    image_docs: List[Document] = []

    pdf = fitz.open(pdf_path)
    try:
        for page_num, page in enumerate(pdf, start=1):
            for img_index, img_info in enumerate(page.get_images(full=True)):
                if len(image_docs) >= MAX_IMAGES_PER_PDF:
                    return image_docs
                try:
                    xref = img_info[0]
                    pix = fitz.Pixmap(pdf, xref)
                    if pix.n - pix.alpha >= 4:           # CMYK etc. -> RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_bytes = pix.tobytes("png")

                    # skip tiny icons and images larger than Groq's ~4MB limit
                    if len(image_bytes) < 5000 or len(image_bytes) > 3_000_000:
                        continue

                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    response = client.chat.completions.create(
                        model=VISION_MODEL,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                {"type": "text",
                                 "text": "Describe this image in detail. If it contains a chart, "
                                         "table, or diagram, extract and describe all "
                                         "data/labels/values you see."},
                            ],
                        }],
                        max_tokens=512,
                    )
                    description = response.choices[0].message.content
                except Exception as e:  # noqa: BLE001
                    description = f"[Image on page {page_num} - description failed: {e}]"

                image_docs.append(Document(
                    page_content=f"[Image from page {page_num}]: {description}",
                    metadata={"source": filename, "type": "image",
                              "page": page_num, "img_index": img_index},
                ))
    finally:
        pdf.close()
    return image_docs


def extract_entities_and_relations(text: str, source: str) -> List[Dict]:
    prompt = f"""Extract up to 15 important relationships from this text.
Return ONLY valid JSON - a list of objects with keys: entity1, relation, entity2.
No explanation, no markdown, just raw JSON array.

Text:
{text[:3000]}

JSON:"""
    try:
        raw = llm_invoke(prompt, max_tokens=1024).strip()
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        triples = json.loads(raw)
        return [t for t in triples if isinstance(t, dict)] if isinstance(triples, list) else []
    except Exception:
        return []


def build_graph(chunks: List[Document]):
    if not chunks:
        return
    full_text = " ".join(c.page_content for c in chunks[:5])[:2000]
    source = chunks[0].metadata.get("source", "unknown")
    for t in extract_entities_and_relations(full_text, source):
        e1 = str(t.get("entity1", "")).strip()
        rel = str(t.get("relation", "")).strip()
        e2 = str(t.get("entity2", "")).strip()
        if e1 and e2:
            _graph.add_edge(e1, e2, relation=rel, source=source)


def graph_query(question: str) -> str:
    if _graph.number_of_nodes() == 0:
        return ""
    q = question.lower()
    parts: List[str] = []
    for node in list(_graph.nodes()):
        if len(node) < 3 or node.lower() not in q:
            continue
        for nb in list(_graph.neighbors(node))[:5]:
            rel = _graph[node][nb].get("relation", "related to") or "related to"
            parts.append(f"{node} {rel} {nb}")
        if len(parts) >= 20:
            break
    return "\n".join(parts)


# ─────────────────────────────────────────────
#  RETRIEVAL
# ─────────────────────────────────────────────
def _doc_key(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode("utf-8", "ignore")).hexdigest()


def _get_bm25() -> Optional[BM25Retriever]:
    if not _all_chunks:
        return None
    if _bm25_cache["n"] != len(_all_chunks):
        _bm25_cache["retriever"] = BM25Retriever.from_documents(_all_chunks, k=TOP_K_BM25)
        _bm25_cache["n"] = len(_all_chunks)
    return _bm25_cache["retriever"]


def hybrid_retrieve(question: str, expanded: Optional[str] = None,
                    k_dense: int = TOP_K_RETRIEVE) -> List[Document]:
    expanded = expanded or question
    dense_docs = get_vector_store().similarity_search(expanded, k=k_dense)

    bm25_docs: List[Document] = []
    try:
        bm25 = _get_bm25()
        if bm25 is not None:
            bm25_docs = bm25.invoke(question)
    except Exception:
        bm25_docs = []

    seen, merged = set(), []
    for doc in dense_docs + bm25_docs:
        key = _doc_key(doc)
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    if not merged:
        return []

    try:
        scores = get_reranker().predict([(question, d.page_content) for d in merged])
        ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:TOP_K_RERANK]]
    except Exception:
        return merged[:TOP_K_RERANK]   # reranker failed - still return something


def expand_query(question: str) -> str:
    try:
        extra = llm_invoke(
            "Rewrite this question in 2 different phrasings for search. "
            f"Return ONLY the rephrased versions, one per line, nothing else.\n\nQuestion: {question}",
            max_tokens=200,
        )
        return question + "\n" + extra
    except Exception:
        return question


def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def push_to_pinecone(chunks: List[Document]):
    get_vector_store().add_documents(chunks)
    _all_chunks.extend(chunks)          # only after a successful upsert


# ─────────────────────────────────────────────
#  FILE PROCESSING
# ─────────────────────────────────────────────
def process_file(file_bytes: bytes, filename: str) -> Tuple[int, int]:
    import tempfile
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyMuPDFLoader(tmp_path) if suffix == ".pdf" else Docx2txtLoader(tmp_path)
        documents = loader.load()
        full_text = " ".join(d.page_content for d in documents)
        chunks = make_splitter().split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = filename

        if ENABLE_CONTEXTUAL_CHUNKING:
            chunks = add_context_to_chunks(chunks, summarize_document(full_text))

        image_docs: List[Document] = []
        if ENABLE_IMAGE_EXTRACTION and suffix == ".pdf":
            try:
                image_docs = extract_images_from_pdf(tmp_path, filename)
            except Exception:
                image_docs = []

        all_chunks = chunks + image_docs

        if ENABLE_GRAPH_BUILDING:
            try:
                build_graph(chunks)
            except Exception:
                pass

        push_to_pinecone(all_chunks)
        return len(documents), len(all_chunks)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def process_audio(file_bytes: bytes, filename: str, progress_cb=None) -> str:
    from groq import Groq as GroqClient

    if len(file_bytes) > MAX_AUDIO_BYTES:
        raise ValueError("Audio file is larger than 25 MB (Groq Whisper limit).")

    if progress_cb:
        progress_cb("Transcribing audio with Groq Whisper...")

    client = GroqClient(api_key=GROQ_API_KEY)
    result = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=(filename, file_bytes),
    )
    text = result.text

    doc_summary = summarize_document(text)
    full_doc = Document(
        page_content=f"[Document context: {doc_summary}]\n\n{text}",
        metadata={"source": filename, "type": "audio_full"},
    )
    chunks = make_splitter().split_documents([full_doc])
    for chunk in chunks:
        chunk.metadata["source"] = filename

    try:
        build_graph(chunks)
    except Exception:
        pass
    push_to_pinecone(chunks)
    return text


# ─────────────────────────────────────────────
#  QUERY  (never raises)
# ─────────────────────────────────────────────
def query_rag(question: str) -> Dict[str, Any]:
    base = {"source_documents": [], "graph_context": "", "expanded_query": question}

    missing = missing_config()
    if missing:
        return {**base, "result": f"Missing configuration: {', '.join(missing)}. "
                                  "Add them in the app's Secrets."}

    try:
        total = get_total_vectors()
    except Exception as e:  # noqa: BLE001
        return {**base, "result": f"Could not reach Pinecone: {e}"}
    if total == 0:
        return {**base, "result": "No documents found. Please upload files first."}

    expanded = expand_query(question)

    try:
        top_docs = hybrid_retrieve(question, expanded)
    except Exception as e:  # noqa: BLE001
        return {**base, "expanded_query": expanded, "result": f"Retrieval failed: {e}"}

    graph_ctx = graph_query(question)
    doc_context = "\n\n---\n\n".join(d.page_content[:CONTEXT_CHARS] for d in top_docs)
    if graph_ctx:
        doc_context = f"[Knowledge Graph Context]\n{graph_ctx}\n\n---\n\n{doc_context}"

    try:
        answer = llm_invoke(QA_PROMPT.format(context=doc_context, question=question),
                            max_tokens=1024)
    except Exception as e:  # noqa: BLE001
        answer = f"The language model request failed: {e}"

    return {
        "result": answer,
        "source_documents": top_docs,
        "graph_context": graph_ctx,
        "expanded_query": expanded,
    }


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
def _parse_score(raw: str, default: float = 5.0) -> float:
    m = re.search(r"\d+(?:\.\d+)?", raw or "")
    if not m:
        return default
    return min(10.0, max(0.0, float(m.group(0))))


def _score(prompt: str) -> float:
    try:
        return _parse_score(llm_invoke(prompt, max_tokens=10))
    except Exception:
        return 5.0


def evaluate_rag(test_cases: List[Dict]) -> Dict[str, Any]:
    """
    test_cases: list of {"question": ..., "ground_truth": ...}
    Returns faithfulness, answer_relevancy, (correctness), overall, details.
    """
    faithfulness_scores, relevancy_scores, correctness_scores = [], [], []
    details: List[Dict[str, str]] = []

    for tc in test_cases:
        q = tc["question"]
        gt = (tc.get("ground_truth") or "").strip()
        result = query_rag(q)
        answer = result["result"]
        context = "\n".join(d.page_content for d in result["source_documents"])
        details.append({"question": q, "answer": answer, "ground_truth": gt})

        faithfulness_scores.append(_score(
            f"""Rate from 0-10 how well this answer is supported by the context.
Return ONLY a single integer number, nothing else.
Context: {context[:1500]}
Answer: {answer}
Score:""") / 10)

        relevancy_scores.append(_score(
            f"""Rate from 0-10 how relevant this answer is to the question.
Return ONLY a single integer number, nothing else.
Question: {q}
Answer: {answer}
Score:""") / 10)

        if gt:
            correctness_scores.append(_score(
                f"""You are a strict factual evaluator.
Compare the predicted answer to the ground truth and rate correctness from 0-10.

Scoring rules:
- Same key names / facts / values as the ground truth -> 8-10.
- Different names, wrong facts, or contradiction -> 0-3.
- Partial match -> 4-7.
- Do NOT reward an answer just because it sounds confident or well-written.
- Spelling variations of the SAME name (e.g. "Dr. Pushpalatha" vs "Dr Pushpalatha") are correct.
- Clearly different names or values (e.g. "Pushpalatha" vs "Puppalatha") are WRONG -> 0-2.
- Return ONLY a single integer from 0 to 10, nothing else.

Question        : {q}
Ground Truth    : {gt}
Predicted Answer: {answer}
Score:""") / 10)

    n = max(len(test_cases), 1)
    avg_faith = round(sum(faithfulness_scores) / n, 3)
    avg_rel = round(sum(relevancy_scores) / n, 3)

    out: Dict[str, Any] = {
        "faithfulness": avg_faith,
        "answer_relevancy": avg_rel,
        "num_cases": len(test_cases),
        "has_ground_truth": len(correctness_scores) > 0,
        "details": details,
    }
    if correctness_scores:
        avg_correct = round(sum(correctness_scores) / len(correctness_scores), 3)
        out["correctness"] = avg_correct
        out["overall"] = round(avg_faith * 0.25 + avg_rel * 0.25 + avg_correct * 0.50, 3)
    else:
        out["overall"] = round((avg_faith + avg_rel) / 2, 3)
    return out


if __name__ == "__main__":
    docs = PyMuPDFLoader("test.pdf").load()
    full = " ".join(d.page_content for d in docs)
    summary = summarize_document(full)
    chunks = make_splitter().split_documents(docs)
    for c in chunks:
        c.metadata["source"] = "test.pdf"
    chunks = add_context_to_chunks(chunks, summary)
    build_graph(chunks)
    push_to_pinecone(chunks)

    result = query_rag("What is this document about?")
    print("\nAnswer:", result["result"])
    print("\nGraph context:", result["graph_context"])