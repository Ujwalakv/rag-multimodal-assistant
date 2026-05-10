"""
rag_pipeline.py  —  Enhanced RAG Pipeline
Features:
  - Hybrid search (BM25 sparse + dense embeddings)
  - Cross-encoder reranking
  - BGE-large embeddings
  - Contextual chunking
  - Image extraction + vision description
  - Graph RAG (entity/relationship extraction)
  - RAGAS evaluation metrics
"""

from __future__ import annotations

import os
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
INDEX_NAME       = os.environ.get("PINECONE_INDEX_NAME", "rag-index-v2")
EMBEDDING_DIM    = 1024                          # BGE-large dimension
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"     # upgraded from MiniLM
LLM_MODEL        = "llama-3.3-70b-versatile"
WHISPER_MODEL    = "whisper-large-v3"
CHUNK_SIZE       = 1500
CHUNK_OVERLAP    = 200
TOP_K_RETRIEVE   = 20   # retrieve many, rerank to fewer
TOP_K_RERANK     = 5    # keep only top 5 after reranking
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Speed toggles — set to False to skip slow LLM calls during upload ──
ENABLE_CONTEXTUAL_CHUNKING = True   # adds ~5s per doc (1 LLM call)
ENABLE_IMAGE_EXTRACTION    = True   # adds ~10s per image in PDF
ENABLE_GRAPH_BUILDING      = True   # adds ~5s per doc (1 LLM call)

PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the context below.
The context may come from PDFs, Word documents, audio transcriptions, or image descriptions.

Rules:
1. If the question asks for a list, return EVERY item — do not stop early.
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
#  SINGLETONS
# ─────────────────────────────────────────────
_embeddings   = None
_pc_index     = None
_vector_store = None
_reranker     = None
_graph        = nx.Graph()          # in-memory knowledge graph
_all_chunks: List[Document] = []    # kept in RAM for BM25


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
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",          # required for sparse-dense hybrid
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index ready!")
    _pc_index = pc.Index(INDEX_NAME)
    return _pc_index


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


# ─────────────────────────────────────────────
#  CONTEXTUAL CHUNKING
#  Prepends a short AI-generated summary to each
#  chunk so retrieval survives "lost in the middle"
# ─────────────────────────────────────────────

def add_context_to_chunks(chunks: List[Document], doc_summary: str) -> List[Document]:
    """Prepend document-level context to each chunk."""
    enriched = []
    for chunk in chunks:
        chunk.page_content = (
            f"[Document context: {doc_summary}]\n\n{chunk.page_content}"
        )
        enriched.append(chunk)
    return enriched


def summarize_document(text: str) -> str:
    """Ask the LLM for a one-line summary of the full document text."""
    try:
        llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
        sample = text[:3000]   # first 3k chars is enough for a summary
        response = llm.invoke(
            f"In ONE sentence, describe what this document is about:\n\n{sample}"
        )
        return response.content.strip()
    except Exception:
        return "Unknown document"


# ─────────────────────────────────────────────
#  IMAGE EXTRACTION + VISION DESCRIPTION
# ─────────────────────────────────────────────

def extract_images_from_pdf(pdf_path: str, filename: str) -> List[Document]:
    """
    Extract embedded images from a PDF using PyMuPDF,
    describe each with Groq vision, return as Documents.
    """
    import fitz  # PyMuPDF
    import base64
    from groq import Groq as GroqClient

    client = GroqClient(api_key=GROQ_API_KEY)
    image_docs = []

    pdf = fitz.open(pdf_path)
    for page_num, page in enumerate(pdf, start=1):
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext   = base_image["ext"]

            # Skip tiny images (likely icons/decorations)
            if len(image_bytes) < 5000:
                continue

            b64 = base64.b64encode(image_bytes).decode("utf-8")
            media_type = f"image/{image_ext}" if image_ext != "jpg" else "image/jpeg"

            try:
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{b64}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "Describe this image in detail. "
                                        "If it contains a chart, table, or diagram, "
                                        "extract and describe all data/labels/values you see."
                                    ),
                                },
                            ],
                        }
                    ],
                    max_tokens=512,
                )
                description = response.choices[0].message.content
            except Exception as e:
                description = f"[Image on page {page_num} — description failed: {e}]"

            image_docs.append(
                Document(
                    page_content=f"[Image from page {page_num}]: {description}",
                    metadata={
                        "source": filename,
                        "type": "image",
                        "page": page_num,
                        "img_index": img_index,
                    },
                )
            )

    pdf.close()
    print(f"  → Extracted and described {len(image_docs)} images from '{filename}'")
    return image_docs


# ─────────────────────────────────────────────
#  GRAPH RAG — Entity/Relationship Extraction
# ─────────────────────────────────────────────

def extract_entities_and_relations(text: str, source: str) -> List[Dict]:
    """
    Ask the LLM to extract (entity1, relation, entity2) triples.
    Returns a list of triple dicts.
    """
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
    prompt = f"""Extract up to 15 important relationships from this text.
Return ONLY valid JSON — a list of objects with keys: entity1, relation, entity2.
No explanation, no markdown, just raw JSON array.

Text:
{text[:3000]}

JSON:"""
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        triples = json.loads(raw)
        return triples if isinstance(triples, list) else []
    except Exception:
        return []


def build_graph(chunks: List[Document]):
    """Add entity nodes and relation edges to the in-memory graph."""
    global _graph
    full_text = " ".join(c.page_content for c in chunks[:10])   # sample
    source = chunks[0].metadata.get("source", "unknown") if chunks else "unknown"
    triples = extract_entities_and_relations(full_text, source)
    for t in triples:
        e1 = str(t.get("entity1", "")).strip()
        rel = str(t.get("relation", "")).strip()
        e2 = str(t.get("entity2", "")).strip()
        if e1 and e2:
            _graph.add_edge(e1, e2, relation=rel, source=source)
    print(f"  → Graph now has {_graph.number_of_nodes()} nodes, {_graph.number_of_edges()} edges")


def graph_query(question: str) -> str:
    """
    Simple graph traversal: find entities in question that exist in graph,
    return their neighbours and relations as context string.
    """
    if _graph.number_of_nodes() == 0:
        return ""
    words = set(re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", question))
    graph_context_parts = []
    for word in words:
        if _graph.has_node(word):
            neighbors = list(_graph.neighbors(word))[:5]
            for nb in neighbors:
                rel = _graph[word][nb].get("relation", "related to")
                graph_context_parts.append(f"{word} {rel} {nb}")
    return "\n".join(graph_context_parts)


# ─────────────────────────────────────────────
#  HYBRID RETRIEVAL — BM25 + Dense + Reranking
# ─────────────────────────────────────────────

def hybrid_retrieve(question: str, k_dense: int = TOP_K_RETRIEVE) -> List[Document]:
    """
    1. Dense retrieval from Pinecone (semantic)
    2. BM25 retrieval from in-memory chunks (keyword)
    3. Merge, deduplicate
    4. Cross-encoder rerank → top TOP_K_RERANK
    """
    # Dense retrieval
    dense_docs = get_vector_store().similarity_search(question, k=k_dense)

    # BM25 retrieval (keyword)
    bm25_docs = []
    if _all_chunks:
        bm25 = BM25Retriever.from_documents(_all_chunks, k=10)
        bm25_docs = bm25.invoke(question)

    # Merge + deduplicate by content
    seen = set()
    merged = []
    for doc in dense_docs + bm25_docs:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    if not merged:
        return []

    # Cross-encoder rerank
    reranker = get_reranker()
    pairs = [(question, doc.page_content) for doc in merged]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:TOP_K_RERANK]]

    return top_docs


# ─────────────────────────────────────────────
#  QUERY EXPANSION (lightweight — no extra LLM call)
# ─────────────────────────────────────────────

def expand_query(question: str) -> str:
    try:
        llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
        response = llm.invoke(
            f"Rewrite this question in 2 different phrasings for search. "
            f"Return ONLY the rephrased versions, one per line, nothing else.\n\nQuestion: {question}"
        )
        return question + "\n" + response.content
    except Exception:
        return question


# ─────────────────────────────────────────────
#  INGESTION
# ─────────────────────────────────────────────

def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def push_to_pinecone(chunks: List[Document]):
    global _all_chunks
    _all_chunks.extend(chunks)
    get_vector_store().add_documents(chunks)


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

        # Contextual chunking (optional — 1 LLM call)
        if ENABLE_CONTEXTUAL_CHUNKING:
            print(f"  → Summarizing '{filename}' for contextual chunking...")
            doc_summary = summarize_document(full_text)
            print(f"  → Summary: {doc_summary}")
            chunks = add_context_to_chunks(chunks, doc_summary)

        # Image extraction (optional — 1 vision API call per image)
        image_docs = []
        if ENABLE_IMAGE_EXTRACTION and suffix == ".pdf":
            print(f"  → Extracting images from '{filename}'...")
            image_docs = extract_images_from_pdf(tmp_path, filename)

        all_chunks = chunks + image_docs

        # Knowledge graph (optional — 1 LLM call)
        if ENABLE_GRAPH_BUILDING:
            print(f"  → Building knowledge graph from '{filename}'...")
            build_graph(chunks)

        push_to_pinecone(all_chunks)
        return len(documents), len(all_chunks)
    finally:
        os.unlink(tmp_path)


def process_audio(file_bytes: bytes, filename: str, progress_cb=None) -> str:
    import tempfile
    from groq import Groq as GroqClient

    suffix = os.path.splitext(filename)[1].lower() or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if progress_cb:
            progress_cb("Transcribing audio with Groq Whisper...")
        client = GroqClient(api_key=GROQ_API_KEY)
        with open(tmp_path, "rb") as af:
            result = client.audio.transcriptions.create(model=WHISPER_MODEL, file=af)
        text = result.text

        doc_summary = summarize_document(text)
        full_doc = Document(
            page_content=f"[Document context: {doc_summary}]\n\n{text}",
            metadata={"source": filename, "type": "audio_full"},
        )
        chunks = make_splitter().split_documents([full_doc])
        for chunk in chunks:
            chunk.metadata["source"] = filename

        build_graph(chunks)
        push_to_pinecone(chunks)
        return text
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
#  MAIN RAG QUERY
# ─────────────────────────────────────────────

def query_rag(question: str) -> Dict[str, Any]:
    stats = get_pinecone_index().describe_index_stats()
    total = stats.get("total_vector_count", 0)
    if total == 0:
        return {
            "result": "No documents found. Please upload files first.",
            "source_documents": [],
            "graph_context": "",
            "reranked_scores": [],
        }

    expanded = expand_query(question)

    # Hybrid retrieval + rerank
    top_docs = hybrid_retrieve(expanded)

    # Graph context
    graph_ctx = graph_query(question)

    # Build context
    doc_context = "\n\n---\n\n".join(d.page_content for d in top_docs)
    if graph_ctx:
        doc_context = f"[Knowledge Graph Context]\n{graph_ctx}\n\n---\n\n{doc_context}"

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
    final_prompt = QA_PROMPT.format(context=doc_context, question=question)
    response = llm.invoke(final_prompt)

    return {
        "result": response.content,
        "source_documents": top_docs,
        "graph_context": graph_ctx,
        "expanded_query": expanded,
    }


# ─────────────────────────────────────────────
#  RAGAS EVALUATION
# ─────────────────────────────────────────────

def evaluate_rag(test_cases: List[Dict]) -> Dict[str, float]:
    """
    Lightweight RAGAS-style evaluation without the ragas library.
    test_cases: list of {"question": ..., "ground_truth": ...}
    Returns avg faithfulness proxy and answer relevancy proxy.
    """
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)
    faithfulness_scores = []
    relevancy_scores = []

    for tc in test_cases:
        q = tc["question"]
        gt = tc.get("ground_truth", "")
        result = query_rag(q)
        answer = result["result"]
        context = "\n".join(d.page_content for d in result["source_documents"])

        # Faithfulness: is the answer supported by context?
        faith_prompt = f"""Rate from 0-10 how well this answer is supported by the context.
Return ONLY a number.
Context: {context[:1500]}
Answer: {answer}
Score:"""
        try:
            faith_score = float(llm.invoke(faith_prompt).content.strip())
        except Exception:
            faith_score = 5.0
        faithfulness_scores.append(faith_score / 10)

        # Relevancy: does the answer address the question?
        rel_prompt = f"""Rate from 0-10 how relevant this answer is to the question.
Return ONLY a number.
Question: {q}
Answer: {answer}
Score:"""
        try:
            rel_score = float(llm.invoke(rel_prompt).content.strip())
        except Exception:
            rel_score = 5.0
        relevancy_scores.append(rel_score / 10)

    return {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3),
        "answer_relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 3),
        "num_cases": len(test_cases),
    }


# ─────────────────────────────────────────────
#  CLI TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    docs   = PyMuPDFLoader("test.pdf").load()
    full   = " ".join(d.page_content for d in docs)
    summary = summarize_document(full)
    chunks  = make_splitter().split_documents(docs)
    for c in chunks:
        c.metadata["source"] = "test.pdf"
    chunks = add_context_to_chunks(chunks, summary)
    build_graph(chunks)
    push_to_pinecone(chunks)

    result = query_rag("What is this document about?")
    print("\n📌 Answer:", result["result"])
    print("\n🕸️  Graph context:", result["graph_context"])