"""
app.py  —  Enhanced RAG Multimodal Assistant
"""

import os
import json
import time

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from rag_pipeline import (
    get_pinecone_index,
    get_vector_store,
    get_embeddings,
    process_file,
    process_audio,
    query_rag,
    evaluate_rag,
    _graph,
    _all_chunks,
    GROQ_API_KEY,
    LLM_MODEL,
)

UPLOADED_FILES_LOG = "uploaded_files.json"

st.set_page_config(
    page_title="RAG Assistant Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px;
    margin: 6px 0;
    border-left: 4px solid #7c3aed;
}
.source-badge {
    background: #374151;
    color: #d1d5db;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    margin: 2px;
    display: inline-block;
}
.graph-info {
    background: #064e3b;
    border-radius: 8px;
    padding: 10px;
    border-left: 3px solid #10b981;
}
.answer-box {
    background: #1e293b;
    border-radius: 10px;
    padding: 20px;
    border-left: 4px solid #3b82f6;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 RAG Multimodal Assistant Pro")
st.caption("Hybrid Search · Cross-Encoder Reranking · Graph RAG · Vision · RAGAS Evaluation")


def load_log():
    if os.path.exists(UPLOADED_FILES_LOG):
        with open(UPLOADED_FILES_LOG, "r") as f:
            return json.load(f)
    return []


def save_log(files):
    with open(UPLOADED_FILES_LOG, "w") as f:
        json.dump(files, f)


with st.sidebar:
    st.title("📁 File Manager")

    st.markdown("### 🔌 Pinecone Status")
    try:
        stats = get_pinecone_index().describe_index_stats()
        total = stats.get("total_vector_count", 0)
        st.success(f"✅ Connected — {total} vectors stored")
        st.caption(f"Index: `{os.environ.get('PINECONE_INDEX_NAME', 'rag-index-v2')}`")
    except Exception as e:
        st.error(f"❌ Pinecone error: {e}")

    from rag_pipeline import _graph as kg
    if kg.number_of_nodes() > 0:
        st.success(f"🕸️ Graph: {kg.number_of_nodes()} entities, {kg.number_of_edges()} relations")

    st.markdown("---")

    uploaded_log = load_log()
    if uploaded_log:
        st.markdown("**📄 Uploaded files:**")
        for fname in uploaded_log:
            ext = os.path.splitext(fname)[1].lower()
            icon = "🎙️" if ext in (".mp3", ".wav", ".m4a") else "📄"
            st.write(f"{icon} {fname}")
    else:
        st.info("No files uploaded yet.")

    st.markdown("---")

    new_files = st.file_uploader(
        "Upload PDFs, Word docs, or Audio",
        type=["pdf", "docx", "mp3", "wav", "m4a"],
        accept_multiple_files=True,
    )

    if new_files:
        for uf in new_files:
            if uf.name in uploaded_log:
                st.warning(f"⚠️ Already uploaded: {uf.name}")
                continue
            ext = os.path.splitext(uf.name)[1].lower()
            if ext in (".mp3", ".wav", ".m4a"):
                try:
                    transcribed = process_audio(
                        uf.read(), uf.name,
                        progress_cb=lambda msg: st.info(msg)
                    )
                    uploaded_log.append(uf.name)
                    save_log(uploaded_log)
                    st.success(f"✅ Audio processed: {uf.name}")
                    with st.expander("📄 Transcript preview"):
                        st.write(transcribed[:500] + "...")
                except Exception as e:
                    st.error(f"❌ Audio error: {e}")
            else:
                try:
                    with st.spinner(f"Processing {uf.name}..."):
                        pages, chunks = process_file(uf.read(), uf.name)
                    uploaded_log.append(uf.name)
                    save_log(uploaded_log)
                    st.success(f"✅ {uf.name} — {pages} pages, {chunks} chunks")
                except Exception as e:
                    st.error(f"❌ File error: {e}")

    st.markdown("---")

    with st.expander("⚙️ Settings"):
        show_graph  = st.checkbox("Show knowledge graph", value=True)
        show_debug  = st.checkbox("Show debug panel", value=False)
        show_scores = st.checkbox("Show reranking scores", value=True)

        st.markdown("**⚡ Speed vs Quality (upload)**")
        enable_ctx = st.checkbox("Contextual chunking (+5s/doc)", value=True)
        enable_img = st.checkbox("Image extraction (+10s/image)", value=True)
        enable_grp = st.checkbox("Graph building (+5s/doc)", value=True)

        import rag_pipeline as rp
        rp.ENABLE_CONTEXTUAL_CHUNKING = enable_ctx
        rp.ENABLE_IMAGE_EXTRACTION    = enable_img
        rp.ENABLE_GRAPH_BUILDING      = enable_grp

    if st.button("🗑️ Clear everything", type="secondary"):
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            idx_name = os.environ.get("PINECONE_INDEX_NAME", "rag-index-v2")
            if idx_name in [i.name for i in pc.list_indexes()]:
                pc.delete_index(idx_name)
            if os.path.exists(UPLOADED_FILES_LOG):
                os.remove(UPLOADED_FILES_LOG)
            st.cache_resource.clear()
            st.success("Cleared! Please refresh the page.")
        except Exception as e:
            st.error(f"Clear error: {e}")


uploaded_log = load_log()

tab_chat, tab_graph, tab_eval = st.tabs(["💬 Chat", "🕸️ Knowledge Graph", "📊 Evaluation"])


with tab_chat:
    st.subheader("Ask anything across your files")

    if uploaded_log:
        cols = st.columns(len(uploaded_log) if len(uploaded_log) <= 4 else 4)
        for i, fname in enumerate(uploaded_log):
            with cols[i % 4]:
                ext = os.path.splitext(fname)[1].lower()
                icon = "🎙️" if ext in (".mp3", ".wav", ".m4a") else "📄"
                st.markdown(
                    f'<div class="source-badge">{icon} {fname}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning("⬅️ Upload at least one file from the sidebar to get started.")

    st.markdown("---")

    question = st.text_input(
        "Your question:",
        placeholder="e.g. What are the key findings in the report?",
        key="main_question",
    )

    col_ask, col_clear = st.columns([1, 5])
    with col_ask:
        ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

    if (ask_btn or question) and question:
        if not uploaded_log:
            st.error("Please upload a file first.")
        else:
            try:
                stats = get_pinecone_index().describe_index_stats()
                total = stats.get("total_vector_count", 0)
                if total == 0:
                    st.error("⚠️ Pinecone index is empty. Re-upload your files.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Cannot reach Pinecone: {e}")
                st.stop()

            with st.spinner("🔍 Hybrid search → Reranking → Generating answer..."):
                result = query_rag(question)

            answer     = result["result"]
            top_docs   = result["source_documents"]
            graph_ctx  = result["graph_context"]
            expanded_q = result.get("expanded_query", question)

            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            if graph_ctx:
                st.markdown(
                    f'<div class="graph-info">🕸️ <b>Knowledge Graph enriched this answer:</b><br>'
                    f'<code>{graph_ctx[:300]}</code></div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            col_src, col_chunks = st.columns(2)
            with col_src:
                st.markdown("**📎 Sources used:**")
                sources = {doc.metadata.get("source", "Unknown") for doc in top_docs}
                for s in sources:
                    st.write(f"• {s}")

            with col_chunks:
                st.markdown(f"**🔢 Chunks after reranking:** {len(top_docs)}")
                types = {}
                for doc in top_docs:
                    t = doc.metadata.get("type", "text")
                    types[t] = types.get(t, 0) + 1
                for t, cnt in types.items():
                    st.write(f"• {t}: {cnt}")

            if show_debug:
                with st.expander("🔬 Debug — Retrieved & Reranked Chunks"):
                    st.markdown(f"**Expanded query:**\n```\n{expanded_q}\n```")
                    for i, doc in enumerate(top_docs):
                        st.markdown(
                            f"**Chunk {i+1}** — `{doc.metadata.get('source')}` | "
                            f"type: `{doc.metadata.get('type', 'text')}`"
                        )
                        st.text(doc.page_content[:400])
                        st.markdown("---")


with tab_graph:
    st.subheader("🕸️ Knowledge Graph Explorer")
    from rag_pipeline import _graph as kg

    if kg.number_of_nodes() == 0:
        st.info("Upload documents to build the knowledge graph automatically.")
    else:
        col_info, col_search = st.columns(2)
        with col_info:
            st.metric("Entities (nodes)", kg.number_of_nodes())
            st.metric("Relations (edges)", kg.number_of_edges())

        with col_search:
            search_entity = st.text_input(
                "Search entity in graph:",
                placeholder="e.g. Python, Company Name ..."
            )
            if search_entity:
                if kg.has_node(search_entity):
                    neighbors = list(kg.neighbors(search_entity))
                    st.success(f"Found! Connected to: {', '.join(neighbors[:10])}")
                    for nb in neighbors[:10]:
                        rel = kg[search_entity][nb].get("relation", "related to")
                        st.write(f"  → **{search_entity}** `{rel}` **{nb}**")
                else:
                    close = [n for n in kg.nodes() if search_entity.lower() in n.lower()]
                    if close:
                        st.warning(f"Not found exactly. Similar: {', '.join(close[:5])}")
                    else:
                        st.error("Entity not found in graph.")

        st.markdown("---")
        st.markdown("**Graph Visualization** (top 50 nodes by degree)")

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        top_nodes = sorted(kg.degree(), key=lambda x: x[1], reverse=True)[:50]
        subgraph = kg.subgraph([n for n, _ in top_nodes])

        pos = nx.spring_layout(subgraph, seed=42, k=2)
        degrees = dict(subgraph.degree())
        node_sizes = [300 + degrees[n] * 100 for n in subgraph.nodes()]

        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color="#7c3aed", alpha=0.85, ax=ax)
        nx.draw_networkx_labels(subgraph, pos, font_color="white", font_size=7, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, edge_color="#4b5563", arrows=True, alpha=0.6, ax=ax)

        edge_labels = {(u, v): d.get("relation", "")[:15] for u, v, d in subgraph.edges(data=True)}
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color="#9ca3af", font_size=6, ax=ax)

        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📋 All relations (click to expand)"):
            rows = []
            for u, v, d in kg.edges(data=True):
                rows.append({
                    "Entity 1": u,
                    "Relation": d.get("relation", ""),
                    "Entity 2": v,
                    "Source": d.get("source", ""),
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ── TAB 3: EVALUATION ────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("📊 RAG Evaluation Dashboard")
    st.caption("Evaluate your RAG pipeline with faithfulness, answer relevancy, and correctness scores")

    if not uploaded_log:
        st.warning("Upload documents before running evaluation.")
    else:
        # ── Explanation of metrics ────────────────────────────────────────────
        with st.expander("ℹ️ How are these metrics calculated?", expanded=False):
            st.markdown("""
| Metric | What it measures | Uses ground truth? |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | ❌ No |
| **Answer Relevancy** | Does the answer address the question asked? | ❌ No |
| **Correctness** | Does the answer match the expected ground truth? | ✅ Yes — required |
| **Overall** | Weighted average (correctness = 50% when GT provided) | — |

> ⚠️ **Why ground truth matters:** Without a ground truth, the system can only check whether the  
> answer *sounds* relevant and is supported by context — it cannot catch factually wrong answers.  
> Always provide a ground truth to get meaningful scores.
            """)

        st.markdown("**Add test Q&A pairs below** (ground truth is optional but strongly recommended):")

        if "eval_cases" not in st.session_state:
            st.session_state.eval_cases = [{"question": "", "ground_truth": ""}]

        for i, case in enumerate(st.session_state.eval_cases):
            with st.expander(f"Test Case {i+1}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.eval_cases[i]["question"] = st.text_input(
                        "Question", value=case["question"], key=f"q_{i}"
                    )
                with c2:
                    st.session_state.eval_cases[i]["ground_truth"] = st.text_input(
                        "Ground Truth (strongly recommended)", value=case["ground_truth"], key=f"gt_{i}"
                    )

        col_add, col_run = st.columns(2)
        with col_add:
            if st.button("➕ Add test case"):
                st.session_state.eval_cases.append({"question": "", "ground_truth": ""})
                st.rerun()

        with col_run:
            run_eval = st.button("▶️ Run Evaluation", type="primary")

        if run_eval:
            valid_cases = [c for c in st.session_state.eval_cases if c["question"].strip()]
            if not valid_cases:
                st.error("Add at least one question.")
            else:
                with st.spinner(f"Running evaluation on {len(valid_cases)} test case(s)..."):
                    metrics = evaluate_rag(valid_cases)

                st.markdown("---")
                st.markdown("### Results")

                has_gt = metrics.get("has_ground_truth", False)

                if has_gt:
                    # Show all 4 metrics
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        faith = metrics["faithfulness"]
                        color = "🟢" if faith >= 0.7 else "🟡" if faith >= 0.4 else "🔴"
                        st.metric(f"{color} Faithfulness", f"{faith:.1%}")
                        st.caption("Grounded in context?")
                    with m2:
                        rel = metrics["answer_relevancy"]
                        color = "🟢" if rel >= 0.7 else "🟡" if rel >= 0.4 else "🔴"
                        st.metric(f"{color} Answer Relevancy", f"{rel:.1%}")
                        st.caption("Addresses the question?")
                    with m3:
                        correct = metrics["correctness"]
                        color = "🟢" if correct >= 0.7 else "🟡" if correct >= 0.4 else "🔴"
                        st.metric(f"{color} Correctness", f"{correct:.1%}")
                        st.caption("Matches ground truth?")
                    with m4:
                        overall = metrics["overall"]
                        color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.4 else "🔴"
                        st.metric(f"{color} Overall Score", f"{overall:.1%}")
                        st.caption("Weighted (GT = 50%)")
                    
                    st.info(
                        "💡 **Correctness** compares the predicted answer directly against your "
                        "ground truth. A wrong answer will score low here even if it sounds confident.",
                        icon="ℹ️",
                    )
                else:
                    # No ground truth — show only 3 metrics with a warning
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        faith = metrics["faithfulness"]
                        color = "🟢" if faith >= 0.7 else "🟡" if faith >= 0.4 else "🔴"
                        st.metric(f"{color} Faithfulness", f"{faith:.1%}")
                        st.caption("Grounded in context?")
                    with m2:
                        rel = metrics["answer_relevancy"]
                        color = "🟢" if rel >= 0.7 else "🟡" if rel >= 0.4 else "🔴"
                        st.metric(f"{color} Answer Relevancy", f"{rel:.1%}")
                        st.caption("Addresses the question?")
                    with m3:
                        overall = metrics["overall"]
                        color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.4 else "🔴"
                        st.metric(f"{color} Overall Score", f"{overall:.1%}")
                        st.caption("Average of both metrics")

                    st.warning(
                        "⚠️ No ground truth provided. **Correctness cannot be measured.** "
                        "The system can only check if answers sound relevant — it cannot detect "
                        "factually wrong answers. Add ground truths for meaningful evaluation.",
                        icon="⚠️",
                    )

                # Per-question breakdown
                st.markdown("### Per-question answers")
                for i, case in enumerate(valid_cases):
                    with st.expander(f"Q{i+1}: {case['question'][:60]}..."):
                        result = query_rag(case["question"])
                        st.markdown(f"**Answer:** {result['result']}")
                        if case["ground_truth"]:
                            st.markdown(f"**Ground Truth:** {case['ground_truth']}")

        st.markdown("---")
        with st.expander("💡 Evaluation Tips"):
            st.markdown("""
- **Always provide ground truths** — without them, a wrong answer can still score 100% on faithfulness/relevancy  
- **Faithfulness > 0.7** → answers are well-grounded in your documents  
- **Answer Relevancy > 0.7** → answers address what was asked  
- **Correctness > 0.7** → answers match the expected facts (requires ground truth)  
- Low faithfulness often means your chunks are too small or TOP_K is too low  
- Low correctness means the document may not contain the right information, or retrieval needs tuning  
- Add diverse test cases (specific facts, list questions, relationship questions)
            """)