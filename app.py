"""
app.py  -  Enhanced RAG Multimodal Assistant (hardened)
"""

import os
import json
import html

import streamlit as st
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import rag_pipeline as rp
from rag_pipeline import (
    get_pinecone_index,
    get_total_vectors,
    process_file,
    process_audio,
    query_rag,
    evaluate_rag,
    reset_all,
    missing_config,
)

UPLOADED_FILES_LOG = "uploaded_files.json"
AUDIO_EXTS = (".mp3", ".wav", ".m4a")

st.set_page_config(
    page_title="RAG Assistant Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.source-badge {
    background: #374151; color: #d1d5db; padding: 2px 8px;
    border-radius: 12px; font-size: 0.8em; margin: 2px; display: inline-block;
}
.graph-info {
    background: #064e3b; border-radius: 8px; padding: 10px;
    border-left: 3px solid #10b981;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 RAG Multimodal Assistant Pro")
st.caption("Hybrid Search · Cross-Encoder Reranking · Graph RAG · Vision · Evaluation")

# ── Fail early with a readable message if secrets are missing ────────────────
_missing = missing_config()
if _missing:
    st.error(
        f"Missing secrets: {', '.join(_missing)}. "
        "On Streamlit Cloud open **Manage app → Settings → Secrets** and add them."
    )
    st.stop()


def load_log():
    try:
        if os.path.exists(UPLOADED_FILES_LOG):
            with open(UPLOADED_FILES_LOG, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def save_log(files):
    try:
        with open(UPLOADED_FILES_LOG, "w") as f:
            json.dump(files, f)
    except Exception:
        pass


def icon_for(fname: str) -> str:
    return "🎙️" if os.path.splitext(fname)[1].lower() in AUDIO_EXTS else "📄"


# ═════════════════════════════ SIDEBAR ═══════════════════════════════════════
with st.sidebar:
    st.title("📁 File Manager")

    st.markdown("### 🔌 Pinecone Status")
    try:
        total = get_total_vectors()
        st.success(f"✅ Connected - {total} vectors stored")
        st.caption(f"Index: `{rp.INDEX_NAME}`")
    except Exception as e:
        st.error(f"❌ Pinecone error: {e}")

    if rp._graph.number_of_nodes() > 0:
        st.success(f"🕸️ Graph: {rp._graph.number_of_nodes()} entities, "
                   f"{rp._graph.number_of_edges()} relations")

    st.markdown("---")

    uploaded_log = load_log()
    if uploaded_log:
        st.markdown("**📄 Uploaded files:**")
        for fname in uploaded_log:
            st.write(f"{icon_for(fname)} {fname}")
    else:
        st.info("No files uploaded yet.")

    st.markdown("---")

    with st.expander("⚙️ Settings"):
        show_debug = st.checkbox("Show debug panel", value=False)
        st.markdown("**⚡ Speed vs Quality (upload)**")
        rp.ENABLE_CONTEXTUAL_CHUNKING = st.checkbox("Contextual chunking (+5s/doc)", value=True)
        rp.ENABLE_IMAGE_EXTRACTION = st.checkbox("Image extraction (+10s/image)", value=True)
        rp.ENABLE_GRAPH_BUILDING = st.checkbox("Graph building (+5s/doc)", value=True)

    new_files = st.file_uploader(
        "Upload PDFs, Word docs, or Audio",
        type=["pdf", "docx", "mp3", "wav", "m4a"],
        accept_multiple_files=True,
    )

    if new_files:
        for uf in new_files:
            if uf.name in uploaded_log:
                continue
            ext = os.path.splitext(uf.name)[1].lower()
            try:
                if ext in AUDIO_EXTS:
                    with st.spinner(f"Transcribing {uf.name}..."):
                        transcribed = process_audio(uf.getvalue(), uf.name)
                    uploaded_log.append(uf.name)
                    save_log(uploaded_log)
                    st.success(f"✅ Audio processed: {uf.name}")
                    with st.expander("📄 Transcript preview"):
                        st.write(transcribed[:500] + "...")
                else:
                    with st.spinner(f"Processing {uf.name}..."):
                        pages, chunks = process_file(uf.getvalue(), uf.name)
                    uploaded_log.append(uf.name)
                    save_log(uploaded_log)
                    st.success(f"✅ {uf.name} - {pages} pages, {chunks} chunks")
            except Exception as e:
                st.error(f"❌ Could not process {uf.name}: {e}")

    st.markdown("---")

    if st.button("🗑️ Clear everything", type="secondary"):
        try:
            reset_all()
            if os.path.exists(UPLOADED_FILES_LOG):
                os.remove(UPLOADED_FILES_LOG)
            for k in ("last_question", "last_result", "eval_cases"):
                st.session_state.pop(k, None)
            st.success("Cleared! Refresh the page.")
        except Exception as e:
            st.error(f"Clear error: {e}")


uploaded_log = load_log()
tab_chat, tab_graph, tab_eval = st.tabs(["💬 Chat", "🕸️ Knowledge Graph", "📊 Evaluation"])


# ═════════════════════════════ CHAT TAB ══════════════════════════════════════
with tab_chat:
    st.subheader("Ask anything across your files")

    if uploaded_log:
        cols = st.columns(min(len(uploaded_log), 4))
        for i, fname in enumerate(uploaded_log):
            with cols[i % len(cols)]:
                st.markdown(
                    f'<div class="source-badge">{icon_for(fname)} {html.escape(fname)}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("⬅️ Upload at least one file from the sidebar to get started.")

    st.markdown("---")

    question = st.text_input(
        "Your question:",
        placeholder="e.g. What are the key findings in the report?",
        key="main_question",
    ).strip()

    ask_btn = st.button("🔍 Ask", type="primary")

    # Run only on button click or a NEW question (not on every unrelated rerun)
    is_new = question and question != st.session_state.get("last_question")
    if question and (ask_btn or is_new):
        if not uploaded_log:
            st.error("Please upload a file first.")
        else:
            with st.spinner("🔍 Hybrid search → Reranking → Generating answer..."):
                try:
                    st.session_state["last_result"] = query_rag(question)
                except Exception as e:  # last line of defence
                    st.session_state["last_result"] = {
                        "result": f"Unexpected error: {e}",
                        "source_documents": [], "graph_context": "",
                        "expanded_query": question,
                    }
                st.session_state["last_question"] = question

    result = st.session_state.get("last_result")
    if question and result and st.session_state.get("last_question") == question:
        answer = result["result"]
        top_docs = result["source_documents"]
        graph_ctx = result["graph_context"]
        expanded_q = result.get("expanded_query", question)

        st.markdown("### 💡 Answer")
        with st.container(border=True):
            st.markdown(answer)   # markdown only - no raw HTML injection

        if graph_ctx:
            st.markdown(
                '<div class="graph-info">🕸️ <b>Knowledge Graph enriched this answer:</b><br>'
                f'<code>{html.escape(graph_ctx[:300])}</code></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col_src, col_chunks = st.columns(2)
        with col_src:
            st.markdown("**📎 Sources used:**")
            for s in {d.metadata.get("source", "Unknown") for d in top_docs}:
                st.write(f"• {s}")
        with col_chunks:
            st.markdown(f"**🔢 Chunks after reranking:** {len(top_docs)}")
            types = {}
            for d in top_docs:
                t = d.metadata.get("type", "text")
                types[t] = types.get(t, 0) + 1
            for t, cnt in types.items():
                st.write(f"• {t}: {cnt}")

        if show_debug:
            with st.expander("🔬 Debug - Retrieved & Reranked Chunks"):
                st.markdown(f"**Expanded query:**\n```\n{expanded_q}\n```")
                for i, d in enumerate(top_docs):
                    st.markdown(
                        f"**Chunk {i+1}** - `{d.metadata.get('source')}` | "
                        f"type: `{d.metadata.get('type', 'text')}`"
                    )
                    st.text(d.page_content[:400])
                    st.markdown("---")


# ═════════════════════════════ GRAPH TAB ═════════════════════════════════════
with tab_graph:
    st.subheader("🕸️ Knowledge Graph Explorer")
    kg = rp._graph

    if kg.number_of_nodes() == 0:
        st.info("Upload documents to build the knowledge graph automatically. "
                "(The graph lives in memory and resets when the app restarts.)")
    else:
        col_info, col_search = st.columns(2)
        with col_info:
            st.metric("Entities (nodes)", kg.number_of_nodes())
            st.metric("Relations (edges)", kg.number_of_edges())

        with col_search:
            search_entity = st.text_input("Search entity in graph:",
                                          placeholder="e.g. Python, Company Name ...")
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
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")

            top_nodes = sorted(kg.degree(), key=lambda x: x[1], reverse=True)[:50]
            subgraph = kg.subgraph([n for n, _ in top_nodes])
            pos = nx.spring_layout(subgraph, seed=42, k=2)
            degrees = dict(subgraph.degree())
            node_sizes = [300 + degrees[n] * 100 for n in subgraph.nodes()]

            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                                   node_color="#7c3aed", alpha=0.85, ax=ax)
            nx.draw_networkx_labels(subgraph, pos, font_color="white", font_size=7, ax=ax)
            nx.draw_networkx_edges(subgraph, pos, edge_color="#4b5563", alpha=0.6, ax=ax)
            edge_labels = {(u, v): d.get("relation", "")[:15]
                           for u, v, d in subgraph.edges(data=True)}
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels,
                                         font_color="#9ca3af", font_size=6, ax=ax)
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not draw graph: {e}")

        with st.expander("📋 All relations (click to expand)"):
            rows = [{"Entity 1": u, "Relation": d.get("relation", ""),
                     "Entity 2": v, "Source": d.get("source", "")}
                    for u, v, d in kg.edges(data=True)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ═════════════════════════════ EVAL TAB ══════════════════════════════════════
def _color(x: float) -> str:
    return "🟢" if x >= 0.7 else "🟡" if x >= 0.4 else "🔴"


with tab_eval:
    st.subheader("📊 RAG Evaluation Dashboard")
    st.caption("Faithfulness, answer relevancy, and correctness (LLM-judged)")

    if not uploaded_log:
        st.warning("Upload documents before running evaluation.")
    else:
        with st.expander("ℹ️ How are these metrics calculated?"):
            st.markdown("""
| Metric | What it measures | Uses ground truth? |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | No |
| **Answer Relevancy** | Does the answer address the question asked? | No |
| **Correctness** | Does the answer match the expected ground truth? | Yes (required) |
| **Overall** | Weighted average (correctness = 50% when GT provided) | - |

Without a ground truth the system cannot catch factually wrong answers, so always provide one.
            """)

        st.markdown("**Add test Q&A pairs below** (ground truth is optional but strongly recommended):")

        if "eval_cases" not in st.session_state:
            st.session_state.eval_cases = [{"question": "", "ground_truth": ""}]

        for i, case in enumerate(st.session_state.eval_cases):
            with st.expander(f"Test Case {i+1}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.eval_cases[i]["question"] = st.text_input(
                        "Question", value=case["question"], key=f"q_{i}")
                with c2:
                    st.session_state.eval_cases[i]["ground_truth"] = st.text_input(
                        "Ground Truth (strongly recommended)",
                        value=case["ground_truth"], key=f"gt_{i}")

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
                try:
                    with st.spinner(f"Running evaluation on {len(valid_cases)} test case(s)..."):
                        metrics = evaluate_rag(valid_cases)
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    st.stop()

                st.markdown("---")
                st.markdown("### Results")
                has_gt = metrics.get("has_ground_truth", False)

                items = [("Faithfulness", metrics["faithfulness"], "Grounded in context?"),
                         ("Answer Relevancy", metrics["answer_relevancy"], "Addresses the question?")]
                if has_gt:
                    items.append(("Correctness", metrics["correctness"], "Matches ground truth?"))
                items.append(("Overall Score", metrics["overall"],
                              "Weighted (GT = 50%)" if has_gt else "Average of both metrics"))

                for col, (label, val, cap) in zip(st.columns(len(items)), items):
                    with col:
                        st.metric(f"{_color(val)} {label}", f"{val:.1%}")
                        st.caption(cap)

                if not has_gt:
                    st.warning("No ground truth provided - correctness cannot be measured.")

                st.markdown("### Per-question answers")
                for i, d in enumerate(metrics.get("details", [])):
                    with st.expander(f"Q{i+1}: {d['question'][:60]}"):
                        st.markdown(f"**Answer:** {d['answer']}")
                        if d["ground_truth"]:
                            st.markdown(f"**Ground Truth:** {d['ground_truth']}")

        with st.expander("💡 Evaluation Tips"):
            st.markdown("""
- **Always provide ground truths**; without them a wrong answer can still score high.
- **Faithfulness > 0.7** means answers are well grounded in your documents.
- **Answer Relevancy > 0.7** means answers address what was asked.
- **Correctness > 0.7** means answers match the expected facts.
- Add diverse test cases (specific facts, list questions, relationship questions).
            """)