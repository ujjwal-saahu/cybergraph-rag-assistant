import os
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="CyberGraph RAG",
    page_icon="🛡️",
    layout="wide",
)


st.title("🛡️ CyberGraph RAG")
st.caption(
    "Agentic Cybersecurity Knowledge Assistant powered by FastAPI, Qdrant, LangChain, Ollama, and agentic RAG checks."
)


with st.sidebar:
    st.header("System Status")

    if st.button("Check Backend Health"):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=20)
            if response.status_code == 200:
                st.success("Backend is running")
                st.json(response.json())
            else:
                st.error("Backend health check failed")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")

    st.divider()

    st.header("Vector DB")

    if st.button("Check Vector DB"):
        try:
            response = requests.get(f"{BACKEND_URL}/documents/vector-db/info", timeout=20)
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error("Could not fetch vector DB info")
        except Exception as e:
            st.error(f"Error: {e}")


tab_upload, tab_chat, tab_documents, tab_workflow = st.tabs(
    ["📄 Upload Documents", "💬 Ask Questions", "📚 Documents", "🧠 Workflow"]
)


with tab_upload:
    st.subheader("Upload PDF / TXT / Markdown / DOCX / CSV")

    uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "txt", "md", "markdown", "docx", "csv"],
)

    if uploaded_file is not None:
        st.info(f"Selected file: {uploaded_file.name}")

        if st.button("Upload and Index Document"):
            with st.spinner("Uploading, chunking, embedding, and indexing document..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }

                    response = requests.post(
                        f"{BACKEND_URL}/documents/upload",
                        files=files,
                        timeout=300,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("Document processed and indexed successfully")

                        col1, col2, col3 = st.columns(3)

                        chunking = result.get("chunking", {})

                        col1.metric(
                            "Parent Chunks",
                            chunking.get("parent_chunks", 0),
                        )
                        col2.metric(
                            "Child Chunks",
                            chunking.get("child_chunks", 0),
                        )
                        col3.metric(
                            "Indexed Chunks",
                            chunking.get("saved_child_chunks_to_qdrant", 0),
                        )

                        with st.expander("Document Preview"):
                            st.write(result.get("preview", ""))

                        with st.expander("Full Upload Response"):
                            st.json(result)

                    else:
                        st.error("Upload failed")
                        st.write(response.text)

                except Exception as e:
                    st.error(f"Upload error: {e}")


with tab_chat:
    st.subheader("Ask a Question from Uploaded Documents")

    question = st.text_area(
        "Your question",
        placeholder="Example: What is his cybersecurity experience?",
        height=100,
    )

    top_k = st.slider(
        "Top-K child chunks for retrieval",
        min_value=1,
        max_value=10,
        value=3,
    )

    if st.button("Ask CyberGraph RAG"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Running agentic RAG pipeline..."):
                try:
                    payload = {
                        "question": question,
                        "top_k": top_k,
                    }

                    response = requests.post(
                        f"{BACKEND_URL}/chat/",
                        json=payload,
                        timeout=600,
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.success("Answer generated")

                        st.markdown("## Answer")
                        st.write(result.get("answer", ""))

                        st.divider()

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Parent Contexts",
                            result.get("parent_context_count", 0),
                        )
                        col2.metric(
                            "Relevant Contexts",
                            result.get("relevant_context_count", 0),
                        )
                        col3.metric(
                            "Answer Regenerated",
                            str(result.get("answer_regenerated", False)),
                        )

                        with st.expander("Rewritten Query"):
                            st.write(result.get("rewritten_query", ""))

                        with st.expander("Sources"):
                            sources = result.get("sources", [])
                            if sources:
                                st.json(sources)
                            else:
                                st.info("No sources returned.")

                        with st.expander("Relevance Grades"):
                            grades = result.get("relevance_grades", [])
                            if grades:
                                st.json(grades)
                            else:
                                st.info("No relevance grades returned.")

                        with st.expander("Hallucination Check"):
                            hallucination_check = result.get("hallucination_check", {})
                            st.json(hallucination_check)

                        with st.expander("Full Response JSON"):
                            st.json(result)

                    else:
                        st.error("Chat request failed")
                        st.write(response.text)

                except Exception as e:
                    st.error(f"Chat error: {e}")


with tab_documents:
    st.subheader("Processed Documents")

    if st.button("Refresh Documents"):
        try:
            response = requests.get(f"{BACKEND_URL}/documents/", timeout=60)

            if response.status_code == 200:
                result = response.json()
                documents = result.get("documents", [])

                if not documents:
                    st.info("No processed documents found.")
                else:
                    st.write(f"Total processed documents: {len(documents)}")

                    for doc in documents:
                        document_id = doc.get("document_id", "")
                        filename = doc.get("filename", "Unknown document")

                        with st.expander(f"{filename} | Document ID: {document_id}"):
                            st.write(f"Document ID: `{document_id}`")
                            st.write(f"Path: `{doc.get('path')}`")
                            st.write(f"Characters: {doc.get('characters')}")
                            st.write(doc.get("preview", ""))

            else:
                st.error("Could not fetch documents")
                st.write(response.text)

        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    st.subheader("Delete Document")

    st.warning(
        "Deleting a document will remove its uploaded file, Markdown file, parent chunks, "
        "and rebuild the vector index from remaining documents."
    )

    delete_document_id = st.text_input(
        "Enter document_id to delete",
        placeholder="Copy document_id from the processed documents list above",
    )

    if st.button("Delete Document and Re-index"):
        if not delete_document_id.strip():
            st.warning("Please enter a document_id.")
        else:
            with st.spinner("Deleting document and rebuilding index..."):
                try:
                    response = requests.delete(
                        f"{BACKEND_URL}/documents/{delete_document_id.strip()}",
                        timeout=600,
                    )

                    if response.status_code == 200:
                        st.success("Document deleted and index rebuilt.")
                        st.json(response.json())
                    else:
                        st.error("Delete failed.")
                        st.write(response.text)

                except Exception as e:
                    st.error(f"Delete error: {e}")

    st.divider()

    st.subheader("Re-index All Documents")

    st.info(
        "Use this if the vector database becomes inconsistent or if you want to rebuild "
        "the full index from all remaining Markdown files."
    )

    if st.button("Re-index All Remaining Documents"):
        with st.spinner("Rebuilding parent chunks and vector index..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/documents/reindex",
                    timeout=600,
                )

                if response.status_code == 200:
                    st.success("Re-indexing completed.")
                    st.json(response.json())
                else:
                    st.error("Re-indexing failed.")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Re-index error: {e}")

    st.divider()

    st.subheader("Parent Chunks")

    if st.button("Refresh Parent Chunks"):
        try:
            response = requests.get(
                f"{BACKEND_URL}/documents/parent-chunks",
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                parent_chunks = result.get("parent_chunks", [])

                if not parent_chunks:
                    st.info("No parent chunks found.")
                else:
                    st.write(f"Total parent chunks: {len(parent_chunks)}")

                    for chunk in parent_chunks[:20]:
                        source = chunk.get("source", "Unknown source")
                        parent_id = chunk.get("parent_id", "Unknown parent ID")
                        document_id = chunk.get("document_id", "Unknown document ID")

                        with st.expander(
                            f"{source} | Document ID: {document_id} | Parent ID: {parent_id}"
                        ):
                            st.write(f"Document ID: `{document_id}`")
                            st.write(f"Parent ID: `{parent_id}`")
                            st.write(f"Characters: {chunk.get('characters')}")
                            st.write(chunk.get("preview", ""))

            else:
                st.error("Could not fetch parent chunks")
                st.write(response.text)

        except Exception as e:
            st.error(f"Error: {e}")


with tab_workflow:
    st.subheader("CyberGraph RAG Agentic Workflow")

    st.caption(
        "This workflow shows how the system rewrites queries, retrieves parent-child context, "
        "grades relevance, checks hallucination, and returns a grounded answer."
    )

    if st.button("Load Workflow Diagram"):
        try:
            response = requests.get(f"{BACKEND_URL}/workflow/graphviz", timeout=30)

            if response.status_code == 200:
                result = response.json()
                diagram = result.get("diagram", "")

                st.graphviz_chart(diagram)

            else:
                st.error("Could not load workflow diagram.")

        except Exception as e:
            st.error(f"Workflow loading error: {e}")

    st.divider()

    if st.button("Load Workflow Steps"):
        try:
            response = requests.get(f"{BACKEND_URL}/workflow/steps", timeout=30)

            if response.status_code == 200:
                result = response.json()

                steps = result.get("steps", [])
                edges = result.get("edges", [])

                st.markdown("### Workflow Steps")

                for index, step in enumerate(steps, start=1):
                    with st.expander(f"{index}. {step.get('name')}"):
                        st.write(step.get("description"))

                st.markdown("### Workflow Edges")
                st.json(edges)

            else:
                st.error("Could not load workflow steps.")

        except Exception as e:
            st.error(f"Workflow steps loading error: {e}")

    st.divider()

    if st.button("Load Mermaid Diagram for README"):
        try:
            response = requests.get(f"{BACKEND_URL}/workflow/mermaid", timeout=30)

            if response.status_code == 200:
                result = response.json()
                mermaid = result.get("diagram", "")

                st.code(mermaid, language="markdown")

            else:
                st.error("Could not load Mermaid diagram.")

        except Exception as e:
            st.error(f"Mermaid loading error: {e}")          