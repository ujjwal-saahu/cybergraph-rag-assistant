class WorkflowVisualizerService:
    """
    Provides workflow metadata and visualization formats
    for the CyberGraph RAG agentic pipeline.
    """

    def get_workflow_steps(self) -> list[dict]:
        """
        Return ordered workflow steps.
        """

        return [
            {
                "id": "user_question",
                "name": "User Question",
                "description": "The user asks a question from uploaded documents.",
            },
            {
                "id": "query_rewrite",
                "name": "Query Rewriting Agent",
                "description": "Rewrites the user question into a clearer retrieval query.",
            },
            {
                "id": "child_search",
                "name": "Child Chunk Search",
                "description": "Searches smaller child chunks in Qdrant vector database.",
            },
            {
                "id": "parent_retrieval",
                "name": "Parent Context Retrieval",
                "description": "Loads larger parent chunks using parent_id from child matches.",
            },
            {
                "id": "relevance_grading",
                "name": "Relevance Grading Agent",
                "description": "Checks whether retrieved contexts are useful for answering the question.",
            },
            {
                "id": "answer_generation",
                "name": "Grounded Answer Generation",
                "description": "Generates an answer using only relevant retrieved context.",
            },
            {
                "id": "hallucination_check",
                "name": "Hallucination Checker Agent",
                "description": "Verifies whether the answer is fully supported by retrieved context.",
            },
            {
                "id": "safe_regeneration",
                "name": "Optional Safer Regeneration",
                "description": "Regenerates the answer if unsupported claims are detected.",
            },
            {
                "id": "final_answer",
                "name": "Final Answer with Sources",
                "description": "Returns the final answer, sources, relevance grades, and hallucination check.",
            },
        ]

    def get_edges(self) -> list[dict]:
        """
        Return graph edges.
        """

        return [
            {"from": "user_question", "to": "query_rewrite"},
            {"from": "query_rewrite", "to": "child_search"},
            {"from": "child_search", "to": "parent_retrieval"},
            {"from": "parent_retrieval", "to": "relevance_grading"},
            {"from": "relevance_grading", "to": "answer_generation"},
            {"from": "answer_generation", "to": "hallucination_check"},
            {"from": "hallucination_check", "to": "safe_regeneration"},
            {"from": "safe_regeneration", "to": "final_answer"},
        ]

    def get_graphviz_dot(self) -> str:
        """
        Return Graphviz DOT format for Streamlit visualization.
        """

        return """
digraph CyberGraphRAG {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11];

    user_question [label="User Question", fillcolor="#E8F1FF"];
    query_rewrite [label="Query Rewriting Agent", fillcolor="#FFF4D6"];
    child_search [label="Qdrant Child Chunk Search", fillcolor="#EAF7EA"];
    parent_retrieval [label="Parent Context Retrieval", fillcolor="#EAF7EA"];
    relevance_grading [label="Relevance Grading Agent", fillcolor="#FFF4D6"];
    answer_generation [label="Grounded Answer Generation", fillcolor="#E8F1FF"];
    hallucination_check [label="Hallucination Checker Agent", fillcolor="#FFE8E8"];
    safe_regeneration [label="Optional Safer Regeneration", fillcolor="#FFE8E8"];
    final_answer [label="Final Answer + Sources", fillcolor="#EDE8FF"];

    user_question -> query_rewrite;
    query_rewrite -> child_search;
    child_search -> parent_retrieval;
    parent_retrieval -> relevance_grading;
    relevance_grading -> answer_generation;
    answer_generation -> hallucination_check;
    hallucination_check -> safe_regeneration [label="if unsupported"];
    hallucination_check -> final_answer [label="if grounded"];
    safe_regeneration -> final_answer;
}
"""

    def get_mermaid(self) -> str:
        """
        Return Mermaid diagram for README documentation.
        """

        return """
flowchart TD
    A[User Question]
    B[Query Rewriting Agent]
    C[Qdrant Child Chunk Search]
    D[Parent Context Retrieval]
    E[Relevance Grading Agent]
    F[Grounded Answer Generation]
    G[Hallucination Checker Agent]
    H[Optional Safer Regeneration]
    I[Final Answer with Sources]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G -->|Grounded| I
    G -->|Unsupported Claims| H
    H --> I
"""