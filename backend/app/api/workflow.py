from fastapi import APIRouter

from app.services.workflow_visualizer_service import WorkflowVisualizerService

router = APIRouter(prefix="/workflow", tags=["Workflow"])

workflow_service = WorkflowVisualizerService()


@router.get("/steps")
def get_workflow_steps():
    """
    Return agentic RAG workflow steps and edges.
    """

    return {
        "name": "CyberGraph RAG Agentic Workflow",
        "steps": workflow_service.get_workflow_steps(),
        "edges": workflow_service.get_edges(),
    }


@router.get("/graphviz")
def get_graphviz_workflow():
    """
    Return Graphviz DOT workflow diagram.
    """

    return {
        "format": "graphviz",
        "diagram": workflow_service.get_graphviz_dot(),
    }


@router.get("/mermaid")
def get_mermaid_workflow():
    """
    Return Mermaid workflow diagram for README/docs.
    """

    return {
        "format": "mermaid",
        "diagram": workflow_service.get_mermaid(),
    }