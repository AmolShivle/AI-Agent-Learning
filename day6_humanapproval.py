
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# --------------------------------------------------
# 1. Pipeline State
# --------------------------------------------------

class PipelineState(TypedDict):
    pipeline_name: str
    status: str
    error_message: str
    recommendation: str
    retry_approved: bool
    retry_result: str
    final_response: str


# --------------------------------------------------
# 2. Mock Pipeline Repository
# --------------------------------------------------

pipelines = {
    "customer_etl": {
        "status": "FAILED",
        "error": "Database connection timeout"
    },
    "sales_etl": {
        "status": "FAILED",
        "error": "Out of memory error"
    },
    "netflix_etl": {
        "status": "SUCCESS",
        "error": ""
    }
}


# --------------------------------------------------
# 3. Check Pipeline
# --------------------------------------------------

def check_pipeline(state: PipelineState):

    pipeline_name = state["pipeline_name"].strip().lower()

    pipeline = pipelines.get(pipeline_name)

    if pipeline is None:
        return {
            "status": "NOT_FOUND",
            "error_message": "Pipeline not found"
        }

    return {
        "status": pipeline["status"],
        "error_message": pipeline["error"]
    }


# --------------------------------------------------
# 4. Conditional Routing
# --------------------------------------------------

def route_pipeline(state: PipelineState):

    if state["status"] == "FAILED":
        return "analyze_failure"

    return "generate_response"


# --------------------------------------------------
# 5. Analyze Failure
# --------------------------------------------------

def analyze_failure(state: PipelineState):

    error = state["error_message"]

    if "timeout" in error.lower():
        recommendation = (
            "Check database connectivity and credentials."
        )

    elif "memory" in error.lower():
        recommendation = (
            "Review Spark executor memory and partition configuration."
        )

    else:
        recommendation = "Check pipeline logs and investigate the error."

    print("\nFailure Analysis:")
    print("Error:", error)
    print("Recommendation:", recommendation)

    return {
        "recommendation": recommendation
    }


# --------------------------------------------------
# 6. Human Approval
# --------------------------------------------------

def human_approval(state: PipelineState):

    decision = interrupt({
        "message": "Pipeline failed. Approve retry?",
        "pipeline": state["pipeline_name"],
        "error": state["error_message"],
        "recommendation": state["recommendation"]
    })

    return {
        "retry_approved": bool(decision)
    }


# --------------------------------------------------
# 7. Retry Pipeline
# --------------------------------------------------

def retry_pipeline(state: PipelineState):

    pipeline_name = state["pipeline_name"].strip().lower()

    if not state["retry_approved"]:

        print("\nRetry rejected by human.")

        return {
            "retry_result": "Retry rejected by human."
        }

    print("\nRetrying pipeline:", pipeline_name)

    # Simulate successful retry
    pipelines[pipeline_name]["status"] = "SUCCESS"

    return {
        "status": "SUCCESS",
        "retry_result": "Pipeline retry successful."
    }


# --------------------------------------------------
# 8. Generate Final Response
# --------------------------------------------------

def generate_response(state: PipelineState):

    if state["status"] == "SUCCESS":

        if state["retry_result"]:
            message = (
                f"{state['pipeline_name']} completed successfully. "
                f"{state['retry_result']}"
            )
        else:
            message = (
                f"{state['pipeline_name']} is already successful. "
                "No retry required."
            )

    elif state["status"] == "FAILED":

        message = (
            f"{state['pipeline_name']} remains FAILED. "
            f"{state['retry_result']}"
        )

    else:
        message = (
            f"{state['pipeline_name']} status: {state['status']}. "
            f"{state['error_message']}"
        )

    return {
        "final_response": message
    }


# --------------------------------------------------
# 9. Build LangGraph
# --------------------------------------------------

builder = StateGraph(PipelineState)

builder.add_node("check_pipeline", check_pipeline)
builder.add_node("analyze_failure", analyze_failure)
builder.add_node("human_approval", human_approval)
builder.add_node("retry_pipeline", retry_pipeline)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "check_pipeline")

builder.add_conditional_edges(
    "check_pipeline",
    route_pipeline,
    {
        "analyze_failure": "analyze_failure",
        "generate_response": "generate_response"
    }
)

builder.add_edge("analyze_failure", "human_approval")
builder.add_edge("human_approval", "retry_pipeline")
builder.add_edge("retry_pipeline", "generate_response")
builder.add_edge("generate_response", END)


# --------------------------------------------------
# 10. Compile Graph with Checkpointer
# --------------------------------------------------

checkpointer = InMemorySaver()

app = builder.compile(checkpointer=checkpointer)


# --------------------------------------------------
# 11. Execute Pipeline
# --------------------------------------------------

def run_pipeline(pipeline_name: str, approved: bool):

    print("\n" + "=" * 50)
    print("Starting:", pipeline_name)
    print("=" * 50)

    config = {
        "configurable": {
            "thread_id": f"{pipeline_name}-001"
        }
    }

    initial_state = {
        "pipeline_name": pipeline_name,
        "status": "",
        "error_message": "",
        "recommendation": "",
        "retry_approved": False,
        "retry_result": "",
        "final_response": ""
    }

    # First execution
    result = app.invoke(initial_state, config=config)

    # Check if human approval is required
    if result.get("__interrupt__"):

        print("\nHuman Approval Required:")
        print(result["__interrupt__"])

        decision = approved

        print("\nHuman Decision:", decision)

        # Resume the same workflow
        result = app.invoke(
            Command(resume=decision),
            config=config
        )

    print("\nFinal Result:")
    print(result["final_response"])


# --------------------------------------------------
# 12. Main Program
# --------------------------------------------------

if __name__ == "__main__":

    run_pipeline("customer_etl", approved=True)

    run_pipeline("sales_etl", approved=False)

    run_pipeline("netflix_etl", approved=False)
