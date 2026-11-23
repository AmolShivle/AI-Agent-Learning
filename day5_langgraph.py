
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

# 1. Define shared state
class PipelineState(TypedDict):
    pipeline_name: str
    status: str
    error_message: str
    recommendation: str
    final_response: str
    retry_count: int
    retry_approved: bool
    retry_result: str



# 2. Mock pipeline database
pipelines = {
    "netflix_ingestion": "SUCCESS",
    "customer_etl": "FAILED",
    "sales_pipeline": "RUNNING"
}

# 3. Node: Check pipeline status
def check_pipeline(state: PipelineState):

    print("\nChecking pipeline...")

    name = state["pipeline_name"].strip().lower()

    status = pipelines.get(name, "NOT_FOUND")

    error_message = ""

    if status == "FAILED":
        error_message = "Database connection timeout"

    print("Pipeline status:", status)

    return {
        "status": status,
        "error_message": error_message
    }

# 4. Node: Analyze failure
def analyze_failure(state: PipelineState):

    print("\nAnalyzing pipeline failure...")

    error = state["error_message"]

    if "connection" in error.lower():

        recommendation = (
            "Check database connectivity, "
            "credentials and network access."
        )

    else:

        recommendation = (
            "Review pipeline logs and identify "
            "the root cause."
        )

    return {
        "recommendation": recommendation
    }

# 5. Node: Generate final response

def generate_response(state: PipelineState):

    status = state["status"]

    if status == "SUCCESS":

        if state["retry_result"]:
            response = (
                f"Pipeline: {state['pipeline_name']}\n"
                f"Status: SUCCESS\n"
                f"Retry Result: {state['retry_result']}\n"
                f"Retry Attempts: {state['retry_count']}"
            )
        else:
            response = (
                f"Pipeline {state['pipeline_name']} "
                "completed successfully."
            )

    elif status == "FAILED":

        response = (
            f"Pipeline: {state['pipeline_name']}\n"
            f"Status: FAILED\n"
            f"Error: {state['error_message']}\n"
            f"Retry Result: {state['retry_result']}\n"
            f"Recommendation: {state['recommendation']}"
        )

    elif status == "RUNNING":

        response = (
            f"Pipeline {state['pipeline_name']} "
            "is currently running."
        )

    else:

        response = (
            f"Pipeline {state['pipeline_name']} "
            "does not exist."
        )

    return {"final_response": response}

# 6. Conditional routing
def route_pipeline(state: PipelineState):

    if state["status"] == "FAILED":
        return "analyze_failure"

    return "generate_response"



def retry_pipeline(state: PipelineState):

    print("\nRetrying pipeline...")

    # Check retry approval
    if not state["retry_approved"]:

        return {
            "retry_result": "Retry not approved",
            "status": "FAILED"
        }

    #Check maximum retry attempts
    if state["retry_count"] >= 1:

        return {
            "retry_result": "Maximum retry attempts reached",
            "status": "FAILED"
        }

    # Simulate retry execution
    retry_count = state["retry_count"] + 1

    print("Retry attempt:", retry_count)

    # Mock successful retry
    new_status = "SUCCESS"

    return {
        "retry_count": retry_count,
        "status": new_status,
        "retry_result": "Pipeline retry successful",
        "error_message": ""
    }


# 7. Create StateGraph
builder = StateGraph(PipelineState)

# 8. Register nodes
builder.add_node("check_pipeline", check_pipeline)
builder.add_node("analyze_failure", analyze_failure)
builder.add_node("generate_response", generate_response)
builder.add_node("retry_pipeline", retry_pipeline)

# 9. Define edges

# Starting point
builder.add_edge(START, "check_pipeline")

# Conditional routing after checking status
builder.add_conditional_edges(
    "check_pipeline",
    route_pipeline,
    {
        "analyze_failure": "analyze_failure",
        "generate_response": "generate_response"
    }
)

# Failure analysis -> Retry
builder.add_edge(
    "analyze_failure",
    "retry_pipeline"
)

# Retry -> Final response
builder.add_edge("retry_pipeline",
    "generate_response"
)

# Final response -> END
builder.add_edge("generate_response", END)


# 10. Compile graph
pipeline_agent = builder.compile()

# 11. Execute application
print("===================================")
print(" LANGGRAPH PIPELINE AI ASSISTANT")
print(" Type exit to stop")
print("===================================")

while True:

    pipeline_name = input("\nEnter pipeline name: ")

    if pipeline_name.lower().strip() == "exit":
        print("Goodbye!")
        break

    try:

        
        result = pipeline_agent.invoke({

            "pipeline_name": pipeline_name,

            "status": "",

            "error_message": "",

            "recommendation": "",

            "retry_count": 0,

            "retry_approved": True,

            "retry_result": "",

            "final_response": ""

        })


        print("\nFinal Result:")
        print(result["final_response"])

    except Exception as error:
        print("\nError:", error)