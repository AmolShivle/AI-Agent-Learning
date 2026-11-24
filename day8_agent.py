
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END


# --------------------------------
# 1. Define Agent State
# --------------------------------

class AgentState(TypedDict):
    user_query: str
    pipeline_name: Optional[str]
    pipeline_status: Optional[str]
    error_analysis: Optional[str]
    pipeline_runtime: Optional[str]
    final_response: Optional[str]


# --------------------------------
# 2. Tool 1: Check Pipeline Status
# --------------------------------

def check_pipeline_status(pipeline_name: str) -> str:

    pipeline_data = {
        "customer_etl": "FAILED",
        "sales_etl": "SUCCESS",
        "netflix_etl": "RUNNING"
    }

    return pipeline_data.get(
        pipeline_name,
        "UNKNOWN"
    )


# --------------------------------
# 3. Tool 2: Analyze Pipeline Error
# --------------------------------

def analyze_pipeline_error(error_message: str) -> str:

    error = error_message.lower()

    if "timeout" in error:
        return (
            "Root cause: Database connection timeout. "
            "Recommendation: Check database availability, "
            "connection timeout settings and query performance."
        )

    elif "memory" in error:
        return (
            "Root cause: Memory exhaustion. "
            "Recommendation: Review Spark executor memory, "
            "partition sizes and data skew."
        )

    elif "permission" in error:
        return (
            "Root cause: Permission issue. "
            "Recommendation: Validate service principal, "
            "Unity Catalog grants and storage permissions."
        )

    return (
        "Root cause is unclear. "
        "Check pipeline logs for additional details."
    )


 # Tool 3: pipeline runtime

def check_pipeline_runtime(pipeline_name: str) -> str:

    runtime_data = {
        "customer_etl": "2 hours",
        "sales_etl": "30 minutes",
        "netflix_etl": "1 hour"
    }

    return runtime_data.get(
        pipeline_name,
        "UNKNOWN"
    )



# --------------------------------
# 4. Agent Node: Understand Query
# --------------------------------

def agent_node(state: AgentState):

    query = state["user_query"].lower()

    pipeline_name = None

    for name in [
        "customer_etl",
        "sales_etl",
        "netflix_etl"
    ]:
        if name in query:
            pipeline_name = name
            break

    return {
        "pipeline_name": pipeline_name
    }


# --------------------------------
# 5. Tool Execution Node
# --------------------------------

def tool_node(state: AgentState):

    pipeline_name = state["pipeline_name"]
    query = state["user_query"]

    status = check_pipeline_status(pipeline_name)

    analysis = analyze_pipeline_error(query)

    pipeline_runtime = check_pipeline_runtime(pipeline_name)

    return {
        "pipeline_status": status,
        "error_analysis": analysis,
        "pipeline_runtime": pipeline_runtime
    }


# --------------------------------
# 6. Final Response Node
# --------------------------------

def response_node(state: AgentState):

    pipeline = state["pipeline_name"]
    status = state["pipeline_status"]
    analysis = state["error_analysis"]
    pipeline_runtime = state["pipeline_runtime"]


    if status == "FAILED":
        response = (
            f"Pipeline: {pipeline}\n"
            f"Status: {status}\n"
            f"Analysis: {analysis}\n"
            f"Runtime: {pipeline_runtime}\n"
            "Next action: Investigate the root cause before retrying."
        )

    elif status == "SUCCESS":
        response = (
            f"Pipeline: {pipeline}\n"
            f"Status: {status}\n"
            f"Runtime: {pipeline_runtime}\n"
            "No failure investigation is required."
        )

    elif status == "RUNNING":
        response = (
            f"Pipeline: {pipeline}\n"
            f"Status: {status}\n"
            f"Runtime: {pipeline_runtime}\n"
            "Monitor the pipeline until execution completes."
        )

    else:
        response = "Pipeline status could not be determined."

    return {"final_response": response}


# --------------------------------
# 7. Conditional Routing
# --------------------------------

def route_after_agent(state: AgentState):

    if state["pipeline_name"]:
        return "tools"

    return "response"


# --------------------------------
# 8. Build LangGraph
# --------------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("response", response_node)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "response": "response"
    }
)

builder.add_edge("tools", "response")
builder.add_edge("response", END)

app = builder.compile()


# --------------------------------
# 9. Run Agent
# --------------------------------

if __name__ == "__main__":

    # query = (
    #     "My customer_etl pipeline failed "
    #     "due to database timeout. Please analyze it."
    # )

    query = (input("You: ") or query).strip()

    result = app.invoke({
        "user_query": query,
        "pipeline_name": None,
        "pipeline_status": None,
        "error_analysis": None,
        "pipeline_runtime": None,
        "final_response": None
    })

    print("\n===== AI AGENT RESPONSE =====")
    print(result["final_response"])
