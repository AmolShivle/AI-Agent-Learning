
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage


# 1. Define state
class ChatState(MessagesState):
    pipeline_name: str


# 2. AI assistant node
def pipeline_assistant(state: ChatState):

    messages = state["messages"]

    # Read the latest user message
    user_message = messages[-1].content.lower()

    # Simple rule-based response for learning
    if "failed" in user_message:
        response = (
            f"Your {state['pipeline_name']} pipeline failed. "
            "Can you share the error message?"
        )

    elif "timeout" in user_message:
        response = (
            f"The error for {state['pipeline_name']} "
            "was a database connection timeout. "
            "Check database connectivity and credentials."
        )

    elif "previous error" in user_message or "what was my error" in user_message:
        previous_messages = [
            msg.content
            for msg in messages[:-1]
            if isinstance(msg, HumanMessage)
        ]

        response = (
            "Your previous conversation: "
            + " | ".join(previous_messages)
        )

    else:
        response = (
            f"I am monitoring your {state['pipeline_name']} pipeline. "
            "How can I help?"
        )

    return {
        "messages": [AIMessage(content=response)]
    }


# 3. Build graph
builder = StateGraph(ChatState)

builder.add_node("pipeline_assistant", pipeline_assistant)

builder.add_edge(START, "pipeline_assistant")
builder.add_edge("pipeline_assistant", END)


# 4. Add short-term memory
memory = InMemorySaver()

app = builder.compile(checkpointer=memory)


# 5. Chat function
def chat(user_input: str, thread_id: str):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "pipeline_name": "customer_etl"
        },
        config=config
    )

    print("\nAI:", result["messages"][-1].content)


# 6. Run conversation
if __name__ == "__main__":

    thread = "customer-session-001"

    chat(
        "My customer_etl pipeline failed",
        thread
    )

    chat(
        "The error was database connection timeout",
        thread
    )

    chat(
        "What was my previous error?",
        thread
    )
