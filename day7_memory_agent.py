
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage


# =====================================================
# 1. PIPELINE CONFIGURATION
# =====================================================

PIPELINES = [
    "customer_etl",
    "sales_etl",
    "netflix_etl"
]


# =====================================================
# 2. IDENTIFY PIPELINE FROM CONVERSATION
# =====================================================

def find_pipeline(messages) -> Optional[str]:

    for message in reversed(messages):

        if isinstance(message, HumanMessage):

            content = str(message.content).lower()

            for pipeline in PIPELINES:

                if pipeline in content:
                    return pipeline

    return None


# =====================================================
# 3. ANALYZE PIPELINE ERROR
# =====================================================

def analyze_error(message: str):

    message = message.lower()

    if "timeout" in message:

        return (
            "Database connection timeout",
            "Check database connectivity and credentials."
        )

    elif "memory" in message:

        return (
            "Out of memory error",
            "Review Spark executor memory and partitions."
        )

    elif "permission" in message:

        return (
            "Permission denied",
            "Check Unity Catalog permissions and storage access."
        )

    else:

        return (
            "Unknown pipeline error",
            "Check pipeline logs for more details."
        )


# =====================================================
# 4. MAIN ASSISTANT NODE
# =====================================================

def pipeline_assistant(state, config, *, store):

    messages = state["messages"]

    latest_message = str(messages[-1].content)
    user_input = latest_message.lower()

    # User-specific namespace
    user_id = config["configurable"]["user_id"]

    namespace = ("pipeline_memories", user_id)

    # Identify pipeline from conversation history
    pipeline_name = find_pipeline(messages)

    response = ""

    # -------------------------------------------------
    # CASE 1: USER REPORTS PIPELINE FAILURE
    # -------------------------------------------------

    if "failed" in user_input and pipeline_name:

        error, recommendation = analyze_error(user_input)

        # Save long-term memory
        store.put(
            namespace,
            pipeline_name,
            {
                "pipeline_name": pipeline_name,
                "last_error": error,
                "recommendation": recommendation
            }
        )

        response = (
            f"Your {pipeline_name} pipeline failed.\n"
            f"Error: {error}\n"
            f"Recommendation: {recommendation}\n"
            "Information saved in long-term memory."
        )

    # -------------------------------------------------
    # CASE 2: ASK ABOUT PREVIOUS ERROR
    # -------------------------------------------------

    elif (
        "previous error" in user_input
        or "what was my error" in user_input
        or "what happened earlier" in user_input
    ):

        # Search previous user messages in same thread
        previous_messages = [
            str(msg.content)
            for msg in messages[:-1]
            if isinstance(msg, HumanMessage)
        ]

        previous_error_message = next(
            (
                msg for msg in reversed(previous_messages)
                if "failed" in msg.lower()
                or "timeout" in msg.lower()
                or "memory" in msg.lower()
                or "permission" in msg.lower()
            ),
            None
        )

        if previous_error_message:

            error, recommendation = analyze_error(
                previous_error_message
            )

            response = (
                "From our previous conversation:\n"
                f"Error: {error}\n"
                f"Recommendation: {recommendation}"
            )

        else:

            # Fallback to long-term memory
            saved = (
                store.get(namespace, pipeline_name)
                if pipeline_name else None
            )

            if saved:

                response = (
                    f"Saved error: {saved.value['last_error']}\n"
                    f"Recommendation: {saved.value['recommendation']}"
                )

            else:

                response = (
                    "I could not find a previous pipeline error."
                )

    # -------------------------------------------------
    # CASE 3: RETRIEVE LONG-TERM MEMORY
    # -------------------------------------------------

    elif (
        "remember" in user_input
        or "what do you know" in user_input
        or "retrieve memory" in user_input
    ):

        if pipeline_name:

            saved = store.get(namespace, pipeline_name)

            if saved:

                response = (
                    f"Long-term memory for {pipeline_name}:\n"
                    f"Last error: {saved.value['last_error']}\n"
                    f"Recommendation: {saved.value['recommendation']}"
                )

            else:

                response = (
                    f"No saved memory found for {pipeline_name}."
                )

        else:

            memories = store.search(namespace)

            if memories:

                response = "Saved pipeline memories:\n"

                for item in memories:

                    response += (
                        f"\n{item.key}: {item.value}"
                    )

            else:

                response = "No pipeline memories are saved."

    # -------------------------------------------------
    # CASE 4: DEFAULT RESPONSE
    # -------------------------------------------------

    else:

        response = (
            "I am your Data Engineering Assistant.\n"
            "Available pipelines: customer_etl, "
            "sales_etl and netflix_etl.\n"
            "You can report failures or ask about previous errors."
        )

    return {
        "messages": [
            AIMessage(content=response)
        ]
    }


# =====================================================
# 5. BUILD LANGGRAPH
# =====================================================

builder = StateGraph(MessagesState)

builder.add_node(
    "pipeline_assistant",
    pipeline_assistant
)

builder.add_edge(START, "pipeline_assistant")

builder.add_edge(
    "pipeline_assistant",
    END
)


# =====================================================
# 6. CREATE MEMORY COMPONENTS
# =====================================================

# Short-term memory
short_term_memory = InMemorySaver()

# Long-term memory
long_term_memory = InMemoryStore()


# =====================================================
# 7. COMPILE LANGGRAPH
# =====================================================

app = builder.compile(
    checkpointer=short_term_memory,
    store=long_term_memory
)


# =====================================================
# 8. CHAT FUNCTION
# =====================================================

def chat(
    message: str,
    thread_id: str,
    user_id: str
):

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=message)
            ]
        },
        config=config
    )

    print("\nUser:", message)

    print(
        "\nAI:",
        result["messages"][-1].content
    )


# =====================================================
# 9. MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    user = "user_001"

    # -----------------------------------------------
    # TEST 1: SHORT-TERM MEMORY
    # -----------------------------------------------

    print("\n========== THREAD 1 ==========")

    chat(
        "My customer_etl pipeline failed with database timeout",
        "conversation_001",
        user
    )

    chat(
        "What was my previous error?",
        "conversation_001",
        user
    )

    # -----------------------------------------------
    # TEST 2: LONG-TERM MEMORY
    # -----------------------------------------------

    print("\n========== THREAD 2 ==========")

    chat(
        "What do you remember about customer_etl?",
        "conversation_002",
        user
    )

    # -----------------------------------------------
    # TEST 3: SAVE SALES PIPELINE MEMORY
    # -----------------------------------------------

    print("\n========== THREAD 3 ==========")

    chat(
        "My sales_etl pipeline failed with out of memory error",
        "conversation_003",
        user
    )

    # -----------------------------------------------
    # TEST 4: RETRIEVE SALES PIPELINE MEMORY
    # -----------------------------------------------

    print("\n========== THREAD 4 ==========")

    chat(
        "What do you remember about sales_etl?",
        "conversation_004",
        user
    )

    # -----------------------------------------------
    # TEST 5: USER MEMORY ISOLATION
    # -----------------------------------------------

    print("\n========== THREAD 5 ==========")

    chat(
        "What do you remember about customer_etl?",
        "conversation_005",
        "user_002"
    )
