
import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import add_messages

from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------
# 1. Load Environment Variables
# ---------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")


# ---------------------------------
# 2. Initialize OpenAI LLM
# ---------------------------------

llm = ChatOpenAI(
    model="gpt-5.5",
    temperature=0
)


# ---------------------------------
# 3. Define Agent State
# ---------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------
# 4. Define Agent Node
# ---------------------------------

def agent_node(state: AgentState):

    system_prompt = SystemMessage(
        content="""
        You are a helpful conversational AI assistant.

        Remember the information shared by the user
        in the current conversation.

        Answer questions using the conversation history.
        Be friendly and professional.
        """
    )

    response = llm.invoke(
        [system_prompt] + state["messages"]
    )

    return {
        "messages": [response]
    }


# ---------------------------------
# 5. Build LangGraph Workflow
# ---------------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)

builder.add_edge(START, "agent")

builder.add_edge("agent", END)


# ---------------------------------
# 6. Configure SQLite Checkpointer
# ---------------------------------

DB_PATH = "day10_memory.sqlite"


# ---------------------------------
# 7. Compile and Run Agent
# ---------------------------------

def main():

    print("\nWelcome to Day 10 Conversational AI Agent!")

    thread_id = input(
        "Enter conversation ID (e.g. amol-001): "
    ).strip()

    if not thread_id:
        print("Conversation ID cannot be empty.")
        return

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:

        app = builder.compile(
            checkpointer=checkpointer
        )

        print(f"\nConversation ID: {thread_id}")
        print("Type 'exit' to stop.\n")

        while True:

            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            if not user_input:
                continue

            result = app.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config
            )

            response = result["messages"][-1]

            print("\nAI Agent:", response.content)
            print()


# ---------------------------------
# 8. Main Execution
# ---------------------------------

if __name__ == "__main__":
    main()
