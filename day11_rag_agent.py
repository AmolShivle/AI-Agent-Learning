
import os
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.tools import tool
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# -----------------------------------------
# 1. Load Environment Variables
# -----------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY missing in .env")


# -----------------------------------------
# 2. Initialize OpenAI Models
# -----------------------------------------

llm = ChatOpenAI(
    model="gpt-5.5"
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -----------------------------------------
# 3. Load Knowledge Base Documents
# -----------------------------------------

KNOWLEDGE_DIR = Path("knowledge_base")

documents = []

for file_path in KNOWLEDGE_DIR.glob("*.txt"):

    text = file_path.read_text(encoding="utf-8")

    documents.append(
        Document(
            page_content=text,
            metadata={"source": file_path.name}
        )
    )

if not documents:
    raise ValueError("No .txt files found in knowledge_base!")


print(f"Loaded {len(documents)} document(s)")


# -----------------------------------------
# 4. Split Documents into Chunks
# -----------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# -----------------------------------------
# 5. Create Chroma Vector Database
# -----------------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="data_engineering_knowledge",
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

print("Chroma vector database ready!")


# -----------------------------------------
# 6. Create RAG Retrieval Tool
# -----------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal data engineering knowledge base.
    Use this tool for questions about stored documentation.
    """

    results = retriever.invoke(query)

    if not results:
        return "No relevant information found."

    formatted_results = []

    for index, doc in enumerate(results, start=1):

        source = doc.metadata.get("source", "Unknown")

        formatted_results.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Content: {doc.page_content}"
        )

    return "\n\n".join(formatted_results)


# -----------------------------------------
# 7. Register Tools with LLM
# -----------------------------------------

tools = [search_knowledge_base]

llm_with_tools = llm.bind_tools(tools)


# -----------------------------------------
# 8. Define LangGraph State
# -----------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------------------
# 9. Define Agent Node
# -----------------------------------------

def agent_node(state: AgentState):

    system_prompt = SystemMessage(
        content="""
        You are a professional Data Engineering AI Assistant.

        Answer questions about the internal knowledge base
        using the search_knowledge_base tool.

        Always use retrieved documentation for questions
        about the knowledge base.

        Do not invent facts or pretend that information
        exists in the documents when it does not.

        If the retrieved documents do not contain the answer,
        clearly tell the user that the information is not
        available in the knowledge base.

        Explain answers clearly with examples when useful.
        """
    )

    response = llm_with_tools.invoke(
        [system_prompt] + state["messages"]
    )

    return {
        "messages": [response]
    }


# -----------------------------------------
# 10. Build LangGraph Workflow
# -----------------------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)

builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition
)

builder.add_edge("tools", "agent")


# -----------------------------------------
# 11. Compile Graph
# -----------------------------------------

app = builder.compile()


# -----------------------------------------
# 12. Run RAG Agent
# -----------------------------------------

def run_agent(user_input: str):

    result = app.invoke({
        "messages": [
            HumanMessage(content=user_input)
        ]
    })

    return result["messages"][-1].content


# -----------------------------------------
# 13. Main Program
# -----------------------------------------

if __name__ == "__main__":

    print("\nWelcome to Day 11 RAG AI Agent!")
    print("Ask questions about Data Engineering.")
    print("Type 'exit' to stop.\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = run_agent(user_input)

        print("\nAI Agent:", response)
        print()
