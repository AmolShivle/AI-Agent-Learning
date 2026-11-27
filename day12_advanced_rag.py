import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

from langchain_core.documents import Document

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)

from langchain_chroma import Chroma

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import InMemorySaver


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing in .env")

BASE_DIR = Path(__file__).parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "advanced_rag_knowledge"

llm = ChatOpenAI(
    model="gpt-5.5",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# ==========================================
# 2. LOAD AND INDEX DOCUMENTS
# ==========================================

def create_vector_store():

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # Build the index only when it is empty.
    if vector_store._collection.count() == 0:

        print("Creating knowledge base index...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        all_documents = []

        for file_path in KNOWLEDGE_DIR.glob("*.txt"):

            category = file_path.stem

            text = file_path.read_text(
                encoding="utf-8"
            )

            chunks = splitter.split_text(text)

            for chunk in chunks:

                all_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path.name,
                            "category": category
                        }
                    )
                )

        if not all_documents:
            raise ValueError(
                "No TXT documents found in knowledge_base."
            )

        vector_store.add_documents(all_documents)

        print(
            f"Indexed {len(all_documents)} chunks."
        )

    else:

        print(
            "Existing index loaded:",
            vector_store._collection.count(),
            "chunks"
        )

    return vector_store


vector_store = create_vector_store()


# ==========================================
# 3. DEFINE LANGGRAPH STATE
# ==========================================

class RAGState(MessagesState):

    category: str
    standalone_question: str
    context: str
    sources: list[str]
    answer: str


# ==========================================
# 4. QUERY REWRITING NODE
# ==========================================

def rewrite_query(state: RAGState):

    messages = state["messages"]

    current_question = messages[-1].content

    previous_messages = messages[:-1]

    if not previous_messages:

        return {
            "standalone_question": current_question
        }

    history = []

    for message in previous_messages[-6:]:

        role = (
            "User"
            if isinstance(message, HumanMessage)
            else "Assistant"
        )

        history.append(
            f"{role}: {message.content}"
        )

    history_text = "\n".join(history)

    prompt = f"""
You are a query rewriting assistant.

Conversation history:
{history_text}

Latest user question:
{current_question}

Rewrite the latest question as a standalone question.

Preserve the original intent.
Do not answer the question.
Return only the rewritten question.
"""

    response = llm.invoke(prompt)

    rewritten_question = (
        response.content.strip()
        if isinstance(response.content, str)
        else current_question
    )

    print("\nRewritten question:")
    print(rewritten_question)

    return {
        "standalone_question": rewritten_question
    }


# ==========================================
# 5. RETRIEVAL NODE WITH METADATA FILTER
# ==========================================

def retrieve_documents(state: RAGState):

    question = state["standalone_question"]

    category = state.get("category", "all")

    search_filter = None

    if category != "all":

        search_filter = {
            "category": category
        }

    documents = vector_store.similarity_search(
        question,
        k=4,
        filter=search_filter
    )

    if not documents:

        return {
            "context": "",
            "sources": []
        }

    context_parts = []
    sources = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown"
        )

        sources.add(source)

        context_parts.append(
            f"Source: {source}\n"
            f"Content: {document.page_content}"
        )

    context = "\n\n".join(context_parts)

    print("\nRetrieved documents:")
    print(context)

    return {
        "context": context,
        "sources": sorted(sources)
    }


# ==========================================
# 6. GENERATE ANSWER NODE
# ==========================================

def generate_answer(state: RAGState):

    question = state["standalone_question"]

    context = state["context"]

    if not context:

        answer = (
            "I could not find relevant information "
            "in the selected knowledge base."
        )

    else:

        prompt = f"""
You are a Senior Data Engineering AI Assistant.

Answer the question using only the supplied context.

If the context does not contain enough information,
clearly say that the information is unavailable.

Do not invent technical facts.

Context:
{context}

Question:
{question}

Provide a clear and structured answer.
"""

        response = llm.invoke(prompt)

        answer = response.content

    print("\nFinal Answer:")
    print(answer)

    return {
        "answer": answer,
        "messages": [
            AIMessage(content=answer)
        ]
    }


# ==========================================
# 7. BUILD LANGGRAPH WORKFLOW
# ==========================================

builder = StateGraph(RAGState)

builder.add_node(
    "rewrite_query",
    rewrite_query
)

builder.add_node(
    "retrieve_documents",
    retrieve_documents
)

builder.add_node(
    "generate_answer",
    generate_answer
)

builder.add_edge(
    START,
    "rewrite_query"
)

builder.add_edge(
    "rewrite_query",
    "retrieve_documents"
)

builder.add_edge(
    "retrieve_documents",
    "generate_answer"
)

builder.add_edge(
    "generate_answer",
    END
)

app = builder.compile(
    checkpointer=InMemorySaver()
)


# ==========================================
# 8. CHAT FUNCTION
# ==========================================

def chat(
    question: str,
    thread_id: str,
    category: str = "all"
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ],
            "category": category
        },
        config=config
    )

    return result["answer"]


# ==========================================
# 9. RETRIEVAL EVALUATION
# ==========================================

def evaluate_retrieval(
    question: str,
    expected_sources: list[str],
    category: str = "all",
    k: int = 4
):

    search_filter = None

    if category != "all":

        search_filter = {
            "category": category
        }

    results = vector_store.similarity_search(
        question,
        k=k,
        filter=search_filter
    )

    retrieved_sources = {
        doc.metadata.get("source", "unknown")
        for doc in results
    }

    expected = set(expected_sources)

    relevant = retrieved_sources.intersection(
        expected
    )

    precision = (
        len(relevant) / k
        if k > 0
        else 0
    )

    recall = (
        len(relevant) / len(expected)
        if expected
        else 0
    )

    print("\n===== RETRIEVAL EVALUATION =====")

    print("Question:", question)

    print(
        "Retrieved sources:",
        sorted(retrieved_sources)
    )

    print("Expected sources:", sorted(expected))

    print(f"Precision@{k}: {precision:.2f}")

    print(f"Recall@{k}: {recall:.2f}")

    return {
        "precision": precision,
        "recall": recall,
        "retrieved_sources": sorted(retrieved_sources)
    }


# ==========================================
# 10. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("\n===== TEST 1: CONVERSATIONAL RAG =====")

    chat(
        "Explain the three layers of Snowflake architecture.",
        thread_id="session_003",
        category="snowflake"
    )

    # print("\n===== TEST 1: CONVERSATIONAL RAG =====")
    
    # chat(
    #         "What is Unity Catalog?",
    #         thread_id="session_001",
    #         category="databricks"
    #     )
    # chat(
    #     "What are its main features?",
    #     thread_id="session_001",
    #     category="databricks"
    # )

    # print("\n===== TEST 2: METADATA FILTERING =====")

    # chat(
    #     "Explain watermark-based incremental loading.",
    #     thread_id="session_002",
    #     category="azure_data_factory"
    # )

    # print("\n===== TEST 3: RETRIEVAL EVALUATION =====")

    # evaluate_retrieval(
    #     question="What is Unity Catalog?",
    #     expected_sources=["databricks.txt"],
    #     category="databricks",
    #     k=4
    # )
