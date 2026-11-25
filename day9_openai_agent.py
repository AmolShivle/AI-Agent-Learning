import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


#Load environment variables from .env file
load_dotenv()

#Initialize the OpenAI client with the API key from environment variables
llm = ChatOpenAI(model="gpt-5.5", temperature=0)

#Define Calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""
    allowed_chars = "0123456789+-*/(). "

    if not all(char in allowed_chars for char in expression):
        raise ValueError("Invalid characters in expression")
    
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        raise ValueError(f"Calculation: {e}")


#Define a custom greeting tool
@tool
def custom_greeting(name: str) -> str:
    """Generate a personalized greeting message."""
    return f"Hello, {name}! Welcome to the AI Agent."


#Employee Details tool
@tool
def employee_details(employee_id: str) -> str: 
    """Fetch employee details based on the provided employee ID."""
    # For demonstration purposes, we'll use a mock database
    mock_database = {
        "E001": {"name": "Alice", "position": "Software Engineer", "department": "IT"},
        "E002": {"name": "Bob", "position": "Data Scientist", "department": "Analytics"},
        "E003": {"name": "Charlie", "position": "Product Manager", "department": "Product"},
    }

    details = mock_database.get(employee_id)
    if details:
        return f"Employee ID: {employee_id}, Name: {details['name']}, Position: {details['position']}, Department: {details['department']}"
    else:
        return f"No details found for Employee ID: {employee_id}"


#Register the tools with LLM
tools =[calculator, custom_greeting, employee_details]

llm_with_tools = llm.bind_tools(tools)

# Define Graph State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


#Agent node
def agent_node(state: AgentState) :
    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages" : [response]
    }

#Build LangGraph workflow
builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

#Compile graph
app = builder.compile()

#Run Agent
def run_agent(user_input: str):

    result=app.invoke({
        "messages":[SystemMessage(
            content="You are a helpful AI assistant. "
                        "Use available tools when necessary."),
                        HumanMessage(content=user_input)]
    })

    return result["messages"][-1].content


#Main Execution
if __name__ == "__main__":
    print("Welcome to Day 9 AI Agent!")
    print("Type 'exit' to stop.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower().strip() == "exit":
            print("Goodbye!")
            break

        try:
            response = run_agent(user_input)
            print("\nAI Agent:", response)
        except Exception as error:
            print("\nApplication error:", error)


