import os
import json

from dotenv import load_dotenv
from openai import OpenAI

# 1. Load API credentials
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = "gpt-5.5"

# 2. Define Python tools

def calculate_cost(records, cost_per_record):
    return records * cost_per_record


def calculate_partitions(data_size_gb, partition_size_mb):
    if data_size_gb <= 0 or partition_size_mb <= 0:
        raise ValueError("Sizes must be positive")

    total_mb = data_size_gb * 1024
    return (total_mb + partition_size_mb - 1) // partition_size_mb


def convert_storage(gb):
    return gb / 1024

def calculate_etl_runtime(number_of_records, processing_speed):

    if number_of_records < 0:
        raise ValueError("Records cannot be negative")

    if processing_speed <= 0:
        raise ValueError("Processing speed must be positive")

    time_seconds = number_of_records / processing_speed
    time_minutes = time_seconds / 60

    return {
        "seconds": round(time_seconds, 2),
        "minutes": round(time_minutes, 2)
    }

# 3. Define tool schemas

tools = [
    {
        "type": "function",
        "name": "calculate_cost",
        "description": "Calculate ETL cost in INR.",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {"type": "integer"},
                "cost_per_record": {"type": "number"}
            },
            "required": ["records", "cost_per_record"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "calculate_partitions",
        "description": "Estimate Spark data partitions.",
        "parameters": {
            "type": "object",
            "properties": {
                "data_size_gb": {"type": "number"},
                "partition_size_mb": {"type": "integer"}
            },
            "required": ["data_size_gb", "partition_size_mb"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "convert_storage",
        "description": "Convert GB into TB.",
        "parameters": {
            "type": "object",
            "properties": {
                "gb": {"type": "number"}
            },
            "required": ["gb"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "calculate_etl_runtime",
        "description": "Calculate ETL runtime.",
        "parameters": {
            "type": "object",
            "properties": {
                "number_of_records": {"type": "integer"},
                "processing_speed": {"type": "number"}
            },
            "required": ["number_of_records", "processing_speed"],
            "additionalProperties": False
        },
        "strict": True
    }
]

# 4. Map tool names to actual Python functions

available_functions = {
    "calculate_cost": calculate_cost,
    "calculate_partitions": calculate_partitions,
    "convert_storage": convert_storage,
    "calculate_etl_runtime": calculate_etl_runtime
    
}


# 5. Execute requested tools safely

def execute_tool(tool_name, arguments):

    if tool_name not in available_functions:
        raise ValueError(f"Unknown tool: {tool_name}")

    function = available_functions[tool_name]

    result = function(**arguments)

    return result


# 6. Build the AI agent

def ask_agent(user_question):

    instructions = """
    You are an expert Data Engineering AI Assistant.

    You have access to Python tools.

    Use the appropriate tool when calculations are required.
    Explain the tool result clearly.
    Do not invent calculation results.
    """

    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=user_question,
        tools=tools
    )

    # 7. Handle tool calls
    while True:

        function_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not function_calls:
            return response.output_text

        tool_outputs = []

        for call in function_calls:

            print(f"\nTool selected: {call.name}")

            arguments = json.loads(call.arguments)

            try:
                result = execute_tool(
                    call.name,
                    arguments
                )

                print("Tool result:", result)

                output = {
                    "success": True,
                    "result": result
                }

            except Exception as error:
                output = {
                    "success": False,
                    "error": str(error)
                }

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(output)
            })

        # 8. Send tool results back to LLM

        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools
        )


# 9. Run application

print("====================================")
print(" DATA ENGINEERING AI AGENT")
print(" Type exit to stop")
print("====================================")

while True:

    question = input("\nYou: ")

    if question.lower().strip() == "exit":
        print("Goodbye!")
        break

    try:
        answer = ask_agent(question)
        print("\nAI Agent:", answer)

    except Exception as error:
        print("\nApplication error:", error)
