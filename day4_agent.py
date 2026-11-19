import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

#load environment variables
load_dotenv() 

#Tool 1 : ETL runtime

@tool
def calculate_etl_runtime(number_of_records: int, processing_speed: float) -> dict:
    """
    Calculate ETL runtime in seconds and minutes.
    """
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

#Tool 2 : ETL cost
@tool
def calculate_etl_cost(records: int, cost_per_record: float) -> str:
    """
    Calculate ETL cost in INR.
    """
    if records < 0 or cost_per_record < 0:
        raise ValueError("Invalid input: records and cost_per_record must be non-negative.")

    return f"INR {records * cost_per_record:.2f}"


#Tool 3 : Spark partitions
@tool
def calculate_spark_partitions(data_size_gb: float, partition_size_mb: float) -> str:
    """
    Calculate the number of Spark partitions.
    """
    if data_size_gb <= 0 or partition_size_mb <= 0:
        raise ValueError("Sizes must be positive")

    total_mb = data_size_gb * 1024
    partitions = (total_mb + partition_size_mb - 1) // partition_size_mb
    return f"Spark partitions: {int(partitions)}"


#Tool 4 : Storage conversion
@tool
def convert_gb_to_tb(gb: float) -> str:
    """
    Convert GB to TB.
    """
    if gb < 0:
        raise ValueError("Size must be non-negative")

    tb = gb / 1024
    return f"{gb} GB is equal to {tb:.2f} TB"



from langchain.tools import tool

#Pipeline status

# Mock pipeline database

pipelines = {
    "netflix_ingestion": "SUCCESS",
    "customer_etl": "FAILED",
    "salecheck s_pipeline": "RUNNING"
}


@tool
def check_pipeline_status(pipeline_name: str) -> str:
    """
    Check the execution status of a Data Engineering pipeline.

    pipeline_name: Name of the pipeline to check.
    """

    # Normalize pipeline name
    pipeline_name = pipeline_name.strip().lower()

    # Check whether pipeline exists
    if pipeline_name in pipelines:

        status = pipelines[pipeline_name]

        return (
            f"Pipeline: {pipeline_name}\n"
            f"Status: {status}"
        )

    else:
        return (
            f"Pipeline '{pipeline_name}' "
            f"does not exist in the pipeline database."
        )



#Register tools
tools = [calculate_etl_runtime, calculate_etl_cost, calculate_spark_partitions, convert_gb_to_tb, check_pipeline_status ]

#Initialize model
model=ChatOpenAI(model="gpt-5.5", api_key=os.getenv("OPENAI_API_KEY"))


#Create agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
    You are an expert Data Engineering Assistant.

    Help users with ETL, Spark, Databricks,
    Python and SQL questions.

    Use available tools for calculations.
    Do not invent calculation results.
    Explain results in simple language.
    """
)


#Interactive chatbot
print("===================================")
print(" Welcome to Data Engineering AI Agent")
print("===================================")

while True:
    question = input("\nYou: ")

    if question.lower().strip() == "exit":
        print("AI: Goodbye! Happy learning.")
        break

    try:
        result = agent.invoke({
            "messages": [
                {"role": "user", 
                 "content": question
                }
            ]
        })
        final_answer = result["messages"][-1].content
        print("\nAI:", final_answer)
    
    except Exception as error:
        print("\nApplication error:", error)
