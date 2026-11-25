
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load API key
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing in .env")

# 2. Initialize LLM
model = ChatOpenAI(
    model="gpt-5.5"
)

# 3. Define reusable prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a Senior Data Engineering Assistant.

        Your expertise includes:
        Python, SQL, PySpark, Databricks,
        Azure Data Factory, Dataiku and BigQuery.

        Explain concepts with practical examples.
        """
    ),
    (
        "human",
        "Explain {question}"
    )
])

# 4. Create output parser
output_parser = StrOutputParser()

# 5. Build LangChain pipeline
chain = prompt | model | output_parser

# 6. Run chatbot
print("===================================")
print(" LANGCHAIN DATA ENGINEERING BOT")
print(" Type exit to stop")
print("===================================")

while True:

    question = input("\nYou: ")

    if question.lower().strip() == "exit":
        print("Goodbye!")
        break

    try:

        response = chain.invoke({
            "question": question
        })

        print("\nAI:", response)

    except Exception as error:
        print("\nError:", error)
