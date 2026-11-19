import os
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load environment variables
load_dotenv()

# Step 2: Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Step 3: Define the AI assistant instructions
SYSTEM_PROMPT = """
You are an expert Data Engineering Assistant.

Your expertise includes:
- Python
- SQL
- Azure Data Factory
- Azure Databricks
- PySpark
- Dataiku
- Data Engineering

Explain concepts with practical examples.
Keep answers clear and beginner-friendly.
"""

# Step 4: Create chatbot conversation
conversation = []

print("===================================")
print(" Welcome to AI Data Engineering Bot")
print(" Type 'exit' to stop")
print("===================================")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("AI: Goodbye! Happy learning.")
        break

    # Add user message
    conversation.append({
        "role": "user",
        "content": user_input
    })

    try:
        # Step 5: Send request to LLM
        response = client.responses.create(
            model="gpt-5.5",
            instructions=SYSTEM_PROMPT,
            input=conversation
        )

        # Step 6: Extract AI response
        ai_response = response.output_text

        print("\nAI:", ai_response)

        # Store assistant response
        conversation.append({
            "role": "assistant",
            "content": ai_response
        })

    except Exception as error:
        print("Error:", error)
