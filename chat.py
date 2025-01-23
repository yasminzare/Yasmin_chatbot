import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith.tracing import traceable
from tavily import TavilyClient
from typing_extensions import TypedDict

load_dotenv()


os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

required_env_vars = ["OPENAI_API_KEY", "LANGSMITH_API_KEY", "TAVILY_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


class State(TypedDict):
    messages: Annotated[list, "List of message dictionaries for LLM"]


@traceable
def format_prompt(subject: str):
    if not subject.strip():
        raise ValueError("Subject cannot be empty.")
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{subject}?"},
    ]


@traceable
def tavilliy_search(subject: str):
    if not subject.strip():
        raise ValueError("Subject cannot be empty for Tavily search.")
    try:
        content = client.search(subject, search_depth="advanced")["results"]
        return content  # You may need to access specific fields in the response object
    except Exception:
        return "No sources found"


@traceable()
def chatbot(state: State):
    try:
        if "messages" not in state or not isinstance(state["messages"], list):
            raise ValueError("State must include a valid 'messages' list.")
        return {"messages": [llm.invoke(state["messages"])]}
    except Exception as e:
        raise RuntimeError(f"Chatbot failed: {e}")


@traceable
def parse_output(response):
    try:
        tavily_response = tavilliy_search(response.content)
        print(f"Response: {response.content} \n Sources: {tavily_response}")
    except AttributeError:
        raise ValueError("Response object is missing a 'content' attribute.")
    except Exception as e:
        raise RuntimeError(f"Error in parsing output: {e}")


@traceable
def run_pipeline():
    while True:
        try:
            # Get user input
            message1 = input("How can I assist you? ").strip()
            if not message1:
                raise ValueError("Input cannot be empty.")

            # Format the input into messages
            messages = format_prompt(message1)

            # Get the OpenAI LLM response
            response = llm.invoke(messages)

            # Parse and combine outputs from multiple services
            print(parse_output(response))
            continue_prompt = (
                input("Do you have any other questions? (y/n): ").strip().lower()
            )
            if continue_prompt == "n":
                print("Goodbye!")
                quit()

        except Exception as e:
            return f"Pipeline failed: {e}"


if __name__ == "__main__":
    print(run_pipeline())
