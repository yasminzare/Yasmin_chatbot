import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable
from langchain_openai import ChatOpenAI


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING"]= 'true'
#os.environ["OPENAI_API_KEY"] =
#os.environ["LANGSMITH_API_KEY"] =
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"



llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

class State(TypedDict):
    messages: Annotated[list, add_messages]



@traceable()
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

d = graph.get_graph().draw_mermaid()


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.


@traceable
def format_prompt(subject):
  return [
      {
          "role": "system",
          "content": "You are a helpful assistant.",
      },
      {
          "role": "user",
          "content": f"{subject}?"
      }
  ]




@traceable
def parse_output(response):
  return response.content

@traceable
def run_pipeline():
  message1 = input("How can I assist you? ")
  messages = format_prompt(message1)
  response = llm.invoke(messages)
  return parse_output(response)

print(run_pipeline())