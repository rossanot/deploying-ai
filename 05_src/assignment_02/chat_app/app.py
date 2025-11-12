import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import Optional
import os
from utils import *

from langchain.chat_models import init_chat_model

load_dotenv('../.secrets')

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY environment variable")


# QUERY_STUDENT_INFO = "Extract information on the student that performs the best on basic chemical topics"
# QUERY_STUDENT_INFO += "I require you suggest a meeting within the email body. The potential meeting date or dates should considering today's date"
# QUERY_STUDENT_INFO += "as well as the upcoming holidays within the current month or next month (if required). Mention the holidays in the email."

# Invoke the call
# messages = [HumanMessage(content=generate_prompt(QUERY_STUDENT_INFO, top_n=1, my_collection=collection, my_client_obj=embeddings_client))]
# messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()

PROJECT_PATH = '../'
PROJECT_DATA_PATH = '../data'

# filepaths
JSON_PATH = os.path.join(PROJECT_DATA_PATH, 'students-performance.jsonl')
CHROMA_DB_DIR = os.path.join(PROJECT_DATA_PATH, 'chroma_db')

# Embeddings
COLLECTION_NAME = "chem_class"
EMBEDDINGS_MODEL_NAME = "text-embedding-3-small"

embeddings_client = OpenAI()
chroma_client = chromadb.Client()


ids, documents = define_embeddings_items(JSON_PATH)
embeddings = get_embeddings(documents, embeddings_client)
collection = get_collection(embeddings, documents, ids, chroma_client, COLLECTION_NAME)
llm = get_assistant_agent()


def simple_chat(message: str, history: list[dict]) -> str:
    langchain_messages = []
    n = 0
    for msg in history:
        if msg['role'] == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))
            n += 1
    
    langchain_messages.append(
        HumanMessage(
            content=generate_prompt(message, top_n=1, my_collection=collection, my_client_obj=embeddings_client)
            )
            )
    
    state = {
        "messages": langchain_messages,
        "llm_calls": n
    }

    response = llm.invoke(state)

    return response['messages'][len(response['messages']) - 1].content

    
chat = gr.ChatInterface(
    fn=simple_chat,
    type="messages"
)

if __name__ == "__main__":
    chat.launch()
