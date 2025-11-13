import json
from pathlib import Path
import os
import re
import datetime
from langchain.tools import tool
import requests
import json

import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from langchain_community.document_loaders import JSONLoader
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.messages import (AnyMessage, SystemMessage)
from langgraph.graph import StateGraph, START, END

from openai import OpenAI

from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from IPython.display import Image, display


def get_metadata(record:dict, metadata: dict) -> dict:
    """Get metadata of records
    """
    # print(record)
    # record = record["kwargs"]["metadata"]
    metadata["student_id"] = record.get("student_id")
    metadata["student_name"] = record.get("student_name")
    metadata["parent_email"]= record.get("parent_email")
    metadata["teacher_name"]= record.get("teacher_name")

    
    if "score" in record: metadata["score"] = record["score"]
    if "grade" in record: metadata["grade"] = record["grade"]
    return metadata

def get_record_text(
        rec: dict
        ) -> str:
    """Turn each student record into a single text string for embedding
    """
    return f"{rec["kwargs"]["metadata"]["student_name"]}: {rec["kwargs"]["page_content"]}"

def get_embedding(
        text,
        my_client_obj,
        model="text-embedding-3-small"
        ):
    text = text.replace("\n", " ")
    return my_client_obj.embeddings.create(input=[text], model=model).data[0].embedding

def query_chromadb(
        query,
        my_collection,
        my_client_obj,
        top_n=2
        ):
    """Return query results
    """
    query_embedding = get_embedding(query, my_client_obj)
    results = my_collection.query(query_embeddings=[query_embedding], n_results=top_n)
    return results
    # return [(id, score, text) for id, score, text in zip(results['ids'][0], results['distances'][0], results['documents'][0])]


def get_name_and_comment(text):
    """Extract info from docs
    """
    range_match = re.search(r'^[A-Za-z]+\s[A-Za-z]+:\s', text).span()
    return (
        text[range_match[0]:range_match[1]-2],
        text[range_match[1]:]
        )


def get_context_data(
        query,
        my_collection,
        my_client_obj,
        top_n=2
        ):
    """Return query results
    """
    query_embedding = get_embedding(query, my_client_obj)
    results = my_collection.query(query_embeddings=[query_embedding], n_results=top_n)

    context_data = []
    for idx, result_item_id in enumerate(results['ids'][0]):
        details = {}
        details['student_id'] = result_item_id
        docs_info = get_name_and_comment(results['documents'][0][idx])
        details['student_name'] = docs_info[0]
        details['comments']  = docs_info[1]
        context_data.append(details)
    return context_data

def get_system_prompt():
    system_prompt  = "You are the personal assistant of a high school principal.\n"
    system_prompt += "The high-school in question stands out for the performance of its students in Chemistry across the entire country.\n\n"
    system_prompt += "You are very polite and an expert in communication as well as in conflict management.\n\n"
    system_prompt += "Your name and position are given below, include them in emails and reports.\n\n"
    system_prompt += "<assistant_info>\n"
    system_prompt += "Your name: Alicia Keys\n"
    system_prompt += "Your position: High School Program Administrator\nSt. Regis High Shool"
    system_prompt += "</assistant_info>\n"
    system_prompt += "Your capabilities include but are not limited to:"
    system_prompt += "(a) Summarizing of the performance of a sample of high-school chemistry students\n\n"
    system_prompt += "(b) Drafting emails.\n\n"
    system_prompt += "(c) Suggesting meeting dates.\n\n"
    system_prompt += "(d) Creating summary reports of students' performance.\n\n"
    system_prompt += "(e) Reporting today's date\n\n"
    system_prompt += "Refuse to do anything else unrelated to your capabilities, e.g., .\n\n"
    system_prompt += "REMEMBER:\n\n"
    system_prompt += "(1) You never use profanities or biased vocabulary and/or expressions.\n\n"
    system_prompt += "(2) You never disclose student's PII in emails\n\n"
    return system_prompt

def generate_prompt(
        query: str,
        top_n: int,
        my_collection: chromadb.api.models.Collection,
        my_client_obj: OpenAI
        ):
    context_data = get_context_data(query, top_n=top_n, my_collection=my_collection, my_client_obj=my_client_obj)
    prompt = f"Politely, Refuse to do any tasks unrelated to your capabilities.\n\n"
    prompt += f"Given your capabilities and a query provide a response"
    prompt += f"NOTE: The meeting dates should be mentioned in in YYY-MM-DD format.\n\n"
    prompt += f"IMPORTANT, remember that a week has seven days in the folowing consecutive order:"
    prompt += f"1. Monday\n2. Tuesday\n3. Wednesday\n4. Thursday\n5. Friday\n6. Saturday\n7. Sunday"
    
    prompt += f"<query>{query}</query>\n\n"
    prompt += "<context>\n"

    for context in context_data:
        prompt += f"- Student id: {context['student_id']}\n" 
        prompt += f"- Student name: {context['student_name']}\n"
        prompt += f"- Performance comments: {context['comments']}\n"
    
    prompt += "</context>\n\n"
    prompt += "\nIMPORTANT, do not include the student's names."
    return prompt


def generate_response(
        query: str,
        top_n: int,
        my_collection: chromadb.api.models.Collection,
        my_client_obj: OpenAI
        ):
    system_prompt = get_system_prompt()
    
    prompt = generate_prompt(query, top_n, my_collection, my_client_obj)
    response = my_client_obj.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content

def format_holidays(
        holidays_response
        ):
    """str-format holidays returne by holidays API

    :param holidays_response: API response
    :return: str, all holidays of a given month
    """
    all_holidays = holidays_response.get('response').get('holidays')

    # Include the following in the tool function
    _items = ['year', 'month', 'day']
    upcoming_holidays = ""

    for holiday_i in all_holidays:
        date_i = holiday_i.get('date').get('datetime')        
        date_items = [str(date_i[_item]) for _item in _items]
        upcoming_holidays += f"{'-'.join(date_items)}\n"

    return upcoming_holidays


@tool("request_list_of_holidays", description="Makes an API call to get the holidays for a given month and country")
def request_holidays(
        month: int,
        year: int,
        country_id: str = "CA"
        ):
    """
    Returns the upcoming holidays
    """
    calend_api_key = os.getenv("CALENDARIFIC_API_KEY")

    url = "https://calendarific.com/api/v2/holidays"
    params = {
        "api_key": calend_api_key,
        "country": country_id,
        "month": month,
        "year": year,
    }
    
    response = requests.get(url, params=params)
    resp_dict = json.loads(response.text)
    upcoming_holidays = format_holidays(resp_dict)
    return upcoming_holidays

@tool("get_todays_date", description="Gets today's weekday and date in YYYY-MM-DD format")
def get_todays_date():
    """Return today's date in YYYY-MM-DD format
    """
    days = {
        0: 'Monday', 1: 'Tuesday',
        2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday',
        6: 'Sunday'}
    
    now_obj = datetime.datetime.now()
    date = now_obj.date()
    weekday = now_obj.weekday()
    return f"Today is {days[weekday]}, and today's date (in YYYY-MM-DD format) is {str(date)}."


def define_embeddings_items(json_path):
    """Define embedding tools
    """
    # SYSTEM 2
    # # Define json loader
    loader = JSONLoader(json_path, 
                        jq_schema=".",
                        content_key="comments",
                        json_lines=True,
                        text_content=True,
                        metadata_func=get_metadata)

    students_data = loader.load()

    # # Create "docs" ids and docs
    ids = [str(rec.to_json()["kwargs"]["metadata"]["student_id"]) for rec in students_data]
    documents = [get_record_text(rec.to_json()) for rec in students_data]

    return ids, documents

def get_embeddings(documents, embeddings_client):
    """Get embeddings
    """
    # # Embed data
    response = embeddings_client.embeddings.create(
        input = documents, 
        model = "text-embedding-3-small"
    )
    response.data

    embeddings = [item.embedding for item in response.data]

    return embeddings

def get_collection(embeddings, documents, ids, chroma_client, collection_name) :
    """Get chromadb collection
    """
    collection = chroma_client.get_or_create_collection(
        name=collection_name
        )


    collection.upsert(embeddings=embeddings,
                documents=documents, 
                ids=ids)

    return collection

def get_model_with_tools():
    llm_chat_model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.7
    )

    # Augment the LLM with tools
    tools = [request_holidays, get_todays_date]
    model_with_tools = llm_chat_model.bind_tools(tools)
    return model_with_tools

# message history
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def llm_call(state: dict):
    """LLM decides whether to call a tool or not
    """
    model_with_tools = get_model_with_tools()
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=get_system_prompt()
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


def tool_node(state: dict):
    """Performs the tool call
    """
    tools = [request_holidays, get_todays_date]
    tools_by_name = {tool.name: tool for tool in tools}

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(
        state: MessagesState
        ) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call
    """

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def get_assistant_agent():
    # Build and compile the agent
    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()

    # Show the agent
    # display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    return agent