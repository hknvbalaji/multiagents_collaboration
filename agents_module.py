"""
agents_module.py
All agent logic extracted from Multi-Agent.ipynb for use in the Flask app.
"""

import json
import logging
import os
import sqlite3
import traceback
from typing import Dict, List, Optional

import cohere
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveJsonSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "local_info.db")
MEMORY_DB_PATH = os.path.join(BASE_DIR, "agent_memory.db")
RESTAURANT_JSON_PATH = os.path.join(BASE_DIR, "restaurant_data.json")
DEFAULT_MODEL = "mistral-tiny"


# ─── State Model ────────────────────────────────────────────────────────────────

class State(BaseModel):
    city: str
    messages: List[Dict] = Field(default_factory=list)
    events_result: str = ""
    search_result: str = ""
    weather_info: Dict = Field(default_factory=dict)
    analysis_result: str = ""
    restaurant_recommendations: str = ""
    workflow_log: List[Dict] = Field(default_factory=list)


# ─── Lazy-initialized globals ───────────────────────────────────────────────────

_restaurant_vectorstore = None
_cohere_client = None
_graph_app = None
_graph_checkpointer = None
_graph_conn = None


def _get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
    return _cohere_client


def _get_restaurant_vectorstore():
    global _restaurant_vectorstore
    if _restaurant_vectorstore is None:
        logger.info("Building restaurant vector store...")
        df = pd.read_json(RESTAURANT_JSON_PATH)
        df_cleaned = df.drop_duplicates(subset=["name"], keep="first")

        data = df_cleaned.to_dict("records")
        splitter = RecursiveJsonSplitter(max_chunk_size=1000)
        split_texts, metadata_list = [], []

        for item in data:
            chunks = splitter.split_text(json_data=item, convert_lists=True)
            split_texts.extend(chunks)
            meta = {
                "city": item.get("city", ""),
                "stars": item.get("stars", 0),
                "name": item.get("name", ""),
            }
            metadata_list.extend([meta] * len(chunks))

        split_documents = [
            Document(page_content=txt, metadata=md)
            for txt, md in zip(split_texts, metadata_list)
        ]

        embeddings = CohereEmbeddings(
            model="embed-english-light-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        _restaurant_vectorstore = FAISS.from_documents(split_documents, embeddings)
        logger.info(f"Vector store ready: {len(_restaurant_vectorstore.index_to_docstore_id)} chunks")
    return _restaurant_vectorstore


def _get_graph():
    global _graph_app, _graph_checkpointer, _graph_conn
    if _graph_app is None:
        workflow = StateGraph(State)
        workflow.add_node("Events Database Agent", events_database_agent)
        workflow.add_node("Online Search Agent", search_agent)
        workflow.add_node("Weather Agent", weather_agent)
        workflow.add_node("Restaurants Recommendation Agent", query_restaurants_agent)
        workflow.add_node("Analysis Agent", analysis_agent)
        workflow.set_entry_point("Events Database Agent")

        def route_events(state):
            if f"No upcoming events found for {state.city}" in state.events_result:
                return "Online Search Agent"
            return "Weather Agent"

        workflow.add_conditional_edges(
            "Events Database Agent",
            route_events,
            {"Online Search Agent": "Online Search Agent", "Weather Agent": "Weather Agent"}
        )
        workflow.add_edge("Online Search Agent", "Weather Agent")
        workflow.add_edge("Weather Agent", "Restaurants Recommendation Agent")
        workflow.add_edge("Restaurants Recommendation Agent", "Analysis Agent")
        workflow.add_edge("Analysis Agent", END)

        _graph_conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
        _graph_checkpointer = SqliteSaver(_graph_conn)
        _graph_app = workflow.compile(checkpointer=_graph_checkpointer)
        logger.info("LangGraph compiled successfully")
    return _graph_app, _graph_checkpointer


# ─── Mistral API ─────────────────────────────────────────────────────────────────

def call_mistral_api(messages: List[Dict], model: str = DEFAULT_MODEL,
                     temperature: float = 0.7, max_tokens: int = 1000) -> Dict:
    headers = {
        "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            json=payload, headers=headers, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Mistral API Error: {str(e)}")


# ─── Tools ───────────────────────────────────────────────────────────────────────

def events_database_tool(city: str) -> str:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = """
                SELECT event_name, event_date, description
                FROM local_events
                WHERE LOWER(city) = LOWER(?)
                  AND event_date >= DATE('now')
                ORDER BY event_date
                LIMIT 3
            """
            df = pd.read_sql_query(query, conn, params=(city,))
            if not df.empty:
                events = df.apply(
                    lambda row: f"• {row['event_name']} ({row['event_date']}): {row['description']}",
                    axis=1
                )
                return "\n".join(events)
            return f"No upcoming events found for {city}."
    except sqlite3.Error as e:
        return f"Error fetching events for {city}: {str(e)}"


def weather_tool(city: str) -> str:
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Weather API key not configured."
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return (
            f"Weather in {city.title()}:\n"
            f"- Condition: {data['weather'][0]['description'].capitalize()}\n"
            f"- Temperature: {data['main']['temp']}°C (feels like {data['main']['feels_like']}°C)\n"
            f"- Humidity: {data['main']['humidity']}%"
        )
    except Exception as e:
        return f"Error fetching weather for {city}: {str(e)}"


def search_tool(city: str) -> str:
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        query = f"Upcoming events, concerts, and festivals in {city} this month"
        response = client.search(query=query, search_depth="advanced",
                                 include_answer=True, max_results=5)
        if response.get("results"):
            formatted = []
            for i, result in enumerate(response["results"][:5], 1):
                formatted.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   Date: {result.get('date', 'No date provided')}\n"
                    f"   {result.get('content', 'No details')}\n"
                    f"   URL: {result.get('url', '')}\n"
                )
            return "\n\n".join(formatted)
        return f"No event information found for {city}"
    except Exception as e:
        return f"Search failed for {city}: {str(e)}"


def query_restaurants_RAG(city: str) -> str:
    vectorstore = _get_restaurant_vectorstore()
    query = f"Find restaurants with 4 stars or higher ratings in {city} and describe their key features"
    filter_criteria = {"stars": {"$gte": 4}, "city": city}
    relevant_docs = vectorstore.similarity_search(query, k=5, filter=filter_criteria)
    context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""

    prompt = f"""Provide information about highly-rated restaurants (4 stars and above) in {city}.
Only use the context below. If no context, say you couldn't find restaurant info for {city}.

Context:
{context}

Response:"""

    client = _get_cohere_client()
    response = client.chat(model="command-a-03-2025", message=prompt, max_tokens=500)
    return response.text.strip() if hasattr(response, "text") and response.text \
        else f"No restaurant info found for {city}."


# ─── Agent Functions ─────────────────────────────────────────────────────────────

def events_database_agent(state: State) -> State:
    try:
        events_data = events_database_tool(state.city)
        if "No upcoming events found" in events_data:
            state.events_result = f"No upcoming events found for {state.city} in local database."
        else:
            state.events_result = events_data
        state.workflow_log.append({
            "agent": "Events Database Agent",
            "status": "success",
            "source": "Local SQLite DB",
            "preview": state.events_result[:300],
            "routed_to_search": "No upcoming events found" in state.events_result
        })
    except Exception as e:
        state.events_result = f"Error: {str(e)}"
        state.workflow_log.append({"agent": "Events Database Agent", "status": "error", "preview": str(e)})
    return state


def search_agent(state: State) -> State:
    try:
        raw = search_tool(state.city)
        summary = call_mistral_api(
            messages=[{"role": "user", "content": f"Summarize these events:\n\n{raw}"}],
            model="mistral-small", max_tokens=150
        )["choices"][0]["message"]["content"]
        state.search_result = summary
        state.workflow_log.append({
            "agent": "Online Search Agent",
            "status": "success",
            "source": "Tavily Web Search → Mistral Summary",
            "preview": state.search_result[:300]
        })
    except Exception as e:
        state.search_result = f"Search failed: {str(e)}"
        state.workflow_log.append({"agent": "Online Search Agent", "status": "error", "preview": str(e)})
    return state


def weather_agent(state: State) -> State:
    try:
        weather_data = weather_tool(state.city)
        state.weather_info = {"city": state.city, "weather": weather_data}
        state.workflow_log.append({
            "agent": "Weather Agent",
            "status": "success",
            "source": "OpenWeatherMap API",
            "preview": weather_data
        })
    except Exception as e:
        state.weather_info = {"city": state.city, "weather": f"Error: {str(e)}"}
        state.workflow_log.append({"agent": "Weather Agent", "status": "error", "preview": str(e)})
    return state


def query_restaurants_agent(state: State) -> State:
    try:
        state.restaurant_recommendations = query_restaurants_RAG(state.city)
        state.workflow_log.append({
            "agent": "Restaurant Recommendation Agent",
            "status": "success",
            "source": "FAISS Vector Store + Cohere RAG",
            "preview": state.restaurant_recommendations[:300]
        })
    except Exception as e:
        state.restaurant_recommendations = f"Restaurant info unavailable: {str(e)}"
        state.workflow_log.append({"agent": "Restaurant Recommendation Agent", "status": "error", "preview": str(e)})
    return state


def analysis_agent(state: State) -> State:
    prompt = f"""Analyze the following information about {state.city}:

Events from local database: {state.events_result}
Events from online search: {state.search_result}
Weather: {state.weather_info.get('weather', 'N/A')}
Restaurant Recommendations: {state.restaurant_recommendations}

Please provide:
1. A brief weather analysis
2. Suggested activities based on weather/events
3. Outfit recommendations
4. Restaurant summary"""

    try:
        response = call_mistral_api(
            model="mistral-tiny",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        state.analysis_result = response["choices"][0]["message"]["content"] \
            if response.get("choices") else "Unable to generate analysis."
        state.workflow_log.append({
            "agent": "Analysis Agent",
            "status": "success",
            "source": "Mistral LLM (mistral-tiny)",
            "preview": state.analysis_result[:300]
        })
    except Exception as e:
        state.analysis_result = f"Analysis unavailable: {str(e)}"
        state.workflow_log.append({"agent": "Analysis Agent", "status": "error", "preview": str(e)})

    state.messages.append({"role": "assistant", "content": [{"text": state.analysis_result}]})
    return state


# ─── Public API for Flask ─────────────────────────────────────────────────────────

def run_agents(city: str, user_prompt: str = None) -> Dict:
    """Run the full multi-agent pipeline. Returns a result dict for the Flask API."""
    if not user_prompt:
        user_prompt = f"What's happening in {city} and what should I wear?"

    app, _ = _get_graph()
    thread_id = city.lower().replace(" ", "_")
    config = {"recursion_limit": 150, "configurable": {"thread_id": thread_id}}

    current_state = None
    initial_state = State(
        city=city,
        messages=[{"role": "user", "content": user_prompt}],
        workflow_log=[]
    )

    for output in app.stream(initial_state, config=config):
        for agent_name, agent_state in output.items():
            if agent_name != "__end__":
                current_state = agent_state

    if current_state is None:
        return {"success": False, "error": "No state returned from agents"}

    # app.stream() yields dicts, not Pydantic model instances
    if isinstance(current_state, dict):
        weather_info = current_state.get("weather_info", {}) or {}
        return {
            "success": True,
            "city": city,
            "thread_id": thread_id,
            "analysis": current_state.get("analysis_result", ""),
            "events": current_state.get("events_result", ""),
            "search_result": current_state.get("search_result", ""),
            "weather": weather_info.get("weather", "") if isinstance(weather_info, dict) else str(weather_info),
            "restaurants": current_state.get("restaurant_recommendations", ""),
            "workflow_log": current_state.get("workflow_log", []),
        }
    else:
        weather_info = current_state.weather_info or {}
        return {
            "success": True,
            "city": city,
            "thread_id": thread_id,
            "analysis": current_state.analysis_result,
            "events": current_state.events_result,
            "search_result": current_state.search_result,
            "weather": weather_info.get("weather", "") if isinstance(weather_info, dict) else str(weather_info),
            "restaurants": current_state.restaurant_recommendations,
            "workflow_log": current_state.workflow_log,
        }


def get_memory_history() -> List[Dict]:
    """Return list of all past agent runs stored in agent_memory.db."""
    if not os.path.exists(MEMORY_DB_PATH):
        return []
    try:
        _, checkpointer = _get_graph()
        history, seen_threads = [], set()

        for tup in checkpointer.list(None, limit=200):
            thread_id = tup.config.get("configurable", {}).get("thread_id", "")
            if not thread_id or thread_id in seen_threads:
                continue
            seen_threads.add(thread_id)
            cv = tup.checkpoint.get("channel_values", {})
            history.append({
                "thread_id": thread_id,
                "city": cv.get("city", thread_id.replace("_", " ").title()),
                "checkpoint_id": tup.checkpoint.get("id", ""),
                "analysis_preview": str(cv.get("analysis_result", ""))[:200],
                "has_events": bool(cv.get("events_result", "")),
                "has_restaurants": bool(cv.get("restaurant_recommendations", "")),
            })
        return history
    except Exception as e:
        logger.error(f"Memory history error: {e}")
        return []


def get_checkpoint_detail(thread_id: str) -> Dict:
    """Return full saved state for a given thread_id."""
    if not os.path.exists(MEMORY_DB_PATH):
        return {"error": "No memory database found"}
    try:
        _, checkpointer = _get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        tup = checkpointer.get_tuple(config)
        if not tup:
            return {"error": f"No checkpoint found for: {thread_id}"}
        cv = tup.checkpoint.get("channel_values", {})
        return {
            "thread_id": thread_id,
            "city": cv.get("city", ""),
            "events_result": cv.get("events_result", ""),
            "search_result": cv.get("search_result", ""),
            "weather_info": cv.get("weather_info", {}),
            "analysis_result": cv.get("analysis_result", ""),
            "restaurant_recommendations": cv.get("restaurant_recommendations", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def clear_memory():
    """Wipe all checkpoint data and reset the cached graph."""
    import gc
    global _graph_app, _graph_checkpointer, _graph_conn

    # Drop all references so SqliteSaver releases the connection
    old_conn = _graph_conn
    _graph_app = None
    _graph_checkpointer = None
    _graph_conn = None

    if old_conn:
        try:
            old_conn.close()
        except Exception:
            pass

    # Force CPython to collect the now-unreferenced SqliteSaver and its cursor
    gc.collect()

    if not os.path.exists(MEMORY_DB_PATH):
        return

    # Open a brand-new independent connection to wipe the data
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH, timeout=10)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (table,) in cursor.fetchall():
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        logger.info("Memory cleared successfully")
    except Exception as e:
        logger.error(f"Clear memory error: {e}")
        raise
