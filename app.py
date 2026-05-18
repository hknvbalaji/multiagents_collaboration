"""
app.py — Flask web application for the Multi-Agent City Information System
Run:  python app.py
Open: http://localhost:5000
"""

import json
import os
import sqlite3

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv(override=True)

from agents_module import (
    clear_memory,
    get_checkpoint_detail,
    get_memory_history,
    run_agents,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


# ─── Pages ───────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── Agent API ───────────────────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def run():
    data = request.get_json()
    city = (data.get("city") or "").strip()
    prompt = (data.get("prompt") or "").strip()

    if not city:
        return jsonify({"success": False, "error": "City name is required"}), 400
    if not prompt:
        prompt = f"What's happening in {city} and what should I wear?"

    try:
        result = run_agents(city, prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Memory API ──────────────────────────────────────────────────────────────────

@app.route("/api/memory", methods=["GET"])
def memory():
    return jsonify({"history": get_memory_history()})


@app.route("/api/memory/<thread_id>", methods=["GET"])
def memory_detail(thread_id):
    return jsonify(get_checkpoint_detail(thread_id))


@app.route("/api/memory/clear", methods=["POST"])
def memory_clear():
    try:
        clear_memory()
        return jsonify({"success": True, "message": "Memory cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Data Explorer API ───────────────────────────────────────────────────────────

@app.route("/api/events", methods=["GET"])
def events():
    db_path = os.path.join(BASE_DIR, "local_info.db")
    if not os.path.exists(db_path):
        return jsonify({"error": "Events database not found. Run the notebook first to create it."}), 404
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM local_events ORDER BY city, event_date", conn)
        conn.close()
        cities = sorted(df["city"].dropna().unique().tolist()) if "city" in df.columns else []
        return jsonify({
            "events": df.to_dict(orient="records"),
            "total": len(df),
            "cities": cities,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/restaurants", methods=["GET"])
def restaurants():
    json_path = os.path.join(BASE_DIR, "restaurant_data.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "restaurant_data.json not found."}), 404
    try:
        df = pd.read_json(json_path)
        df_cleaned = df.drop_duplicates(subset=["name"], keep="first")
        cities = sorted(df_cleaned["city"].dropna().unique().tolist()) if "city" in df_cleaned.columns else []
        return jsonify({
            "restaurants": df_cleaned.to_dict(orient="records"),
            "total": len(df_cleaned),
            "cities": cities,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Multi-Agent City Information System — Flask App")
    print("=" * 55)
    print("  Open your browser at: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
