# database/database.py
import sqlite3, json, os

DB_PATH = os.path.join(os.path.dirname(__file__), "loan_app.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        applicant_name TEXT,
        loan_type TEXT,
        decision TEXT,
        confidence REAL,
        reasons TEXT,
        suggestions TEXT,
        features TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_application(timestamp, name, loan_type, decision, confidence, reasons, suggestions, features_dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO applications 
        (timestamp, applicant_name, loan_type, decision, confidence, reasons, suggestions, features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, name, loan_type, decision, confidence,
        reasons, suggestions, json.dumps(features_dict)
    ))
    conn.commit()
    conn.close()

def fetch_recent(limit=100):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM applications ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows
