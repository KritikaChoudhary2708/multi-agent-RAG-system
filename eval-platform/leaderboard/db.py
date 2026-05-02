import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "leaderboard.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            model TEXT,
            category TEXT,
            prompt_id TEXT,
            rule_score REAL,
            llm_score REAL,
            ensemble_score REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_result(run_id, model, category,prompt_id, rule_score, llm_score, ensemble_score):
          conn = get_conn()
          cur = conn.cursor()
          now = datetime.utcnow().isoformat()
          cur.execute("""
                    INSERT INTO results 
                    (run_id, model, category, prompt_id, rule_score, llm_score, ensemble_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (run_id, model, category, prompt_id, rule_score, llm_score, ensemble_score, now))
          conn.commit()
          conn.close()

def get_leaderboard():
          conn = get_conn()
          cur = conn.cursor()
          cur.execute('''
                      SELECT model, COUNT(*) as total_runs, AVG(ensemble_score) as avg_ensemble, AVG(rule_score) as avg_rule, AVG(llm_score) as avg_llm
                      FROM results
                      GROUP BY model
                      ORDER BY avg_ensemble DESC
          ''')
          res = cur.fetchall()
          conn.close()
          return [dict(row) for row in res]

