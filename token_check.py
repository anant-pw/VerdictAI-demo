
# token_check.py
import sqlite3
conn = sqlite3.connect("verdictai.db")
rows = conn.execute("""
    SELECT test_id, model, tokens_input, tokens_output, latency_ms 
    FROM llm_calls 
    ORDER BY timestamp DESC LIMIT 5
""").fetchall()
for r in rows:
    print(r)
conn.close()