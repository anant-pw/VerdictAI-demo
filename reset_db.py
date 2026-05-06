# paste this in a file called reset_db.py and run it once
import sqlite3
conn = sqlite3.connect("verdictai.db")
conn.execute("DELETE FROM test_results")
conn.execute("DELETE FROM test_runs")
conn.execute("DELETE FROM scores")
conn.execute("DELETE FROM eval_runs")
conn.execute("DELETE FROM test_cases")
conn.commit()
conn.close()
print("DB cleared.")