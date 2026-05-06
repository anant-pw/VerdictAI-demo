# migrate_schema.py
import sqlite3
import json
from datetime import datetime

def migrate_database():
    """Migrate from old schema to new dashboard-compatible schema."""
    conn = sqlite3.connect('verdictai.db')
    cursor = conn.cursor()
    
    # Create missing tables if they don't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_runs (
            run_id TEXT PRIMARY KEY,
            suite_name TEXT,
            start_time TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            test_id TEXT,
            verdict TEXT,
            score REAL,
            relevance_score REAL,
            hallucination_score REAL,
            reason TEXT,
            timestamp TEXT,
            latency_ms INTEGER,
            metadata TEXT,
            regressed INTEGER
        )
    """)
    
    # Migrate data from eval_runs to test_runs
    cursor.execute("""
        INSERT OR IGNORE INTO test_runs (run_id, suite_name, start_time)
        SELECT run_id, config, started_at FROM eval_runs
        WHERE run_id NOT IN (SELECT run_id FROM test_runs)
    """)
    
    # Migrate data from test_cases and scores to test_results
    cursor.execute("""
        INSERT OR REPLACE INTO test_results (
            run_id, test_id, verdict, timestamp, metadata
        )
        SELECT 
            tc.run_id,
            tc.test_id,
            CASE WHEN tc.passed = 1 THEN 'PASS' ELSE 'FAIL' END as verdict,
            tc.executed_at as timestamp,
            json_object(
                'expected_output', tc.expected_output,
                'actual_output', tc.actual_output,
                'input_data', tc.input_data,
                'error_message', tc.error_message
            ) as metadata
        FROM test_cases tc
        WHERE (tc.run_id, tc.test_id) NOT IN (
            SELECT run_id, test_id FROM test_results
        )
    """)
    
    # Update scores
    # Get overall scores
    cursor.execute("""
        UPDATE test_results
        SET score = (
            SELECT score_value 
            FROM scores s 
            WHERE s.test_id = test_results.test_id 
            AND s.metric_name = 'overall_score'
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM scores s 
            WHERE s.test_id = test_results.test_id 
            AND s.metric_name = 'overall_score'
        )
    """)
    
    # Get relevance scores
    cursor.execute("""
        UPDATE test_results
        SET relevance_score = (
            SELECT score_value 
            FROM scores s 
            WHERE s.test_id = test_results.test_id 
            AND s.metric_name = 'relevance'
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM scores s 
            WHERE s.test_id = test_results.test_id 
            AND s.metric_name = 'relevance'
        )
    """)
    
    # Get hallucination scores
    cursor.execute("""
        UPDATE test_results
        SET hallucination_score = (
            SELECT score_value 
            FROM scores s 
            WHERE s.test_id = test_results.test_id 
            AND s.metric_name = 'hallucination'
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM scores s 
            WHERE s.test_id = test_results.test_id 
            AND s.metric_name = 'hallucination'
        )
    """)
    
    # Get latency from llm_calls
    cursor.execute("""
        UPDATE test_results
        SET latency_ms = (
            SELECT AVG(latency_ms)
            FROM llm_calls lc
            WHERE lc.test_id = test_results.test_id
        )
        WHERE EXISTS (
            SELECT 1 FROM llm_calls lc
            WHERE lc.test_id = test_results.test_id
        )
    """)
    
    conn.commit()
    print("Migration completed successfully!")
    
    # Show summary
    cursor.execute("SELECT COUNT(*) FROM test_results")
    count = cursor.fetchone()[0]
    print(f"Total test results migrated: {count}")
    
    conn.close()

if __name__ == "__main__":
    migrate_database()