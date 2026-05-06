# database/models.py
"""
SQLite schema for VerdictAI evaluation results.
Stores runs, test cases, LLM calls, and scores.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json


class DatabaseManager:
    """Manages SQLite database for evaluation results."""
    
    def __init__(self, db_path: str = "verdictai.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Return dict-like rows
        self._create_tables()
    
    def _create_tables(self):
        """Create schema if not exists."""
        cursor = self.conn.cursor()
        
        # === EXISTING TABLES ===
        
        # Eval runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                config TEXT,
                status TEXT DEFAULT 'running',
                total_tests INTEGER DEFAULT 0,
                passed_tests INTEGER DEFAULT 0,
                failed_tests INTEGER DEFAULT 0,
                avg_score REAL,
                total_cost REAL DEFAULT 0.0
            )
        """)
        
        # Test cases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                test_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                test_name TEXT,
                input_data TEXT,
                expected_output TEXT,
                actual_output TEXT,
                passed BOOLEAN,
                error_message TEXT,
                executed_at TEXT,
                PRIMARY KEY (test_id, run_id),
                FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
            )
        """)
        
        # LLM calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt TEXT,
                response TEXT,
                latency_ms REAL,
                cost_usd REAL DEFAULT 0.0,
                tokens_input INTEGER,
                tokens_output INTEGER,
                timestamp TEXT,
                FOREIGN KEY (test_id) REFERENCES test_cases(test_id)
            )
        """)
        
        # Scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                score_value REAL NOT NULL,
                details TEXT,
                calculated_at TEXT,
                FOREIGN KEY (test_id) REFERENCES test_cases(test_id)
            )
        """)
        
        # === NEW TABLES FOR DASHBOARD ===
        
        # Test runs table (for dashboard)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id TEXT PRIMARY KEY,
                suite_name TEXT,
                start_time TEXT,
                end_time TEXT,
                total_tests INTEGER DEFAULT 0,
                passed_tests INTEGER DEFAULT 0,
                failed_tests INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0,
                avg_relevance REAL,
                avg_hallucination REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Test results table (for dashboard)
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
                regressed INTEGER DEFAULT 0,
                score_drop REAL,
                FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
                UNIQUE(run_id, test_id)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_results_run_id 
            ON test_results(run_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_results_timestamp 
            ON test_results(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_results_verdict 
            ON test_results(verdict)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_results_test_id 
            ON test_results(test_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_runs_start_time 
            ON test_runs(start_time DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scores_test_id_metric 
            ON scores(test_id, metric_name)
        """)

        # Prevent duplicate scores for same test+metric across re-runs
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_scores_unique_test_metric
            ON scores(test_id, metric_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_llm_calls_test_id 
            ON llm_calls(test_id)
        """)
        
        self.conn.commit()
    
    # === EVAL RUN METHODS ===
    
    def create_run(self, run_id: str, config: dict) -> None:
        """Start new eval run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO eval_runs (run_id, started_at, config)
            VALUES (?, ?, ?)
        """, (run_id, datetime.now().isoformat(), json.dumps(config)))
        self.conn.commit()
    
    def update_run(self, run_id: str, **kwargs) -> None:
        """Update eval run fields."""
        allowed_fields = ['ended_at', 'status', 'total_tests', 'passed_tests', 
                         'failed_tests', 'avg_score', 'total_cost']
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [run_id]
        
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE eval_runs SET {set_clause} WHERE run_id = ?", values)
        self.conn.commit()
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get eval run by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # === TEST CASE METHODS ===
    
    def save_test_case(self, test_id: str, run_id: str, **kwargs) -> None:
        """Save test case result."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO test_cases 
            (test_id, run_id, test_name, input_data, expected_output, 
             actual_output, passed, error_message, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
            run_id,
            kwargs.get('test_name'),
            kwargs.get('input_data'),
            kwargs.get('expected_output'),
            kwargs.get('actual_output'),
            kwargs.get('passed'),
            kwargs.get('error_message'),
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def get_test_cases(self, run_id: str) -> List[Dict]:
        """Get all test cases for a run."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM test_cases WHERE run_id = ?", (run_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_test_case(self, test_id: str, run_id: str) -> Optional[Dict]:
        """Get specific test case."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM test_cases WHERE test_id = ? AND run_id = ?", 
            (test_id, run_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # === LLM CALL METHODS ===
    
    def save_llm_call(self, test_id: str, model: str, prompt: str, 
                      response: str, latency_ms: float, **kwargs) -> None:
        """Log LLM API call."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO llm_calls 
            (test_id, model, prompt, response, latency_ms, cost_usd, 
             tokens_input, tokens_output, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
            model,
            prompt,
            response,
            latency_ms,
            kwargs.get('cost_usd', 0.0),
            kwargs.get('tokens_input'),
            kwargs.get('tokens_output'),
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def get_llm_calls(self, test_id: str) -> List[Dict]:
        """Get LLM calls for a test."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM llm_calls WHERE test_id = ? ORDER BY timestamp DESC", 
            (test_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # === SCORE METHODS ===
    
    def save_score(self, test_id: str, metric_name: str, score_value: float, 
                   details: str = None) -> None:
        """Save evaluation score."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO scores (test_id, metric_name, score_value, details, calculated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (test_id, metric_name, score_value, details, datetime.now().isoformat()))
        self.conn.commit()
    
    def get_scores(self, test_id: str) -> List[Dict]:
        """Get all scores for a test."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM scores WHERE test_id = ?", (test_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_score(self, test_id: str, metric_name: str) -> Optional[float]:
        """Get specific score for a test."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT score_value FROM scores WHERE test_id = ? AND metric_name = ?",
            (test_id, metric_name)
        )
        row = cursor.fetchone()
        return row['score_value'] if row else None
    
    # === DASHBOARD METHODS ===
    
    def create_test_run(self, run_id: str, suite_name: str, start_time: datetime = None) -> None:
        """Create a test run entry for dashboard."""
        if start_time is None:
            start_time = datetime.now()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO test_runs (run_id, suite_name, start_time)
            VALUES (?, ?, ?)
        """, (run_id, suite_name, start_time.isoformat()))
        self.conn.commit()
    
    def save_test_result(self, run_id: str, test_id: str, verdict: str, 
                        score: float = None, relevance_score: float = None,
                        hallucination_score: float = None, reason: str = None,
                        latency_ms: int = None, metadata: dict = None,
                        regressed: int = 0, score_drop: float = None) -> None:
        """Save test result to dashboard table."""
        cursor = self.conn.cursor()
        
        # Check if result exists
        cursor.execute(
            "SELECT id FROM test_results WHERE run_id = ? AND test_id = ?",
            (run_id, test_id)
        )
        exists = cursor.fetchone()
        
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        if exists:
            cursor.execute("""
                UPDATE test_results 
                SET verdict = ?, score = ?, relevance_score = ?, 
                    hallucination_score = ?, reason = ?, timestamp = ?,
                    latency_ms = ?, metadata = ?, regressed = ?, score_drop = ?
                WHERE run_id = ? AND test_id = ?
            """, (
                verdict, score, relevance_score, hallucination_score,
                reason, timestamp, latency_ms, metadata_json, 
                regressed, score_drop, run_id, test_id
            ))
        else:
            cursor.execute("""
                INSERT INTO test_results 
                (run_id, test_id, verdict, score, relevance_score, 
                 hallucination_score, reason, timestamp, latency_ms, 
                 metadata, regressed, score_drop)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, test_id, verdict, score, relevance_score,
                hallucination_score, reason, timestamp, latency_ms,
                metadata_json, regressed, score_drop
            ))
        
        self.conn.commit()
    
    def update_test_run_summary(self, run_id: str) -> None:
        """Update test run summary statistics."""
        cursor = self.conn.cursor()
        
        # Get stats from test_results
        cursor.execute("""
            SELECT 
                COUNT(*) as total_tests,
                SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END) as passed_tests,
                SUM(CASE WHEN verdict = 'FAIL' THEN 1 ELSE 0 END) as failed_tests,
                AVG(score) as avg_score,
                AVG(relevance_score) as avg_relevance,
                AVG(hallucination_score) as avg_hallucination
            FROM test_results
            WHERE run_id = ?
        """, (run_id,))
        
        stats = cursor.fetchone()
        
        if stats:
            cursor.execute("""
                UPDATE test_runs
                SET total_tests = ?, passed_tests = ?, failed_tests = ?,
                    avg_score = ?, avg_relevance = ?, avg_hallucination = ?,
                    end_time = ?
                WHERE run_id = ?
            """, (
                stats['total_tests'] or 0,
                stats['passed_tests'] or 0,
                stats['failed_tests'] or 0,
                stats['avg_score'] or 0,
                stats['avg_relevance'],
                stats['avg_hallucination'],
                datetime.now().isoformat(),
                run_id
            ))
        
        self.conn.commit()
    
    def get_test_runs(self, limit: int = 50) -> List[Dict]:
        """Get recent test runs."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM test_runs 
            ORDER BY start_time DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_test_results(self, run_id: str = None, limit: int = 1000) -> List[Dict]:
        """Get test results, optionally filtered by run."""
        cursor = self.conn.cursor()
        
        if run_id:
            cursor.execute("""
                SELECT tr.*, ts.suite_name, ts.start_time as run_start
                FROM test_results tr
                LEFT JOIN test_runs ts ON tr.run_id = ts.run_id
                WHERE tr.run_id = ?
                ORDER BY tr.timestamp DESC
                LIMIT ?
            """, (run_id, limit))
        else:
            cursor.execute("""
                SELECT tr.*, ts.suite_name, ts.start_time as run_start
                FROM test_results tr
                LEFT JOIN test_runs ts ON tr.run_id = ts.run_id
                ORDER BY tr.timestamp DESC
                LIMIT ?
            """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def sync_from_eval_tables(self, run_id: str = None) -> None:
        """Sync data from eval_runs/test_cases/scores to dashboard tables."""
        cursor = self.conn.cursor()
        
        # Sync test_runs from eval_runs
        if run_id:
            cursor.execute("""
                SELECT run_id, config as suite_name, started_at as start_time
                FROM eval_runs
                WHERE run_id = ?
            """, (run_id,))
        else:
            cursor.execute("""
                SELECT run_id, config as suite_name, started_at as start_time
                FROM eval_runs
            """)
        
        eval_runs = cursor.fetchall()
        
        for eval_run in eval_runs:
            # Insert into test_runs
            cursor.execute("""
                INSERT OR IGNORE INTO test_runs (run_id, suite_name, start_time)
                VALUES (?, ?, ?)
            """, (eval_run['run_id'], eval_run['suite_name'], eval_run['start_time']))
            
            # Sync test results
            cursor.execute("""
                SELECT 
                    tc.test_id,
                    tc.run_id,
                    CASE WHEN tc.passed = 1 THEN 'PASS' ELSE 'FAIL' END as verdict,
                    tc.executed_at as timestamp,
                    tc.expected_output,
                    tc.actual_output,
                    tc.input_data,
                    tc.error_message
                FROM test_cases tc
                WHERE tc.run_id = ?
            """, (eval_run['run_id'],))
            
            test_cases = cursor.fetchall()
            
            for tc in test_cases:
                # Get scores - FIXED: Use correct metric names
                overall_score = self.get_score(tc['test_id'], 'judge_score')  # Changed from 'overall_score'
                relevance_score = self.get_score(tc['test_id'], 'relevance_score')  # Changed from 'relevance'
                hallucination_score = self.get_score(tc['test_id'], 'hallucination_score')  # Changed from 'hallucination'
                
                # Get reason from judge_score details
                reason = None
                cursor.execute("""
                    SELECT details FROM scores 
                    WHERE test_id = ? AND metric_name = 'judge_score'
                    LIMIT 1
                """, (tc['test_id'],))
                score_detail = cursor.fetchone()
                if score_detail and score_detail['details']:
                    reason = score_detail['details']
                
                # Get latency
                cursor.execute("""
                    SELECT AVG(latency_ms) as avg_latency
                    FROM llm_calls
                    WHERE test_id = ?
                """, (tc['test_id'],))
                latency = cursor.fetchone()
                
                # Prepare metadata
                metadata = {
                    'expected_output': tc['expected_output'],
                    'actual_output': tc['actual_output'],
                    'input_data': tc['input_data'],
                    'error_message': tc['error_message']
                }
                
                # Save to test_results
                self.save_test_result(
                    run_id=tc['run_id'],
                    test_id=tc['test_id'],
                    verdict=tc['verdict'],
                    score=overall_score,
                    relevance_score=relevance_score,
                    hallucination_score=hallucination_score,
                    reason=reason,
                    latency_ms=latency['avg_latency'] if latency else None,
                    metadata=metadata
                )
            
            # Update run summary
            self.update_test_run_summary(eval_run['run_id'])
        
        self.conn.commit()
    
    # === QUERY METHODS ===
    
    def get_run_summary(self, run_id: str) -> Dict:
        """Get summary stats for a run."""
        cursor = self.conn.cursor()
        
        # Get run info
        run = self.get_run(run_id)
        if not run:
            return {}
        
        # Get average scores by metric
        cursor.execute("""
            SELECT metric_name, AVG(score_value) as avg_score, COUNT(*) as count
            FROM scores
            WHERE test_id IN (SELECT test_id FROM test_cases WHERE run_id = ?)
            GROUP BY metric_name
        """, (run_id,))
        
        metrics = {row['metric_name']: {
            'avg': row['avg_score'],
            'count': row['count']
        } for row in cursor.fetchall()}
        
        return {
            'run_info': run,
            'metrics': metrics
        }
    
    def get_dashboard_stats(self) -> Dict:
        """Get overall dashboard statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT run_id) as total_runs,
                COUNT(*) as total_tests,
                SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END) as passed_tests,
                AVG(score) as avg_score,
                AVG(relevance_score) as avg_relevance,
                AVG(hallucination_score) as avg_hallucination
            FROM test_results
        """)
        
        stats = cursor.fetchone()
        
        return dict(stats) if stats else {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()