"""
Structured logging for VerdictAI evaluation pipeline.
Handles file + console output with different levels.
"""

import sys
from pathlib import Path
from loguru import logger
from datetime import datetime


class EvalLogger:
    """Centralized logger for evaluation runs."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = log_level
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure loguru with file + console handlers."""
        # Remove default handler
        logger.remove()
        
        # Console handler - clean format
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level=self.log_level,
            colorize=True
        )
        
        # File handler - JSON format for parsing
        log_file = self.log_dir / f"verdictai_{datetime.now().strftime('%Y%m%d')}.log"
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        # Structured JSON logs for analytics
        json_file = self.log_dir / f"verdictai_{datetime.now().strftime('%Y%m%d')}.jsonl"
        logger.add(
            json_file,
            format="{message}",
            level="INFO",
            serialize=True,
            rotation="10 MB",
            retention="30 days"
        )
    
    def eval_start(self, run_id: str, config: dict):
        """Log evaluation run start."""
        logger.info(f"Eval run started", extra={
            "event": "eval_start",
            "run_id": run_id,
            "config": config
        })
    
    def eval_end(self, run_id: str, summary: dict):
        """Log evaluation run completion."""
        logger.info(f"Eval run completed", extra={
            "event": "eval_end",
            "run_id": run_id,
            "summary": summary
        })
    
    def test_case_start(self, test_id: str, test_data: dict):
        """Log test case execution start."""
        logger.debug(f"Test case started: {test_id}", extra={
            "event": "test_start",
            "test_id": test_id,
            "data": test_data
        })
    
    def test_case_end(self, test_id: str, result: dict):
        level = "SUCCESS" if result.get("passed") else "ERROR"
        # read "verdict" not "status"
        status = result.get("verdict") or result.get("status", "UNKNOWN")
        logger.log(level, f"Test case {test_id}: {status}", extra={
            "event": "test_end",
            "test_id": test_id,
            "result": result
        })
    
    def llm_call(self, model: str, prompt: str, response: str, latency: float, cost: float = 0.0):
        """Log LLM API call details."""
        logger.debug(f"LLM call to {model}", extra={
            "event": "llm_call",
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "latency_ms": latency,
            "cost_usd": cost
        })
    
    def score_calculated(self, test_id: str, metric: str, score: float, details: dict = None):
        """Log scoring metrics."""
        logger.info(f"Score: {metric} = {score:.2f}", extra={
            "event": "score",
            "test_id": test_id,
            "metric": metric,
            "score": score,
            "details": details or {}
        })
    
    def error(self, message: str, error: Exception = None, context: dict = None):
        """Log errors with context."""
        logger.error(message, extra={
            "event": "error",
            "error_type": type(error).__name__ if error else None,
            "error_msg": str(error) if error else None,
            "context": context or {}
        })


# Global logger instance
eval_logger = EvalLogger()