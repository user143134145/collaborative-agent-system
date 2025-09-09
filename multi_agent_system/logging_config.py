"""Logging configuration for Multi-Agent AI System."""

import logging
import structlog
from typing import Any, Dict
from .config import config


def configure_logging() -> None:
    """Configure structured logging for the system."""
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class SystemLogger:
    """Centralized logging for the multi-agent system."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)
    
    def log_task_start(self, task_id: str, agent_type: str, task_type: str) -> None:
        """Log task start."""
        self.info(
            "Task started",
            task_id=task_id,
            agent_type=agent_type,
            task_type=task_type
        )
    
    def log_task_completion(self, task_id: str, agent_type: str, 
                          execution_time: float, success: bool) -> None:
        """Log task completion."""
        self.info(
            "Task completed",
            task_id=task_id,
            agent_type=agent_type,
            execution_time=execution_time,
            success=success
        )
    
    def log_api_call(self, agent_type: str, model: str, tokens_used: int,
                    response_time: float) -> None:
        """Log API call metrics."""
        self.info(
            "API call completed",
            agent_type=agent_type,
            model=model,
            tokens_used=tokens_used,
            response_time=response_time
        )
    
    def log_memory_operation(self, operation: str, artifact_count: int,
                           search_time: float = None) -> None:
        """Log memory system operations."""
        log_data = {
            "operation": operation,
            "artifact_count": artifact_count
        }
        if search_time is not None:
            log_data["search_time"] = search_time
        
        self.info("Memory operation", **log_data)
    
    def log_cross_validation(self, task_id: str, consensus_score: float,
                           agent_count: int) -> None:
        """Log cross-validation results."""
        self.info(
            "Cross-validation completed",
            task_id=task_id,
            consensus_score=consensus_score,
            agent_count=agent_count
        )


# Initialize logging
configure_logging()