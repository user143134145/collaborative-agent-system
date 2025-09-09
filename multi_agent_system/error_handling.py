"""Enhanced error handling system for Multi-Agent AI System."""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
from multi_agent_system.data_structures import Task, Response, AgentType


class ErrorCategory(Enum):
    """Categories of errors that can occur in the system."""
    API_CONNECTION = "api_connection"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTHENTICATION = "api_authentication"
    API_TIMEOUT = "api_timeout"
    MODEL_EXECUTION = "model_execution"
    PARSING_ERROR = "parsing_error"
    MEMORY_ERROR = "memory_error"
    DOCKER_ERROR = "docker_error"
    DEPENDENCY_ERROR = "dependency_error"
    EXECUTION_ERROR = "execution_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    RESOURCE_LIMIT = "resource_limit"
    CONFIGURATION_ERROR = "configuration_error"
    AGENT_UNAVAILABLE = "agent_unavailable"
    OPENROUTER_ERROR = "openrouter_error"  # New category for OpenRouter-specific errors
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    traceback: Optional[str] = None
    task_id: Optional[str] = None
    agent_type: Optional[str] = None
    timestamp: float = None
    retry_count: int = 0
    max_retries: int = 3
    recovery_strategy: Optional[str] = None
    suggested_action: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def can_retry(self) -> bool:
        """Check if error can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1


class RetryStrategy(Enum):
    """Retry strategies for different error types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    IMMEDIATE_RETRY = "immediate_retry"
    NO_RETRY = "no_retry"


class FallbackStrategy(Enum):
    """Fallback strategies when primary approach fails."""
    ALTERNATIVE_AGENT = "alternative_agent"
    SIMPLIFIED_APPROACH = "simplified_approach"
    PARTIAL_RESULT = "partial_result"
    USER_INTERVENTION = "user_intervention"
    SKIP_STEP = "skip_step"


# Error handling configuration
ERROR_HANDLING_CONFIG: Dict[ErrorCategory, Dict[str, Any]] = {
    ErrorCategory.API_CONNECTION: {
        "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
        "max_retries": 3,
        "fallback_strategy": FallbackStrategy.ALTERNATIVE_AGENT,
        "severity": ErrorSeverity.HIGH
    },
    ErrorCategory.API_RATE_LIMIT: {
        "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
        "max_retries": 5,
        "fallback_strategy": FallbackStrategy.ALTERNATIVE_AGENT,
        "severity": ErrorSeverity.MEDIUM
    },
    ErrorCategory.API_AUTHENTICATION: {
        "retry_strategy": RetryStrategy.NO_RETRY,
        "max_retries": 0,
        "fallback_strategy": FallbackStrategy.USER_INTERVENTION,
        "severity": ErrorSeverity.CRITICAL
    },
    ErrorCategory.API_TIMEOUT: {
        "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
        "max_retries": 2,
        "fallback_strategy": FallbackStrategy.SIMPLIFIED_APPROACH,
        "severity": ErrorSeverity.MEDIUM
    },
    ErrorCategory.MODEL_EXECUTION: {
        "retry_strategy": RetryStrategy.LINEAR_BACKOFF,
        "max_retries": 2,
        "fallback_strategy": FallbackStrategy.ALTERNATIVE_AGENT,
        "severity": ErrorSeverity.HIGH
    },
    ErrorCategory.PARSING_ERROR: {
        "retry_strategy": RetryStrategy.IMMEDIATE_RETRY,
        "max_retries": 1,
        "fallback_strategy": FallbackStrategy.SIMPLIFIED_APPROACH,
        "severity": ErrorSeverity.MEDIUM
    },
    ErrorCategory.MEMORY_ERROR: {
        "retry_strategy": RetryStrategy.LINEAR_BACKOFF,
        "max_retries": 2,
        "fallback_strategy": FallbackStrategy.SKIP_STEP,
        "severity": ErrorSeverity.HIGH
    },
    ErrorCategory.DOCKER_ERROR: {
        "retry_strategy": RetryStrategy.LINEAR_BACKOFF,
        "max_retries": 2,
        "fallback_strategy": FallbackStrategy.USER_INTERVENTION,
        "severity": ErrorSeverity.HIGH
    },
    ErrorCategory.DEPENDENCY_ERROR: {
        "retry_strategy": RetryStrategy.LINEAR_BACKOFF,
        "max_retries": 3,
        "fallback_strategy": FallbackStrategy.SIMPLIFIED_APPROACH,
        "severity": ErrorSeverity.MEDIUM
    },
    ErrorCategory.EXECUTION_ERROR: {
        "retry_strategy": RetryStrategy.LINEAR_BACKOFF,
        "max_retries": 2,
        "fallback_strategy": FallbackStrategy.PARTIAL_RESULT,
        "severity": ErrorSeverity.HIGH
    },
    ErrorCategory.VALIDATION_ERROR: {
        "retry_strategy": RetryStrategy.IMMEDIATE_RETRY,
        "max_retries": 1,
        "fallback_strategy": FallbackStrategy.SKIP_STEP,
        "severity": ErrorSeverity.MEDIUM
    },
    ErrorCategory.NETWORK_ERROR: {
        "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
        "max_retries": 3,
        "fallback_strategy": FallbackStrategy.ALTERNATIVE_AGENT,
        "severity": ErrorSeverity.HIGH
    },
    ErrorCategory.RESOURCE_LIMIT: {
        "retry_strategy": RetryStrategy.NO_RETRY,
        "max_retries": 0,
        "fallback_strategy": FallbackStrategy.USER_INTERVENTION,
        "severity": ErrorSeverity.CRITICAL
    },
    ErrorCategory.CONFIGURATION_ERROR: {
        "retry_strategy": RetryStrategy.NO_RETRY,
        "max_retries": 0,
        "fallback_strategy": FallbackStrategy.USER_INTERVENTION,
        "severity": ErrorSeverity.CRITICAL
    },
    ErrorCategory.AGENT_UNAVAILABLE: {
        "retry_strategy": RetryStrategy.LINEAR_BACKOFF,
        "max_retries": 3,
        "fallback_strategy": FallbackStrategy.ALTERNATIVE_AGENT,
        "severity": ErrorSeverity.HIGH
    },
    ErrorCategory.OPENROUTER_ERROR: {
        "retry_strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
        "max_retries": 3,
        "fallback_strategy": FallbackStrategy.ALTERNATIVE_AGENT,
        "severity": ErrorSeverity.HIGH
    },
    ErrorCategory.UNKNOWN_ERROR: {
        "retry_strategy": RetryStrategy.LINEAR_BACKOFF,
        "max_retries": 1,
        "fallback_strategy": FallbackStrategy.USER_INTERVENTION,
        "severity": ErrorSeverity.MEDIUM
    }
}