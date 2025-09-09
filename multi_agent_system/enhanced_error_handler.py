"""Enhanced error handler for Multi-Agent AI System."""

import asyncio
import time
import traceback
from typing import Dict, List, Optional, Any, Callable
from multi_agent_system.error_handling import ErrorInfo, ErrorCategory, ErrorSeverity, RetryStrategy, FallbackStrategy, ERROR_HANDLING_CONFIG
from multi_agent_system.data_structures import Task, Response, AgentType
from multi_agent_system.logging_config import SystemLogger


class EnhancedErrorHandler:
    """Enhanced error handling system with adaptive retry and fallback mechanisms."""
    
    def __init__(self):
        self.logger = SystemLogger("enhanced_error_handler")
        self.error_history: Dict[str, List[ErrorInfo]] = {}
        
    def categorize_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Categorize an error based on exception type and context."""
        error_message = str(exception)
        error_type = type(exception).__name__
        
        # Determine error category based on exception type and message
        category = self._determine_category(exception, error_message, context)
        severity = self._determine_severity(category, exception, context)
        
        # Get configuration for this error category
        config = ERROR_HANDLING_CONFIG.get(category, ERROR_HANDLING_CONFIG[ErrorCategory.UNKNOWN_ERROR])
        
        error_info = ErrorInfo(
            error_id=f"err_{int(time.time() * 1000000)}",
            category=category,
            severity=severity,
            message=error_message,
            traceback=traceback.format_exc(),
            task_id=context.get("task_id") if context else None,
            agent_type=context.get("agent_type") if context else None,
            max_retries=config["max_retries"],
            recovery_strategy=config["retry_strategy"].value,
            suggested_action=self._get_suggested_action(category)
        )
        
        # Store error in history
        task_id = error_info.task_id or "unknown"
        if task_id not in self.error_history:
            self.error_history[task_id] = []
        self.error_history[task_id].append(error_info)
        
        self.logger.error(
            "Error categorized",
            error_id=error_info.error_id,
            category=category.value,
            severity=severity.value,
            error_message=error_message,
            task_id=task_id
        )
        
        return error_info
    
    def _determine_category(self, exception: Exception, error_message: str, context: Dict[str, Any] = None) -> ErrorCategory:
        """Determine error category based on exception and message."""
        error_lower = error_message.lower()
        
        # OpenRouter-specific errors
        if "openrouter" in error_lower:
            return ErrorCategory.OPENROUTER_ERROR
            
        # Connection errors
        if "connection" in error_lower or "connect" in error_lower:
            return ErrorCategory.API_CONNECTION
            
        # Rate limit errors
        if "rate limit" in error_lower or "quota" in error_lower or "limit exceeded" in error_lower:
            return ErrorCategory.API_RATE_LIMIT
            
        # Authentication errors
        if "auth" in error_lower or "unauthorized" in error_lower or "forbidden" in error_lower:
            return ErrorCategory.API_AUTHENTICATION
            
        # Timeout errors
        if "timeout" in error_lower or "timed out" in error_lower:
            return ErrorCategory.API_TIMEOUT
            
        # Parsing errors
        if "parse" in error_lower or "json" in error_lower or "invalid" in error_lower:
            return ErrorCategory.PARSING_ERROR
            
        # Memory errors
        if "memory" in error_lower or "out of memory" in error_lower:
            return ErrorCategory.MEMORY_ERROR
            
        # Docker errors
        if "docker" in error_lower or "container" in error_lower:
            return ErrorCategory.DOCKER_ERROR
            
        # Dependency errors
        if "import" in error_lower or "module" in error_lower or "package" in error_lower:
            return ErrorCategory.DEPENDENCY_ERROR
            
        # Execution errors
        if "execution" in error_lower or "runtime" in error_lower:
            return ErrorCategory.EXECUTION_ERROR
            
        # Validation errors
        if "validation" in error_lower or "validate" in error_lower:
            return ErrorCategory.VALIDATION_ERROR
            
        # Network errors
        if "network" in error_lower or "dns" in error_lower or "socket" in error_lower:
            return ErrorCategory.NETWORK_ERROR
            
        # Resource limit errors
        if "resource" in error_lower or "limit" in error_lower or "quota" in error_lower:
            return ErrorCategory.RESOURCE_LIMIT
            
        # Configuration errors
        if "config" in error_lower or "configuration" in error_lower:
            return ErrorCategory.CONFIGURATION_ERROR
            
        # Agent unavailable errors
        if "agent" in error_lower or "unavailable" in error_lower:
            return ErrorCategory.AGENT_UNAVAILABLE
            
        # Default to unknown
        return ErrorCategory.UNKNOWN_ERROR
    
    def _determine_severity(self, category: ErrorCategory, exception: Exception, context: Dict[str, Any] = None) -> ErrorSeverity:
        """Determine error severity."""
        # Use configured severity as default
        config = ERROR_HANDLING_CONFIG.get(category, ERROR_HANDLING_CONFIG[ErrorCategory.UNKNOWN_ERROR])
        default_severity = config["severity"]
        
        # Adjust based on context
        if context and context.get("critical_task"):
            # Increase severity for critical tasks
            if default_severity == ErrorSeverity.LOW:
                return ErrorSeverity.MEDIUM
            elif default_severity == ErrorSeverity.MEDIUM:
                return ErrorSeverity.HIGH
            else:
                return ErrorSeverity.CRITICAL
        elif context and context.get("retry_count", 0) > 2:
            # Increase severity for repeated errors
            if default_severity == ErrorSeverity.LOW:
                return ErrorSeverity.MEDIUM
            elif default_severity == ErrorSeverity.MEDIUM:
                return ErrorSeverity.HIGH
            else:
                return ErrorSeverity.CRITICAL
                
        return default_severity
    
    def _get_suggested_action(self, category: ErrorCategory) -> str:
        """Get suggested action for error category."""
        suggestions = {
            ErrorCategory.API_CONNECTION: "Check network connectivity and API endpoint availability",
            ErrorCategory.API_RATE_LIMIT: "Wait for rate limit to reset or use alternative API key",
            ErrorCategory.API_AUTHENTICATION: "Verify API keys and authentication credentials",
            ErrorCategory.API_TIMEOUT: "Increase timeout settings or retry with smaller request",
            ErrorCategory.MODEL_EXECUTION: "Check model availability and input parameters",
            ErrorCategory.PARSING_ERROR: "Validate input format and check for malformed data",
            ErrorCategory.MEMORY_ERROR: "Reduce input size or increase available memory",
            ErrorCategory.DOCKER_ERROR: "Check Docker installation and permissions",
            ErrorCategory.DEPENDENCY_ERROR: "Install missing dependencies or check package versions",
            ErrorCategory.EXECUTION_ERROR: "Review code for syntax or runtime errors",
            ErrorCategory.VALIDATION_ERROR: "Check input validation rules and data format",
            ErrorCategory.NETWORK_ERROR: "Check network connectivity and firewall settings",
            ErrorCategory.RESOURCE_LIMIT: "Increase resource limits or optimize resource usage",
            ErrorCategory.CONFIGURATION_ERROR: "Review configuration files and environment variables",
            ErrorCategory.AGENT_UNAVAILABLE: "Check agent status and restart if necessary",
            ErrorCategory.OPENROUTER_ERROR: "Check OpenRouter API key, model availability, and rate limits",
            ErrorCategory.UNKNOWN_ERROR: "Review error details and contact system administrator"
        }
        return suggestions.get(category, "Review error details for troubleshooting")
    
    async def handle_with_retry(self, operation: Callable, error_context: Dict[str, Any] = None, 
                              max_retries: int = None) -> Any:
        """Execute operation with adaptive retry mechanism."""
        retry_count = 0
        last_exception = None
        
        while True:
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                error_info = self.categorize_error(e, error_context)
                
                # Override max_retries if provided
                if max_retries is not None:
                    error_info.max_retries = max_retries
                    
                # Check if we can retry
                if not error_info.can_retry():
                    self.logger.error(
                        "Max retries exceeded",
                        error_id=error_info.error_id,
                        retry_count=retry_count,
                        max_retries=error_info.max_retries
                    )
                    break
                    
                # Apply retry strategy
                retry_count += 1
                error_info.increment_retry()
                
                retry_delay = self._calculate_retry_delay(error_info, retry_count)
                self.logger.info(
                    "Retrying operation",
                    error_id=error_info.error_id,
                    retry_count=retry_count,
                    delay=retry_delay
                )
                
                # Wait before retry
                await asyncio.sleep(retry_delay)
        
        # If we get here, all retries failed
        raise last_exception
    
    def _calculate_retry_delay(self, error_info: ErrorInfo, retry_count: int) -> float:
        """Calculate delay before retry based on strategy."""
        strategy = RetryStrategy(error_info.recovery_strategy)
        
        if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(2 ** retry_count, 60)  # Max 60 seconds
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(retry_count * 2, 30)  # Max 30 seconds
        elif strategy == RetryStrategy.FIXED_INTERVAL:
            return 5.0  # 5 seconds
        elif strategy == RetryStrategy.IMMEDIATE_RETRY:
            return 0.1  # 100ms
        else:
            return 1.0  # Default 1 second
    
    async def handle_with_fallback(self, primary_operation: Callable, 
                                 fallback_operation: Callable,
                                 error_context: Dict[str, Any] = None) -> Any:
        """Execute operation with fallback mechanism."""
        try:
            return await primary_operation()
        except Exception as e:
            error_info = self.categorize_error(e, error_context)
            self.logger.warning(
                "Primary operation failed, attempting fallback",
                error_id=error_info.error_id,
                category=error_info.category.value
            )
            
            try:
                return await fallback_operation()
            except Exception as fallback_e:
                self.logger.error(
                    "Fallback operation also failed",
                    primary_error_id=error_info.error_id,
                    fallback_error=str(fallback_e)
                )
                raise fallback_e
    
    def get_error_history(self, task_id: str = None) -> List[ErrorInfo]:
        """Get error history for a task or all tasks."""
        if task_id:
            return self.error_history.get(task_id, [])
        else:
            # Flatten all error histories
            all_errors = []
            for errors in self.error_history.values():
                all_errors.extend(errors)
            return all_errors
    
    def clear_error_history(self, task_id: str = None) -> None:
        """Clear error history for a task or all tasks."""
        if task_id:
            if task_id in self.error_history:
                del self.error_history[task_id]
        else:
            self.error_history.clear()
    
    def generate_error_report(self, task_id: str = None) -> Dict[str, Any]:
        """Generate detailed error report."""
        errors = self.get_error_history(task_id)
        
        # Group by category
        category_counts = {}
        severity_counts = {}
        
        for error in errors:
            category = error.category.value
            severity = error.severity.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "errors": [self._error_to_dict(error) for error in errors[-10:]]  # Last 10 errors
        }
    
    def _error_to_dict(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Convert ErrorInfo to dictionary for reporting."""
        return {
            "error_id": error_info.error_id,
            "category": error_info.category.value,
            "severity": error_info.severity.value,
            "message": error_info.message,
            "task_id": error_info.task_id,
            "agent_type": error_info.agent_type,
            "timestamp": error_info.timestamp,
            "retry_count": error_info.retry_count,
            "max_retries": error_info.max_retries,
            "recovery_strategy": error_info.recovery_strategy,
            "suggested_action": error_info.suggested_action
        }