"""Base agent class for Multi-Agent AI System."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .data_structures import AgentType, Response, Task
from .logging_config import SystemLogger
from .enhanced_error_handler import EnhancedErrorHandler


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.logger = SystemLogger(f"agent.{agent_type.value}")
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(config.RATE_LIMIT_REQUESTS_PER_MINUTE)
        self.error_handler = EnhancedErrorHandler()
        self._setup_session()
    
    def _setup_session(self) -> None:
        """Setup HTTP session with appropriate headers."""
        timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self._get_default_headers()
        )
    
    @abstractmethod
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        pass
    
    @abstractmethod
    async def _make_api_call(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Make API call to the agent's model."""
        pass
    
    @abstractmethod
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse API response to extract content."""
        pass
    
    async def process_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Process a task and return a response."""
        start_time = time.time()
        
        self.logger.log_task_start(
            task_id=task.id,
            agent_type=self.agent_type.value,
            task_type=task.type.value
        )
        
        async def _process_task_impl():
            # Prepare prompt with context
            prompt = self._prepare_prompt(task, context or {})
            
            # Make API call with rate limiting
            async with self.rate_limiter:
                api_response = await self._make_api_call(prompt, task=task)
            
            # Parse response
            content = self._parse_response(api_response)
            confidence_score = self._calculate_confidence(api_response, task)
            
            # Create response object
            execution_time = time.time() - start_time
            response = Response(
                task_id=task.id,
                agent_type=self.agent_type,
                content=content,
                confidence_score=confidence_score,
                execution_time=execution_time,
                tokens_used=api_response.get('usage', {}).get('total_tokens'),
                success=True,
                metadata={
                    'model_used': self._get_model_name(),
                    'api_response_metadata': api_response.get('metadata', {})
                }
            )
            
            self.logger.log_task_completion(
                task_id=task.id,
                agent_type=self.agent_type.value,
                execution_time=execution_time,
                success=True
            )
            
            return response
        
        try:
            # Use enhanced error handler with retry mechanism
            error_context = {
                "task_id": task.id,
                "agent_type": self.agent_type.value,
                "task_type": task.type.value
            }
            
            response = await self.error_handler.handle_with_retry(
                _process_task_impl, 
                error_context=error_context
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            # Categorize and log the error
            error_info = self.error_handler.categorize_error(e, {
                "task_id": task.id,
                "agent_type": self.agent_type.value,
                "task_type": task.type.value
            })
            
            self.logger.error(
                "Task processing failed",
                task_id=task.id,
                agent_type=self.agent_type.value,
                error=error_message,
                execution_time=execution_time,
                error_category=error_info.category.value,
                error_severity=error_info.severity.value
            )
            
            return Response(
                task_id=task.id,
                agent_type=self.agent_type,
                content="",
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=error_message,
                metadata={
                    "error_category": error_info.category.value,
                    "error_severity": error_info.severity.value,
                    "suggested_action": error_info.suggested_action
                }
            )
    
    def _prepare_prompt(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare prompt for the API call."""
        prompt_parts = [
            f"Task Type: {task.type.value}",
            f"Title: {task.title}",
            f"Description: {task.description}"
        ]
        
        if task.requirements:
            prompt_parts.append(f"Requirements: {', '.join(task.requirements)}")
        
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            prompt_parts.append(f"Context:\n{context_str}")
        
        return "\n\n".join(prompt_parts)
    
    def _calculate_confidence(self, api_response: Dict[str, Any], task: Task) -> float:
        """Calculate confidence score based on API response and task complexity."""
        # Base confidence from API response if available
        base_confidence = api_response.get('confidence', 0.8)
        
        # Adjust based on task complexity
        complexity_factor = 1.0
        if len(task.requirements) > 5:
            complexity_factor *= 0.9
        if task.priority.value in ['high', 'critical']:
            complexity_factor *= 0.95
        
        return min(base_confidence * complexity_factor, 1.0)
    
    @abstractmethod
    def _get_model_name(self) -> str:
        """Get the model name used by this agent."""
        pass
    
    async def health_check(self) -> bool:
        """Check if the agent is healthy and can make API calls."""
        try:
            # Simple test call
            test_task = Task(
                type=TaskType.ANALYSIS,
                title="Health Check",
                description="Simple health check task"
            )
            response = await self.process_task(test_task)
            return response.success
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.session and not self.session.closed:
            asyncio.create_task(self.session.close())