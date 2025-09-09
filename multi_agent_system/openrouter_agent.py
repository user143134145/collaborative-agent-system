"""OpenRouter.ai agent implementation for Multi-Agent AI System."""

import asyncio
import json
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .data_structures import AgentType, Response, Task
from .logging_config import SystemLogger
from .enhanced_error_handler import EnhancedErrorHandler


class OpenRouterAgent:
    """Base agent for OpenRouter.ai integration."""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.logger = SystemLogger(f"agent.{agent_type.value}")
        self.openrouter_client: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(config.RATE_LIMIT_REQUESTS_PER_MINUTE)
        self.error_handler = EnhancedErrorHandler()
        self._setup_openrouter_client()
    
    def _setup_openrouter_client(self) -> None:
        """Setup OpenRouter client session."""
        if config.USE_OPENROUTER and config.OPENROUTER_API_KEY:
            # Defer session creation until needed
            pass
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        return {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/your-repo/multi-agent-system",
            "X-Title": "Multi-Agent AI System",
            "Content-Type": "application/json"
        }
    
    def _get_model_name(self) -> str:
        """Get the model name used by this agent."""
        model_mapping = {
            AgentType.ORCHESTRATOR: config.OPENROUTER_QWEN_ORCHESTRATOR_MODEL,
            AgentType.PLANNER: config.OPENROUTER_CLAUDE_MODEL,
            AgentType.CODER: config.OPENROUTER_QWEN_CODER_MODEL,
            AgentType.VISION_ANALYZER: config.OPENROUTER_QWEN_VISION_MODEL
        }
        return model_mapping.get(self.agent_type, config.OPENROUTER_QWEN_ORCHESTRATOR_MODEL)
    
    async def _ensure_client(self) -> None:
        """Ensure OpenRouter client is created."""
        if not self.openrouter_client:
            timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)
            self.openrouter_client = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._get_default_headers()
            )
    
    async def _make_api_call(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Make API call to OpenRouter.ai."""
        await self._ensure_client()
        return await self._make_openrouter_api_call(prompt, **kwargs)
    
    async def _make_openrouter_api_call(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> Dict[str, Any]:
        """Make API call to OpenRouter.ai."""
        if not self.openrouter_client:
            await self._ensure_client()
        
        if not self.openrouter_client:
            raise Exception("OpenRouter client not initialized")
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Get model name
        model_name = self._get_model_name()
        
        # Prepare payload
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,  # Default temperature
            "max_tokens": 4096,  # Default max tokens
        }
        
        # Apply agent-specific parameters
        if self.agent_type == AgentType.ORCHESTRATOR:
            payload["temperature"] = 0.3
            payload["max_tokens"] = 4000
        elif self.agent_type == AgentType.PLANNER:
            payload["temperature"] = 0.1
            payload["max_tokens"] = config.CLAUDE_MAX_TOKENS
        elif self.agent_type == AgentType.CODER:
            payload["temperature"] = 0.1
            payload["max_tokens"] = 4000
        elif self.agent_type == AgentType.VISION_ANALYZER:
            payload["temperature"] = 0.2
            payload["max_tokens"] = 2000
        
        # Add any additional parameters from kwargs
        for key, value in kwargs.items():
            payload[key] = value
        
        async with self.openrouter_client.post(
            f"{config.OPENROUTER_API_BASE}/chat/completions",
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse API response to extract content."""
        try:
            if "choices" in api_response:
                return api_response["choices"][0]["message"]["content"]
            elif "data" in api_response:
                return api_response["data"]["choices"][0]["message"]["content"]
            else:
                return str(api_response)
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse OpenRouter response: {e}")
    
    @abstractmethod
    async def process_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Process a task and return a response."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup OpenRouter client resources."""
        if self.openrouter_client:
            await self.openrouter_client.close()
            self.openrouter_client = None
    
    def __del__(self):
        """Cleanup on deletion."""
        # Check if the attribute exists to avoid AttributeError
        if hasattr(self, 'openrouter_client') and self.openrouter_client:
            try:
                import asyncio
                # Only create task if there's a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        asyncio.create_task(self.openrouter_client.close())
                except RuntimeError:
                    # No running event loop, close synchronously
                    pass
            except Exception:
                # Ignore cleanup errors during deletion
                pass