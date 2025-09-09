"""Specialized agent implementations for the Multi-Agent AI System."""

import asyncio
import json
import time
import os
import tempfile
import subprocess
from typing import Any, Dict, List, Optional

import aiohttp
import docker
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_agent import BaseAgent
from .config import config
from .data_structures import AgentType, Response, Task, TaskType
from .logging_config import SystemLogger
from .lstm_toolkit import LSTMToolkit


class Qwen2_5OrchestratorAgent(BaseAgent):
    """Orchestrator agent using Qwen2.5-1M for task coordination and management."""
    
    def __init__(self):
        super().__init__(AgentType.ORCHESTRATOR)
        self.context_window = []
        self.context_tokens = 0
        self.max_context_tokens = config.MAX_CONTEXT_TOKENS
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {config.QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def _get_model_name(self) -> str:
        return config.QWEN_ORCHESTRATOR_MODEL
    
    async def _make_api_call(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Make API call to Qwen2.5-1M model."""
        payload = {
            "model": config.QWEN_ORCHESTRATOR_MODEL,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": self._get_orchestrator_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "max_tokens": 4000,
                "temperature": 0.3,
                "top_p": 0.8
            }
        }
        
        async with self.session.post(
            f"{config.QWEN_API_BASE}/chat/completions",
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"Qwen API error: {response.status} - {error_text}")
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse Qwen API response."""
        try:
            return api_response["output"]["choices"][0]["message"]["content"]
        except KeyError as e:
            raise Exception(f"Failed to parse Qwen response: {e}")
    
    def _get_orchestrator_system_prompt(self) -> str:
        """Get system prompt for orchestrator agent."""
        return """You are an intelligent orchestrator agent responsible for coordinating multiple AI agents to complete complex tasks. Your responsibilities include:

1. TASK ANALYSIS: Analyze incoming tasks and determine their complexity, requirements, and optimal approach
2. TASK DECOMPOSITION: Break complex tasks into manageable subtasks that can be assigned to specialized agents
3. AGENT ROUTING: Determine which agents are best suited for each subtask based on their capabilities
4. CONTEXT MANAGEMENT: Maintain relevant context across multiple interactions while managing token limits
5. RESULT SYNTHESIS: Combine outputs from multiple agents into coherent, comprehensive results
6. QUALITY ASSURANCE: Ensure outputs meet quality standards and requirements

Available agents:
- Claude Reasoning Agent: Excellent for planning, analysis, and logical reasoning
- Qwen3 Coder Agent: Specialized in code generation, review, and optimization
- Qwen2.5-VL Vision Agent: Handles image analysis and visual content processing

Always provide structured responses with clear reasoning for your decisions."""
    
    async def decompose_task(self, task: Task) -> List[Dict[str, Any]]:
        """Decompose a complex task into subtasks."""
        decomposition_prompt = f"""
        Analyze the following task and decompose it into subtasks:
        
        Task: {task.title}
        Description: {task.description}
        Requirements: {', '.join(task.requirements)}
        Type: {task.type.value}
        Priority: {task.priority.value}
        
        Please provide a JSON response with the following structure:
        {{
            "subtasks": [
                {{
                    "title": "Subtask title",
                    "description": "Detailed description",
                    "type": "research|coding|analysis|vision|planning",
                    "assigned_agent": "orchestrator|planner|coder|vision_analyzer",
                    "priority": "low|medium|high|critical",
                    "dependencies": ["subtask_id1", "subtask_id2"],
                    "estimated_time": "time in minutes"
                }}
            ],
            "execution_strategy": "sequential|parallel|hybrid",
            "reasoning": "Explanation of decomposition approach"
        }}
        """
        
        try:
            api_response = await self._make_api_call(decomposition_prompt)
            content = self._parse_response(api_response)
            
            # Parse JSON response
            decomposition = json.loads(content)
            return decomposition.get("subtasks", [])
            
        except Exception as e:
            self.logger.error("Failed to decompose task", task_id=task.id, error=str(e))
            # Fallback: create a single subtask
            return [{
                "title": task.title,
                "description": task.description,
                "type": task.type.value,
                "assigned_agent": "planner",
                "priority": task.priority.value,
                "dependencies": [],
                "estimated_time": "30"
            }]
    
    def _manage_context_window(self, new_content: str) -> None:
        """Manage context window to stay within token limits."""
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        new_tokens = len(new_content) // 4
        
        # Add new content
        self.context_window.append(new_content)
        self.context_tokens += new_tokens
        
        # Remove old content if exceeding limits
        while self.context_tokens > self.max_context_tokens * 0.8:  # Keep 20% buffer
            if self.context_window:
                removed_content = self.context_window.pop(0)
                self.context_tokens -= len(removed_content) // 4
            else:
                break
    
    async def synthesize_results(self, subtask_responses: List[Response]) -> str:
        """Synthesize results from multiple agent responses."""
        synthesis_prompt = f"""
        Synthesize the following agent responses into a comprehensive result:
        
        {chr(10).join([f"Agent: {r.agent_type.value}, Confidence: {r.confidence_score:.2f}, Content: {r.content[:500]}..." for r in subtask_responses])}
        
        Please provide a coherent synthesis that:
        1. Combines all relevant information
        2. Resolves any conflicts or inconsistencies
        3. Highlights key findings and recommendations
        4. Maintains the overall context and objectives
        """
        
        try:
            api_response = await self._make_api_call(synthesis_prompt)
            return self._parse_response(api_response)
        except Exception as e:
            self.logger.error("Failed to synthesize results", error=str(e))
            # Fallback: simple concatenation
            return "\n\n".join([f"**{r.agent_type.value.title()} Response:**\n{r.content}" for r in subtask_responses])


class ClaudeReasoningAgent(BaseAgent):
    """Reasoning and planning agent using Claude with LSTM Toolkit integration."""
    
    def __init__(self):
        super().__init__(AgentType.PLANNER)
        self.anthropic_client = None
        self.lstm_toolkit = LSTMToolkit()
        self._setup_anthropic_client()
    
    def _setup_anthropic_client(self):
        """Setup Anthropic client."""
        try:
            import anthropic
            self.anthropic_client = anthropic.AsyncAnthropic(
                api_key=config.ANTHROPIC_API_KEY
            )
        except ImportError:
            self.logger.error("Anthropic library not installed")
            raise
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": config.ANTHROPIC_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    def _get_model_name(self) -> str:
        return config.CLAUDE_MODEL
    
    async def _make_api_call(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Make API call to Claude with optional LSTM Toolkit integration."""
        try:
            # Check if task contains numerical data that could benefit from LSTM analysis
            task = kwargs.get('task')
            lstm_analysis_result = None
            
            if task and self._should_use_lstm_toolkit(task):
                try:
                    lstm_analysis_result = await self._perform_lstm_analysis(task)
                    # Enhance prompt with LSTM analysis context
                    enhanced_prompt = self._enhance_prompt_with_lstm(prompt, lstm_analysis_result)
                    prompt = enhanced_prompt
                except Exception as e:
                    self.logger.warning("LSTM analysis failed, proceeding without it",
                                       task_id=task.id if task else "unknown", error=str(e))
            
            message = await self.anthropic_client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                temperature=0.1,
                system=self._get_reasoning_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response_content = message.content[0].text
            
            # If we have LSTM results, incorporate them into the final response
            if lstm_analysis_result:
                response_content = self._integrate_lstm_results(response_content, lstm_analysis_result)
            
            return {
                "content": response_content,
                "usage": {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                    "total_tokens": message.usage.input_tokens + message.usage.output_tokens
                },
                "model": message.model,
                "lstm_analysis_used": lstm_analysis_result is not None
            }
            
        except Exception as e:
            self.logger.error("Claude API call failed", error=str(e))
            raise

    def _should_use_lstm_toolkit(self, task: Any) -> bool:
        """Determine if LSTM Toolkit should be used for this task."""
        task_content = f"{task.title} {task.description} {' '.join(task.requirements)}"
        
        # Keywords that indicate numerical/economic data analysis
        numerical_keywords = [
            'data', 'numbers', 'statistics', 'economic', 'financial', 'market',
            'trend', 'forecast', 'prediction', 'time series', 'analysis',
            'stock', 'price', 'revenue', 'profit', 'growth', 'rate',
            'percentage', 'metric', 'indicator', 'dataset', 'values',
            'numerical', 'quantitative', 'metrics', 'kpi', 'performance'
        ]
        
        # Check if task contains numerical keywords
        contains_numerical_content = any(
            keyword.lower() in task_content.lower() for keyword in numerical_keywords
        )
        
        # Additional check for explicit numerical data in requirements
        has_explicit_data = any(
            'data:' in req.lower() or 'dataset:' in req.lower() or 'numbers:' in req.lower()
            for req in task.requirements
        )
        
        return contains_numerical_content or has_explicit_data

    async def _perform_lstm_analysis(self, task: Any) -> Dict[str, Any]:
        """Perform LSTM analysis on task data."""
        analysis_results = {}
        
        try:
            # Extract numerical data from task if present
            numerical_data = self._extract_numerical_data(task)
            
            if numerical_data:
                # Perform time series analysis
                if len(numerical_data) >= 10:  # Minimum data points for meaningful analysis
                    time_series_result = self.lstm_toolkit.analyze_time_series(numerical_data)
                    analysis_results['time_series'] = time_series_result
                
                # Perform economic analysis if applicable
                economic_result = self.lstm_toolkit.economic_analysis(numerical_data)
                analysis_results['economic'] = economic_result
                
                # Generate visualization if data is sufficient and matplotlib is available
                if len(numerical_data) >= 5:
                    try:
                        viz_path = self.lstm_toolkit.visualize_data(numerical_data, f"task_{task.id}")
                        analysis_results['visualization'] = viz_path
                    except Exception as e:
                        self.logger.warning("Visualization failed", error=str(e))
                        analysis_results['visualization'] = "Visualization unavailable"
            
            return analysis_results
            
        except Exception as e:
            self.logger.error("LSTM analysis failed", task_id=task.id, error=str(e))
            return {"error": str(e)}

    def _extract_numerical_data(self, task: Any) -> List[float]:
        """Extract numerical data from task content."""
        numerical_data = []
        
        # Check description for numerical patterns
        import re
        number_pattern = r'\b\d+\.?\d*\b'
        
        # Extract from description
        desc_numbers = re.findall(number_pattern, task.description)
        numerical_data.extend([float(num) for num in desc_numbers])
        
        # Extract from requirements
        for requirement in task.requirements:
            req_numbers = re.findall(number_pattern, requirement)
            numerical_data.extend([float(num) for num in req_numbers])
        
        # Check context for data arrays
        if task.context and 'data' in task.context:
            context_data = task.context.get('data', [])
            if isinstance(context_data, list) and all(isinstance(x, (int, float)) for x in context_data):
                numerical_data.extend(context_data)
        
        return numerical_data

    def _enhance_prompt_with_lstm(self, original_prompt: str, lstm_results: Dict[str, Any]) -> str:
        """Enhance the prompt with LSTM analysis results."""
        enhanced_prompt = f"""{original_prompt}

LSTM ANALYSIS RESULTS:
"""
        
        if 'time_series' in lstm_results:
            ts = lstm_results['time_series']
            enhanced_prompt += f"""
Time Series Analysis:
- Trend: {ts.get('trend', 'N/A')}
- Seasonality: {ts.get('seasonality', 'N/A')}
- Forecast: {ts.get('forecast', 'N/A')}
- Confidence: {ts.get('confidence', 'N/A')}
"""
        
        if 'economic' in lstm_results:
            econ = lstm_results['economic']
            enhanced_prompt += f"""
Economic Analysis:
- Growth Rate: {econ.get('growth_rate', 'N/A')}
- Volatility: {econ.get('volatility', 'N/A')}
- Pattern Type: {econ.get('pattern_type', 'N/A')}
- Risk Assessment: {econ.get('risk_assessment', 'N/A')}
"""
        
        if 'visualization' in lstm_results:
            enhanced_prompt += f"\nData visualization available at: {lstm_results['visualization']}"
        
        enhanced_prompt += """

Please incorporate these LSTM analysis results into your reasoning and recommendations. Focus on:
1. Interpreting the numerical patterns and trends
2. Providing data-driven insights and recommendations
3. Considering the forecasted outcomes in your planning
4. Addressing any risks or opportunities identified in the analysis
"""
        
        return enhanced_prompt

    def _integrate_lstm_results(self, claude_response: str, lstm_results: Dict[str, Any]) -> str:
        """Integrate LSTM results into Claude's response."""
        integration_note = """

--- LSTM ANALYSIS INTEGRATION ---
The above analysis has been enhanced with LSTM neural network processing, providing:
"""
        
        if 'time_series' in lstm_results:
            integration_note += "- Time series forecasting and pattern recognition\n"
        
        if 'economic' in lstm_results:
            integration_note += "- Economic trend analysis and risk assessment\n"
        
        if 'visualization' in lstm_results:
            integration_note += f"- Data visualization: {lstm_results['visualization']}\n"
        
        integration_note += "--------------------------------"
        
        return claude_response + integration_note
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse Claude API response."""
        return api_response.get("content", "")
    
    def _get_reasoning_system_prompt(self) -> str:
        """Get system prompt for reasoning agent with LSTM Toolkit integration."""
        return """You are an expert reasoning and planning agent with advanced numerical analysis capabilities. Your role is to:

1. ANALYZE complex problems and break them down into logical components
2. DEVELOP comprehensive plans and strategies for task execution
3. IDENTIFY potential challenges, risks, and mitigation strategies
4. PROVIDE clear, step-by-step reasoning for your recommendations
5. VALIDATE logical consistency and feasibility of proposed approaches
6. UTILIZE LSTM Toolkit for numerical and time series analysis when appropriate

SPECIAL CAPABILITIES:
- LSTM Toolkit Integration: You have access to an LSTM (Long Short-Term Memory) neural network toolkit for:
  * Time series forecasting and analysis
  * Economic data pattern recognition
  * Numerical data trend detection
  * Predictive modeling and forecasting
  * Statistical analysis and visualization

When you encounter numerical data, time series, economic indicators, or quantitative analysis tasks:
1. First analyze if LSTM analysis would be beneficial
2. If appropriate, use the LSTM Toolkit to process the data
3. Incorporate the LSTM analysis results into your reasoning
4. Provide visualizations and forecasts when relevant

Focus on:
- Clear, structured thinking with quantitative backing
- Evidence-based reasoning supported by data analysis
- Practical, actionable recommendations with numerical justification
- Risk assessment and mitigation with statistical confidence
- Quality assurance considerations with data validation

Always explain your reasoning process, include LSTM analysis results when applicable, and provide confidence assessments for your recommendations."""


class Qwen3CoderAgent(BaseAgent):
    """Coding agent using Qwen3 Coder."""
    
    def __init__(self):
        super().__init__(AgentType.CODER)
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {config.QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def _get_model_name(self) -> str:
        return config.QWEN_CODER_MODEL
    
    async def _make_api_call(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Make API call to Qwen3 Coder."""
        payload = {
            "model": config.QWEN_CODER_MODEL,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": self._get_coder_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        async with self.session.post(
            f"{config.QWEN_API_BASE}/chat/completions",
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"Qwen Coder API error: {response.status} - {error_text}")
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse Qwen Coder API response."""
        try:
            return api_response["output"]["choices"][0]["message"]["content"]
        except KeyError as e:
            raise Exception(f"Failed to parse Qwen Coder response: {e}")
    
    def _get_coder_system_prompt(self) -> str:
        """Get system prompt for coder agent."""
        return """You are an expert software engineer and coding assistant. Your responsibilities include:

1. CODE GENERATION: Write clean, efficient, and well-documented code
2. CODE REVIEW: Analyze code for bugs, performance issues, and best practices
3. TESTING: Generate comprehensive test suites and test cases
4. OPTIMIZATION: Improve code performance and maintainability
5. DOCUMENTATION: Create clear technical documentation and comments

Best practices to follow:
- Write modular, reusable code
- Follow language-specific conventions and style guides
- Include comprehensive error handling
- Add meaningful comments and docstrings
- Consider security and performance implications
- Generate appropriate test cases

Always provide complete, working code solutions with explanations."""


class Qwen2_5VisionAgent(BaseAgent):
    """Vision analysis agent using Qwen2.5-VL."""
    
    def __init__(self):
        super().__init__(AgentType.VISION_ANALYZER)
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {config.QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def _get_model_name(self) -> str:
        return config.QWEN_VISION_MODEL
    
    async def _make_api_call(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Make API call to Qwen2.5-VL."""
        # Handle image data if provided
        messages = [
            {
                "role": "system",
                "content": self._get_vision_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Add image if provided in kwargs
        if "image_data" in kwargs:
            messages[-1]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": kwargs["image_data"]}}
            ]
        
        payload = {
            "model": config.QWEN_VISION_MODEL,
            "input": {"messages": messages},
            "parameters": {
                "max_tokens": 2000,
                "temperature": 0.2
            }
        }
        
        async with self.session.post(
            f"{config.QWEN_API_BASE}/chat/completions",
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"Qwen Vision API error: {response.status} - {error_text}")
    
    def _parse_response(self, api_response: Dict[str, Any]) -> str:
        """Parse Qwen Vision API response."""
        try:
            return api_response["output"]["choices"][0]["message"]["content"]
        except KeyError as e:
            raise Exception(f"Failed to parse Qwen Vision response: {e}")
    
    def _get_vision_system_prompt(self) -> str:
        """Get system prompt for vision agent."""
        return """You are an expert computer vision and image analysis agent. Your capabilities include:

1. IMAGE ANALYSIS: Detailed analysis of visual content, objects, scenes, and patterns
2. TEXT EXTRACTION: OCR and text recognition from images
3. VISUAL CONSISTENCY: Checking consistency between visual and textual information
4. PATTERN DETECTION: Identifying visual patterns, anomalies, and trends
5. QUALITY ASSESSMENT: Evaluating image quality, composition, and technical aspects

Analysis approach:
- Provide detailed, structured descriptions
- Identify key visual elements and their relationships
- Extract and interpret any text content
- Assess visual quality and technical aspects
- Compare with provided context or requirements
- Highlight any inconsistencies or notable features

Always provide confidence scores for your visual interpretations."""