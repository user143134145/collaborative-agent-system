"""OpenRouter-specific agent implementations for Multi-Agent AI System."""

from typing import Any, Dict, List, Optional
import time
import json

from .openrouter_agent import OpenRouterAgent
from .config import config
from .data_structures import AgentType, Task, Response


class OpenRouterOrchestratorAgent(OpenRouterAgent):
    """Orchestrator agent using OpenRouter for task coordination."""
    
    def __init__(self):
        super().__init__(AgentType.ORCHESTRATOR)
        self.context_window = []
        self.context_tokens = 0
        self.max_context_tokens = config.MAX_CONTEXT_TOKENS
    
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
    
    async def process_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Process a task and return a response."""
        start_time = time.time()
        
        try:
            # Prepare prompt with context
            prompt = self._prepare_prompt(task, context or {})
            
            # Make API call
            api_response = await self._make_api_call(prompt)
            
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
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            return Response(
                task_id=task.id,
                agent_type=self.agent_type,
                content="",
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
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


class OpenRouterClaudeReasoningAgent(OpenRouterAgent):
    """Reasoning and planning agent using Claude via OpenRouter."""
    
    def __init__(self):
        super().__init__(AgentType.PLANNER)
    
    async def process_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Process a task and return a response."""
        start_time = time.time()
        
        try:
            # Prepare prompt with context
            prompt = self._prepare_prompt(task, context or {})
            
            # Make API call with system prompt
            api_response = await self._make_api_call(
                prompt,
                system_prompt=self._get_reasoning_system_prompt()
            )
            
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
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            return Response(
                task_id=task.id,
                agent_type=self.agent_type,
                content="",
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
    def _get_reasoning_system_prompt(self) -> str:
        """Get system prompt for reasoning agent."""
        return """You are an expert reasoning and planning agent. Your role is to:

1. ANALYZE complex problems and break them down into logical components
2. DEVELOP comprehensive plans and strategies for task execution
3. IDENTIFY potential challenges, risks, and mitigation strategies
4. PROVIDE clear, step-by-step reasoning for your recommendations
5. VALIDATE logical consistency and feasibility of proposed approaches

Focus on:
- Clear, structured thinking with quantitative backing
- Evidence-based reasoning supported by data analysis
- Practical, actionable recommendations with numerical justification
- Risk assessment and mitigation with statistical confidence
- Quality assurance considerations with data validation

Always explain your reasoning process and provide confidence assessments for your recommendations."""
    
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


class OpenRouterQwen3CoderAgent(OpenRouterAgent):
    """Coding agent using Qwen3 Coder via OpenRouter."""
    
    def __init__(self):
        super().__init__(AgentType.CODER)
    
    async def process_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Process a task and return a response."""
        start_time = time.time()
        
        try:
            # Prepare prompt with context
            prompt = self._prepare_prompt(task, context or {})
            
            # Make API call with system prompt
            api_response = await self._make_api_call(
                prompt,
                system_prompt=self._get_coder_system_prompt()
            )
            
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
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            return Response(
                task_id=task.id,
                agent_type=self.agent_type,
                content="",
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
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


class OpenRouterQwen2_5VisionAgent(OpenRouterAgent):
    """Vision analysis agent using Qwen2.5-VL via OpenRouter."""
    
    def __init__(self):
        super().__init__(AgentType.VISION_ANALYZER)
    
    async def process_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Response:
        """Process a task and return a response."""
        start_time = time.time()
        
        try:
            # Prepare prompt with context
            prompt = self._prepare_prompt(task, context or {})
            
            # Make API call with system prompt
            api_response = await self._make_api_call(
                prompt,
                system_prompt=self._get_vision_system_prompt()
            )
            
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
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            return Response(
                task_id=task.id,
                agent_type=self.agent_type,
                content="",
                confidence_score=0.0,
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
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