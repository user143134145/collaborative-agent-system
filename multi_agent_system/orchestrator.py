"""Multi-Agent Orchestrator for coordinating AI agents."""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import config
from .data_structures import Task, Response, AgentType, TaskStatus, TaskType, TaskPriority
from .specialized_agents import Qwen2_5OrchestratorAgent, ClaudeReasoningAgent, Qwen3CoderAgent, Qwen2_5VisionAgent
from .openrouter_agents import OpenRouterOrchestratorAgent, OpenRouterClaudeReasoningAgent, OpenRouterQwen3CoderAgent, OpenRouterQwen2_5VisionAgent
from .execution_agent import ExecutionAgent
from .memory_system import PersistentMemorySystem
from .logging_config import SystemLogger
from .enhanced_error_handler import EnhancedErrorHandler


@dataclass
class OrchestrationResult:
    """Result of task orchestration."""
    task_id: str
    final_response: Response
    subtask_responses: List[Response]
    execution_time: float
    confidence_score: float
    artifacts: List[str]
    metadata: Dict[str, Any]


class MultiAgentOrchestrator:
    """Main orchestrator that coordinates all agents and manages task execution."""
    
    def __init__(self):
        self.logger = SystemLogger("orchestrator")
        self.agents: Dict[AgentType, Any] = {}
        self.memory_system: Optional[PersistentMemorySystem] = None
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, Task] = {}
        self.agent_states: Dict[AgentType, str] = {}
        self.error_handler = EnhancedErrorHandler()
        
    async def initialize(self) -> bool:
        """Initialize the orchestrator and all components."""
        try:
            self.logger.info("Initializing multi-agent orchestrator")
            
            # Initialize memory system
            self.memory_system = PersistentMemorySystem()
            
            # Initialize agents based on configuration
            if config.USE_OPENROUTER and config.OPENROUTER_API_KEY:
                self.logger.info("Using OpenRouter.ai for agent integration")
                self.agents[AgentType.ORCHESTRATOR] = OpenRouterOrchestratorAgent()
                self.agents[AgentType.PLANNER] = OpenRouterClaudeReasoningAgent()
                self.agents[AgentType.CODER] = OpenRouterQwen3CoderAgent()
                self.agents[AgentType.VISION_ANALYZER] = OpenRouterQwen2_5VisionAgent()
            else:
                self.logger.info("Using individual API providers")
                self.agents[AgentType.ORCHESTRATOR] = Qwen2_5OrchestratorAgent()
                self.agents[AgentType.PLANNER] = ClaudeReasoningAgent()
                self.agents[AgentType.CODER] = Qwen3CoderAgent()
                self.agents[AgentType.VISION_ANALYZER] = Qwen2_5VisionAgent()
            
            self.agents[AgentType.EXECUTION] = ExecutionAgent()
            
            # Initialize agent states
            for agent_type in self.agents.keys():
                self.agent_states[agent_type] = "idle"
            
            self.logger.info("Orchestrator initialized successfully", agent_count=len(self.agents))
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize orchestrator", error=str(e))
            return False
    
    async def get_checkpoints_for_task(self, task_id: str) -> List['Checkpoint']:
        """Get all checkpoints for a specific task."""
        if self.memory_system:
            return await self.memory_system.get_checkpoints_for_task(task_id)
        return []
    
    async def process_task(self, task: Task) -> OrchestrationResult:
        """Process a task through the multi-agent system."""
        start_time = time.time()
        task_id = task.id
        
        self.logger.info("Processing task", task_id=task_id, task_type=task.type.value)
        task.update_status(TaskStatus.IN_PROGRESS)
        self.active_tasks[task_id] = task
        
        async def _process_task_impl():
            # Step 1: Search memory for related knowledge
            related_knowledge = await self._search_memory(task)
            
            # Step 2: Decompose complex tasks using orchestrator agent
            if self._requires_decomposition(task):
                subtasks = await self.agents[AgentType.ORCHESTRATOR].decompose_task(task)
                subtask_responses = await self._execute_subtasks(subtasks, task, related_knowledge)
            else:
                # Direct execution for simple tasks
                agent_type = self._select_agent_for_task(task)
                response = await self._execute_with_agent(task, agent_type, related_knowledge)
                subtask_responses = [response]
            
            # Step 3: Cross-validate results if multiple agents involved
            if len(subtask_responses) > 1:
                validated_response = await self._cross_validate_responses(subtask_responses, task)
            else:
                validated_response = subtask_responses[0]
            
            # Step 4: Store results in memory
            await self._store_knowledge_artifacts(validated_response, task)
            
            # Step 5: Prepare final result
            execution_time = time.time() - start_time
            result = OrchestrationResult(
                task_id=task_id,
                final_response=validated_response,
                subtask_responses=subtask_responses,
                execution_time=execution_time,
                confidence_score=validated_response.confidence_score,
                artifacts=validated_response.artifacts,
                metadata={"related_knowledge_count": len(related_knowledge)}
            )
            
            task.update_status(TaskStatus.COMPLETED)
            self.logger.info("Task completed successfully", task_id=task_id, execution_time=execution_time)
            
            return result
        
        try:
            # Use enhanced error handler
            error_context = {
                "task_id": task_id,
                "task_type": task.type.value,
                "critical_task": task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]
            }
            
            result = await self.error_handler.handle_with_retry(
                _process_task_impl,
                error_context=error_context,
                max_retries=2
            )
            
            return result
            
        except Exception as e:
            task.update_status(TaskStatus.FAILED)
            error_info = self.error_handler.categorize_error(e, {
                "task_id": task_id,
                "task_type": task.type.value
            })
            
            self.logger.error("Task processing failed", 
                            task_id=task_id, 
                            error=str(e),
                            error_category=error_info.category.value,
                            error_severity=error_info.severity.value)
            
            # Create error response
            error_response = Response(
                task_id=task_id,
                agent_type=AgentType.ORCHESTRATOR,
                content=f"Task failed: {str(e)}",
                confidence_score=0.0,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                metadata={
                    "error_category": error_info.category.value,
                    "error_severity": error_info.severity.value,
                    "suggested_action": error_info.suggested_action
                }
            )
            
            return OrchestrationResult(
                task_id=task_id,
                final_response=error_response,
                subtask_responses=[],
                execution_time=time.time() - start_time,
                confidence_score=0.0,
                artifacts=[],
                metadata={
                    "error": str(e),
                    "error_category": error_info.category.value,
                    "error_severity": error_info.severity.value
                }
            )
    
    async def _search_memory(self, task: Task) -> List[Any]:
        """Search memory for related knowledge artifacts."""
        if not self.memory_system:
            return []
        
        try:
            query = f"{task.title} {task.description}"
            results = await self.memory_system.search_artifacts(query, top_k=3)
            return results
        except Exception as e:
            self.logger.warning("Memory search failed", task_id=task.id, error=str(e))
            return []
    
    def _requires_decomposition(self, task: Task) -> bool:
        """Determine if a task requires decomposition into subtasks."""
        # Complex tasks with multiple requirements or high priority need decomposition
        return (len(task.requirements) > 2 or 
                task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL] or
                task.type in [TaskType.RESEARCH, TaskType.ANALYSIS])
    
    def _select_agent_for_task(self, task: Task) -> AgentType:
        """Select the most appropriate agent for a task."""
        agent_mapping = {
            TaskType.RESEARCH: AgentType.PLANNER,
            TaskType.PLANNING: AgentType.PLANNER,
            TaskType.CODING: AgentType.CODER,
            TaskType.ANALYSIS: AgentType.PLANNER,
            TaskType.VISION: AgentType.VISION_ANALYZER,
            TaskType.VALIDATION: AgentType.ORCHESTRATOR,
            TaskType.EXECUTION: AgentType.EXECUTION
        }
        
        # Special case: if the task is about executing code, use ExecutionAgent
        if "execute" in task.description.lower() or "run" in task.description.lower():
            if any(keyword in task.description.lower() for keyword in ["code", "program", "script", "test"]):
                return AgentType.EXECUTION
        
        return agent_mapping.get(task.type, AgentType.PLANNER)
    
    async def _execute_subtasks(self, subtasks: List[Dict[str, Any]], parent_task: Task, context: List[Any]) -> List[Response]:
        """Execute multiple subtasks in sequence or parallel."""
        responses = []
        
        for subtask_data in subtasks:
            # Create subtask from decomposition data
            subtask = Task(
                type=TaskType(subtask_data.get("type", "analysis")),
                priority=TaskPriority(subtask_data.get("priority", "medium")),
                title=subtask_data.get("title", ""),
                description=subtask_data.get("description", ""),
                requirements=subtask_data.get("requirements", []),
                parent_task_id=parent_task.id,
                context={
                    "parent_task": parent_task.title,
                    "related_knowledge": context
                }
            )
            
            # Select agent for subtask
            agent_type_str = subtask_data.get("assigned_agent", "planner")
            agent_type_mapping = {
                "orchestrator": AgentType.ORCHESTRATOR,
                "planner": AgentType.PLANNER,
                "coder": AgentType.CODER,
                "vision_analyzer": AgentType.VISION_ANALYZER,
                "execution": AgentType.EXECUTION
            }
            agent_type = agent_type_mapping.get(agent_type_str, AgentType.PLANNER)
            
            # Execute subtask
            response = await self._execute_with_agent(subtask, agent_type, context)
            responses.append(response)
        
        return responses
    
    async def _execute_with_agent(self, task: Task, agent_type: AgentType, context: List[Any]) -> Response:
        """Execute a task with a specific agent."""
        if agent_type not in self.agents:
            raise ValueError(f"Agent type {agent_type} not available")
        
        agent = self.agents[agent_type]
        self.agent_states[agent_type] = "busy"
        
        try:
            # Prepare context for the agent
            context_dict = {}
            if context:
                context_dict["related_knowledge"] = [str(art[0]) for art in context[:2]]  # Limit context
            
            response = await agent.process_task(task, context_dict)
            return response
            
        finally:
            self.agent_states[agent_type] = "idle"
    
    async def _cross_validate_responses(self, responses: List[Response], original_task: Task) -> Response:
        """Cross-validate multiple agent responses using the CrossValidator."""
        from cross_validation import cross_validator, ValidationStrategy
        
        if not responses:
            raise ValueError("No responses to validate")
        
        # Use confidence-weighted validation strategy
        validation_result = await cross_validator.validate_responses(
            responses, original_task, ValidationStrategy.CONFIDENCE_WEIGHTED
        )
        
        if validation_result.final_response:
            return validation_result.final_response
        else:
            # Fallback: return highest confidence response
            return max(responses, key=lambda r: r.confidence_score)
    
    async def _store_knowledge_artifacts(self, response: Response, task: Task) -> None:
        """Store valuable knowledge from the response in memory."""
        if not self.memory_system or not response.success:
            return
        
        try:
            from data_structures import KnowledgeArtifact
            
            # Only store high-confidence responses
            if response.confidence_score >= 0.7:
                artifact = KnowledgeArtifact(
                    title=f"Knowledge from: {task.title}",
                    content=response.content,
                    artifact_type=task.type.value,
                    tags=[task.type.value, response.agent_type.value],
                    source_task_id=task.id,
                    source_agent=response.agent_type,
                    relevance_score=response.confidence_score
                )
                
                await self.memory_system.store_artifact(artifact)
                self.logger.info("Knowledge artifact stored", artifact_id=artifact.id)
                
        except Exception as e:
            self.logger.warning("Failed to store knowledge artifact", error=str(e))
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        return {
            "active_tasks": len(self.active_tasks),
            "agent_states": self.agent_states,
            "memory_system_ready": self.memory_system is not None,
            "agents_available": list(self.agents.keys())
        }
    
    async def shutdown(self) -> None:
        """Clean shutdown of the orchestrator and all components."""
        self.logger.info("Shutting down orchestrator")
        
        # Cleanup all agents
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        # Clear active tasks
        self.active_tasks.clear()
        
        self.logger.info("Orchestrator shutdown complete")