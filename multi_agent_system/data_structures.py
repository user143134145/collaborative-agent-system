"""Data structures for Multi-Agent AI System."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import uuid


class TaskType(str, Enum):
    """Types of tasks that can be processed."""
    RESEARCH = "research"
    CODING = "coding"
    ANALYSIS = "analysis"
    VISION = "vision"
    PLANNING = "planning"
    VALIDATION = "validation"
    EXECUTION = "execution"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    CODER = "coder"
    VISION_ANALYZER = "vision_analyzer"
    EXECUTION = "execution"


class CheckpointType(str, Enum):
    """Types of checkpoints in the autonomous coding process."""
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    PROJECT_STRUCTURE = "project_structure"
    MAIN_CODE_GENERATION = "main_code_generation"
    ADDITIONAL_FILES = "additional_files"
    TEST_GENERATION = "test_generation"
    CODE_VALIDATION = "code_validation"
    CODE_FORMATTING = "code_formatting"
    APPLICATION_EXECUTION = "application_execution"
    FINAL_RESULT = "final_result"


class Task(BaseModel):
    """Represents a task to be processed by the system."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    title: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = Field(default_factory=list)
    assigned_agent: Optional[AgentType] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_status(self, status: TaskStatus) -> None:
        """Update task status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()


class Response(BaseModel):
    """Represents a response from an agent."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    agent_type: AgentType
    content: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    execution_time: float  # in seconds
    tokens_used: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)  # File paths or URLs
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_artifact(self, artifact_path: str) -> None:
        """Add an artifact to the response."""
        if artifact_path not in self.artifacts:
            self.artifacts.append(artifact_path)


class KnowledgeArtifact(BaseModel):
    """Represents a piece of knowledge stored in the memory system."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    artifact_type: str  # "code", "research", "analysis", "documentation"
    tags: List[str] = Field(default_factory=list)
    source_task_id: Optional[str] = None
    source_agent: Optional[AgentType] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    relevance_score: float = 0.0
    
    def increment_access(self) -> None:
        """Increment access count and update timestamp."""
        self.access_count += 1
        self.updated_at = datetime.utcnow()


class AgentState(BaseModel):
    """Represents the current state of an agent."""
    
    agent_type: AgentType
    status: str  # "idle", "busy", "error", "offline"
    current_task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    error_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossValidationResult(BaseModel):
    """Results from cross-validation between multiple agents."""
    
    task_id: str
    responses: List[Response]
    consensus_score: float = Field(ge=0.0, le=1.0)
    final_response: Optional[Response] = None
    discrepancies: List[str] = Field(default_factory=list)
    confidence_scores: List[float] = Field(default_factory=list)
    validation_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MultiModalWorkflowResult(BaseModel):
    """Results from multi-modal consistency workflow."""
    
    task_id: str
    visual_analysis: Optional[Response] = None
    text_analysis: Optional[Response] = None
    consistency_score: float = Field(ge=0.0, le=1.0)
    inconsistencies: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SystemMetrics(BaseModel):
    """System performance and health metrics."""
    
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    active_agents: int = 0
    memory_usage_mb: float = 0.0
    vector_db_size: int = 0
    uptime_seconds: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ExecutionPipeline(BaseModel):
    """Represents the execution pipeline state."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    root_task_id: str
    pipeline_tasks: List[str] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    current_stage: str = "initialization"
    progress_percentage: float = 0.0
    estimated_completion: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Checkpoint(BaseModel):
    """Represents a checkpoint in the autonomous coding process."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    checkpoint_type: CheckpointType
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def mark_completed(self, execution_time: float = 0.0) -> None:
        """Mark checkpoint as completed successfully."""
        self.success = True
        self.execution_time = execution_time
        self.timestamp = datetime.utcnow()
        
    def mark_failed(self, error_message: str, execution_time: float = 0.0) -> None:
        """Mark checkpoint as failed."""
        self.success = False
        self.error_message = error_message
        self.execution_time = execution_time
        self.timestamp = datetime.utcnow()


class AutonomousAppCheckpointData(BaseModel):
    """Data structure for autonomous application generation checkpoints."""
    
    project_info: Optional[Dict[str, Any]] = None
    app_structure: Optional[Dict[str, Any]] = None
    main_code: Optional[str] = None
    additional_files: Optional[Dict[str, str]] = None
    test_files: Optional[Dict[str, str]] = None
    validation_result: Optional[Dict[str, Any]] = None
    formatted_code: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    generated_files: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    issues: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    confidence: Optional[float] = None