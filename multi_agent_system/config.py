"""Configuration management for Multi-Agent AI System."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from the correct location
# Try multiple locations to be flexible
possible_env_paths = [
    '../.env',  # Root .env file
    '.env',     # Local .env file
    '.env.local',  # Local development .env file
    '../../.env',  # Parent directory .env file
    './multi_agent_system/.env',  # Multi-agent system .env file
    '../multi_agent_system/.env'  # Alternative multi-agent system .env file
]

# Try to load from environment variable first
env_path = os.getenv('ENV_FILE_PATH')
if env_path and os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Try multiple locations to be flexible
    for env_path in possible_env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            break
    else:
        # If no .env file found, load from environment
        load_dotenv()

class Config:
    """System configuration management."""
    
    def __init__(self):
        self._reload_from_env()
    
    def _reload_from_env(self):
        """Reload configuration from environment variables."""
        # API Keys
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
        self.OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
        
        # API Endpoints
        self.QWEN_API_BASE: str = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/api/v1")
        self.OPENROUTER_API_BASE: str = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        
        # Model Configuration
        self.QWEN_CODER_MODEL: str = os.getenv("QWEN_CODER_MODEL", "qwen-coder-plus")
        self.QWEN_ORCHESTRATOR_MODEL: str = os.getenv("QWEN_ORCHESTRATOR_MODEL", "qwen2.5-1m")
        self.QWEN_VISION_MODEL: str = os.getenv("QWEN_VISION_MODEL", "qwen2-vl-7b-instruct")
        
        # Claude Configuration
        self.CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        self.CLAUDE_MAX_TOKENS: int = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
        
        # OpenRouter Model Mapping (when using OpenRouter)
        self.OPENROUTER_QWEN_ORCHESTRATOR_MODEL: str = os.getenv("OPENROUTER_QWEN_ORCHESTRATOR_MODEL", "qwen/qwen2.5-72b-instruct:free")
        self.OPENROUTER_CLAUDE_MODEL: str = os.getenv("OPENROUTER_CLAUDE_MODEL", "anthropic/claude-3.5-sonnet")
        self.OPENROUTER_QWEN_CODER_MODEL: str = os.getenv("OPENROUTER_QWEN_CODER_MODEL", "qwen/qwen3-coder:free")
        self.OPENROUTER_QWEN_VISION_MODEL: str = os.getenv("OPENROUTER_QWEN_VISION_MODEL", "qwen/qwen2.5-vl-32b-instruct:free")
        
        # OpenAI Configuration (for embeddings)
        self.OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # Vector Database
        self.CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self.CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_artifacts")
        
        # System Configuration
        self.MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
        self.REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
        self.RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Memory System
        self.MEMORY_SEARCH_TOP_K: int = int(os.getenv("MEMORY_SEARCH_TOP_K", "5"))
        self.MEMORY_SIMILARITY_THRESHOLD: float = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.7"))
        self.MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "1000000"))
        
        # Cross Validation
        self.CROSS_VALIDATION_THRESHOLD: float = float(os.getenv("CROSS_VALIDATION_THRESHOLD", "0.8"))
        self.MIN_CONFIDENCE_SCORE: float = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.6"))
        
        # Use OpenRouter flag
        self.USE_OPENROUTER: bool = os.getenv("USE_OPENROUTER", "false").lower() == "true"
        
        # Testing Mode (use free tier models)
        self.USE_TESTING_MODE: bool = os.getenv("USE_TESTING_MODE", "false").lower() == "true"
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if self.USE_OPENROUTER:
            # When using OpenRouter, we only need the OpenRouter API key
            if not self.OPENROUTER_API_KEY:
                raise ValueError("Missing required OPENROUTER_API_KEY when USE_OPENROUTER=true")
        else:
            # When not using OpenRouter, we need the individual API keys
            required_keys = [
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "QWEN_API_KEY"
            ]
            
            missing_keys = []
            for key in required_keys:
                if not getattr(self, key):
                    missing_keys.append(key)
            
            if missing_keys:
                raise ValueError(f"Missing required environment variables: {missing_keys}")
        
        return True

# Global config instance
config = Config()