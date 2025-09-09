"""Persistent Memory System using Chroma vector database."""

import asyncio
import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
import openai
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import config
from .data_structures import KnowledgeArtifact
from .logging_config import SystemLogger


class PersistentMemorySystem:
    """Vector-based memory system for storing and retrieving knowledge artifacts."""
    
    def __init__(self):
        self.logger = SystemLogger("memory_system")
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the memory system components."""
        try:
            # Initialize Chroma client
            os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embedding models
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            if config.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=config.OPENAI_API_KEY
                )
            
            self.logger.info(
                "Memory system initialized",
                collection_name=config.CHROMA_COLLECTION_NAME,
                persist_directory=config.CHROMA_PERSIST_DIRECTORY
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize memory system", error=str(e))
            raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI or local model."""
        try:
            if self.openai_client:
                # Use OpenAI embeddings
                response = await self.openai_client.embeddings.create(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    input=text
                )
                return response.data[0].embedding
            else:
                # Use local sentence transformer
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
                
        except Exception as e:
            self.logger.error("Failed to generate embedding", error=str(e))
            # Fallback to local model
            if self.embedding_model:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            raise
    
    async def store_artifact(self, artifact: KnowledgeArtifact) -> bool:
        """Store a knowledge artifact in the memory system."""
        try:
            start_time = time.time()
            
            # Generate embedding if not provided
            if not artifact.embedding:
                content_for_embedding = f"{artifact.title}\n{artifact.content}"
                artifact.embedding = await self._generate_embedding(content_for_embedding)
            
            # Prepare metadata
            metadata = {
                "title": artifact.title,
                "artifact_type": artifact.artifact_type,
                "tags": ",".join(artifact.tags),
                "source_task_id": artifact.source_task_id or "",
                "source_agent": artifact.source_agent.value if artifact.source_agent else "",
                "created_at": artifact.created_at.isoformat(),
                "access_count": artifact.access_count,
                **artifact.metadata
            }
            
            # Store in Chroma
            self.collection.add(
                embeddings=[artifact.embedding],
                documents=[artifact.content],
                metadatas=[metadata],
                ids=[artifact.id]
            )
            
            storage_time = time.time() - start_time
            self.logger.log_memory_operation(
                operation="store",
                artifact_count=1,
                search_time=storage_time
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to store artifact",
                artifact_id=artifact.id,
                error=str(e)
            )
            return False
    
    async def search_artifacts(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = None
    ) -> List[Tuple[KnowledgeArtifact, float]]:
        """Search for relevant knowledge artifacts."""
        try:
            start_time = time.time()
            
            top_k = top_k or config.MEMORY_SEARCH_TOP_K
            similarity_threshold = similarity_threshold or config.MEMORY_SIMILARITY_THRESHOLD
            
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Search in Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            artifacts_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    similarity_score = 1.0 - distance
                    
                    if similarity_score >= similarity_threshold:
                        # Reconstruct artifact
                        artifact = KnowledgeArtifact(
                            id=results['ids'][0][i],
                            title=metadata.get('title', ''),
                            content=doc,
                            artifact_type=metadata.get('artifact_type', ''),
                            tags=metadata.get('tags', '').split(',') if metadata.get('tags') else [],
                            source_task_id=metadata.get('source_task_id'),
                            source_agent=metadata.get('source_agent'),
                            created_at=metadata.get('created_at'),
                            access_count=metadata.get('access_count', 0),
                            metadata={k: v for k, v in metadata.items() 
                                    if k not in ['title', 'artifact_type', 'tags', 'source_task_id', 
                                               'source_agent', 'created_at', 'access_count']}
                        )
                        
                        # Update access count
                        artifact.increment_access()
                        
                        artifacts_with_scores.append((artifact, similarity_score))
            
            search_time = time.time() - start_time
            self.logger.log_memory_operation(
                operation="search",
                artifact_count=len(artifacts_with_scores),
                search_time=search_time
            )
            
            return artifacts_with_scores
            
        except Exception as e:
            self.logger.error("Failed to search artifacts", query=query, error=str(e))
            return []
    
    async def get_artifact(self, artifact_id: str) -> Optional[KnowledgeArtifact]:
        """Retrieve a specific artifact by ID."""
        try:
            results = self.collection.get(
                ids=[artifact_id],
                include=["documents", "metadatas"]
            )
            
            if results['documents'] and results['documents'][0]:
                doc = results['documents'][0]
                metadata = results['metadatas'][0]
                
                artifact = KnowledgeArtifact(
                    id=artifact_id,
                    title=metadata.get('title', ''),
                    content=doc,
                    artifact_type=metadata.get('artifact_type', ''),
                    tags=metadata.get('tags', '').split(',') if metadata.get('tags') else [],
                    source_task_id=metadata.get('source_task_id'),
                    source_agent=metadata.get('source_agent'),
                    created_at=metadata.get('created_at'),
                    access_count=metadata.get('access_count', 0),
                    metadata={k: v for k, v in metadata.items() 
                            if k not in ['title', 'artifact_type', 'tags', 'source_task_id', 
                                       'source_agent', 'created_at', 'access_count']}
                )
                
                return artifact
            
            return None
            
        except Exception as e:
            self.logger.error("Failed to get artifact", artifact_id=artifact_id, error=str(e))
            return None
    
    async def update_artifact(self, artifact: KnowledgeArtifact) -> bool:
        """Update an existing artifact."""
        try:
            # Generate new embedding if content changed
            if not artifact.embedding:
                content_for_embedding = f"{artifact.title}\n{artifact.content}"
                artifact.embedding = await self._generate_embedding(content_for_embedding)
            
            # Prepare metadata
            metadata = {
                "title": artifact.title,
                "artifact_type": artifact.artifact_type,
                "tags": ",".join(artifact.tags),
                "source_task_id": artifact.source_task_id or "",
                "source_agent": artifact.source_agent.value if artifact.source_agent else "",
                "created_at": artifact.created_at.isoformat(),
                "updated_at": artifact.updated_at.isoformat(),
                "access_count": artifact.access_count,
                **artifact.metadata
            }
            
            # Update in Chroma
            self.collection.update(
                ids=[artifact.id],
                embeddings=[artifact.embedding],
                documents=[artifact.content],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to update artifact",
                artifact_id=artifact.id,
                error=str(e)
            )
            return False
    
    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from the memory system."""
        try:
            self.collection.delete(ids=[artifact_id])
            self.logger.info("Artifact deleted", artifact_id=artifact_id)
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to delete artifact",
                artifact_id=artifact_id,
                error=str(e)
            )
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory collection."""
        try:
            count = self.collection.count()
            return {
                "total_artifacts": count,
                "collection_name": config.CHROMA_COLLECTION_NAME,
                "persist_directory": config.CHROMA_PERSIST_DIRECTORY
            }
            
        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            return {"total_artifacts": 0, "error": str(e)}
    
    async def cleanup_old_artifacts(self, days_old: int = 30) -> int:
        """Clean up artifacts older than specified days."""
        # This would require implementing date-based filtering
        # For now, return 0 as placeholder
        return 0
    
    async def store_checkpoint(self, checkpoint: 'Checkpoint') -> bool:
        """Store a checkpoint in the memory system."""
        try:
            start_time = time.time()
            
            # Create a knowledge artifact from the checkpoint
            artifact = KnowledgeArtifact(
                title=f"Checkpoint: {checkpoint.checkpoint_type.value} for task {checkpoint.task_id}",
                content=json.dumps(checkpoint.data, indent=2),
                artifact_type="checkpoint",
                tags=["checkpoint", checkpoint.checkpoint_type.value, checkpoint.task_id],
                source_task_id=checkpoint.task_id,
                source_agent=None,  # Checkpoints are system-level
                metadata={
                    "checkpoint_id": checkpoint.id,
                    "checkpoint_type": checkpoint.checkpoint_type.value,
                    "success": checkpoint.success,
                    "error_message": checkpoint.error_message,
                    "execution_time": checkpoint.execution_time,
                    "timestamp": checkpoint.timestamp.isoformat()
                }
            )
            
            # Generate embedding
            content_for_embedding = f"{artifact.title}\n{artifact.content}"
            artifact.embedding = await self._generate_embedding(content_for_embedding)
            
            # Prepare metadata
            metadata = {
                "title": artifact.title,
                "artifact_type": artifact.artifact_type,
                "tags": ",".join(artifact.tags),
                "source_task_id": artifact.source_task_id or "",
                "source_agent": artifact.source_agent.value if artifact.source_agent else "",
                "created_at": artifact.created_at.isoformat(),
                "access_count": artifact.access_count,
                **artifact.metadata
            }
            
            # Store in Chroma
            self.collection.add(
                embeddings=[artifact.embedding],
                documents=[artifact.content],
                metadatas=[metadata],
                ids=[artifact.id]
            )
            
            storage_time = time.time() - start_time
            self.logger.log_memory_operation(
                operation="store_checkpoint",
                artifact_count=1,
                search_time=storage_time
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to store checkpoint",
                checkpoint_id=checkpoint.id,
                error=str(e)
            )
            return False
    
    async def get_checkpoints_for_task(self, task_id: str) -> List['Checkpoint']:
        """Retrieve all checkpoints for a specific task."""
        try:
            # Search in Chroma for checkpoints with the task ID
            results = self.collection.get(
                where={"source_task_id": task_id},
                include=["documents", "metadatas"]
            )
            
            # Process results
            checkpoints = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    # Reconstruct checkpoint
                    try:
                        from datetime import datetime
                        from .data_structures import Checkpoint, CheckpointType
                        
                        # Parse timestamp
                        timestamp_str = metadata.get('timestamp', '')
                        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()
                        
                        checkpoint = Checkpoint(
                            id=results['ids'][i],
                            task_id=metadata.get('source_task_id', ''),
                            checkpoint_type=CheckpointType(metadata.get('checkpoint_type', 'requirements_analysis')),
                            data=json.loads(doc) if doc else {},
                            timestamp=timestamp,
                            success=metadata.get('success', True),
                            error_message=metadata.get('error_message'),
                            execution_time=metadata.get('execution_time', 0.0),
                            metadata={k: v for k, v in metadata.items() 
                                    if k not in ['title', 'artifact_type', 'tags', 'source_task_id', 
                                               'source_agent', 'created_at', 'access_count', 
                                               'checkpoint_id', 'checkpoint_type', 'success', 
                                               'error_message', 'execution_time', 'timestamp']}
                        )
                        
                        checkpoints.append(checkpoint)
                    except Exception as parse_error:
                        self.logger.warning("Failed to parse checkpoint", error=str(parse_error))
            
            return checkpoints
            
        except Exception as e:
            self.logger.error("Failed to get checkpoints for task", task_id=task_id, error=str(e))
            return []

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'client') and self.client:
            # Chroma client cleanup is handled automatically
            pass