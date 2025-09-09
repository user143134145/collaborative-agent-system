"""Cross-validation system for multi-agent consensus and confidence scoring."""

import asyncio
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .data_structures import Response, AgentType, Task
from .logging_config import SystemLogger
from .config import config


class ValidationStrategy(str, Enum):
    """Strategies for cross-validation."""
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    HIERARCHICAL = "hierarchical"
    THRESHOLD_CONSENSUS = "threshold_consensus"


@dataclass
class ValidationResult:
    """Result of cross-validation."""
    consensus_score: float
    final_response: Optional[Response] = None
    discrepancies: List[str] = None
    confidence_scores: Dict[str, float] = None
    validation_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.discrepancies is None:
            self.discrepancies = []
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.validation_metadata is None:
            self.validation_metadata = {}


class CrossValidator:
    """System for validating and achieving consensus across multiple agent responses."""
    
    def __init__(self):
        self.logger = SystemLogger("cross_validator")
        self.validation_strategies = {
            ValidationStrategy.MAJORITY_VOTE: self._majority_vote,
            ValidationStrategy.CONFIDENCE_WEIGHTED: self._confidence_weighted,
            ValidationStrategy.HIERARCHICAL: self._hierarchical,
            ValidationStrategy.THRESHOLD_CONSENSUS: self._threshold_consensus
        }
    
    async def validate_responses(self, responses: List[Response], task: Task, 
                               strategy: ValidationStrategy = ValidationStrategy.CONFIDENCE_WEIGHTED) -> ValidationResult:
        """Validate multiple agent responses and achieve consensus."""
        if not responses:
            raise ValueError("No responses to validate")
        
        self.logger.info(
            "Starting cross-validation", 
            task_id=task.id, 
            response_count=len(responses),
            strategy=strategy.value
        )
        
        # Filter out failed responses
        valid_responses = [r for r in responses if r.success and r.confidence_score >= config.MIN_CONFIDENCE_SCORE]
        
        if not valid_responses:
            return ValidationResult(
                consensus_score=0.0,
                discrepancies=["All responses failed or have low confidence"]
            )
        
        # Apply selected validation strategy
        validation_method = self.validation_strategies.get(strategy, self._confidence_weighted)
        result = await validation_method(valid_responses, task)
        
        # Detect inconsistencies
        result.discrepancies.extend(self._detect_inconsistencies(valid_responses))
        
        self.logger.log_cross_validation(
            task_id=task.id,
            consensus_score=result.consensus_score,
            agent_count=len(valid_responses)
        )
        
        return result
    
    async def _majority_vote(self, responses: List[Response], task: Task) -> ValidationResult:
        """Majority vote consensus strategy."""
        # Group responses by content similarity
        response_groups = self._group_similar_responses(responses)
        
        if not response_groups:
            return ValidationResult(consensus_score=0.0)
        
        # Find the largest group
        largest_group = max(response_groups, key=len)
        consensus_score = len(largest_group) / len(responses)
        
        # Select representative response from largest group (highest confidence)
        final_response = max(largest_group, key=lambda r: r.confidence_score)
        
        return ValidationResult(
            consensus_score=consensus_score,
            final_response=final_response,
            confidence_scores={r.agent_type.value: r.confidence_score for r in responses}
        )
    
    async def _confidence_weighted(self, responses: List[Response], task: Task) -> ValidationResult:
        """Confidence-weighted consensus strategy."""
        if len(responses) == 1:
            return ValidationResult(
                consensus_score=responses[0].confidence_score,
                final_response=responses[0],
                confidence_scores={responses[0].agent_type.value: responses[0].confidence_score}
            )
        
        # Calculate weighted average confidence
        total_confidence = sum(r.confidence_score for r in responses)
        consensus_score = total_confidence / len(responses)
        
        # For similar responses, combine them; for divergent, select highest confidence
        response_groups = self._group_similar_responses(responses)
        
        if len(response_groups) == 1:
            # All responses are similar - combine content
            combined_content = self._combine_responses(responses)
            final_response = self._create_combined_response(combined_content, responses, task)
        else:
            # Divergent responses - select highest confidence from largest group
            largest_group = max(response_groups, key=len)
            final_response = max(largest_group, key=lambda r: r.confidence_score)
            consensus_score *= (len(largest_group) / len(responses))  # Adjust for divergence
        
        return ValidationResult(
            consensus_score=consensus_score,
            final_response=final_response,
            confidence_scores={r.agent_type.value: r.confidence_score for r in responses}
        )
    
    async def _hierarchical(self, responses: List[Response], task: Task) -> ValidationResult:
        """Hierarchical validation with orchestrator as final arbiter."""
        # First, try confidence-weighted approach
        initial_result = await self._confidence_weighted(responses, task)
        
        # If consensus is low, use orchestrator to make final decision
        if initial_result.consensus_score < config.CROSS_VALIDATION_THRESHOLD:
            self.logger.info("Low consensus, using hierarchical validation", task_id=task.id)
            
            # For now, return the highest confidence response
            # In a full implementation, this would call the orchestrator agent
            final_response = max(responses, key=lambda r: r.confidence_score)
            initial_result.final_response = final_response
            initial_result.consensus_score = final_response.confidence_score
        
        return initial_result
    
    async def _threshold_consensus(self, responses: List[Response], task: Task) -> ValidationResult:
        """Threshold-based consensus requiring minimum agreement."""
        response_groups = self._group_similar_responses(responses)
        
        if not response_groups:
            return ValidationResult(consensus_score=0.0)
        
        # Find groups that meet the threshold
        threshold = config.CROSS_VALIDATION_THRESHOLD
        qualifying_groups = [group for group in response_groups if len(group) / len(responses) >= threshold]
        
        if not qualifying_groups:
            # No group meets threshold - return highest confidence response
            final_response = max(responses, key=lambda r: r.confidence_score)
            return ValidationResult(
                consensus_score=final_response.confidence_score,
                final_response=final_response,
                confidence_scores={r.agent_type.value: r.confidence_score for r in responses},
                discrepancies=["No consensus group met the threshold requirement"]
            )
        
        # Select the largest qualifying group
        largest_group = max(qualifying_groups, key=len)
        consensus_score = len(largest_group) / len(responses)
        final_response = max(largest_group, key=lambda r: r.confidence_score)
        
        return ValidationResult(
            consensus_score=consensus_score,
            final_response=final_response,
            confidence_scores={r.agent_type.value: r.confidence_score for r in responses}
        )
    
    def _group_similar_responses(self, responses: List[Response], similarity_threshold: float = 0.7) -> List[List[Response]]:
        """Group responses based on content similarity."""
        if not responses:
            return []
        
        groups = []
        processed = set()
        
        for i, response in enumerate(responses):
            if i in processed:
                continue
            
            group = [response]
            processed.add(i)
            
            for j, other_response in enumerate(responses[i+1:], start=i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_similarity(response.content, other_response.content)
                if similarity >= similarity_threshold:
                    group.append(other_response)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple heuristic."""
        # Simple Jaccard similarity based on words
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _detect_inconsistencies(self, responses: List[Response]) -> List[str]:
        """Detect inconsistencies between agent responses."""
        inconsistencies = []
        
        if len(responses) <= 1:
            return inconsistencies
        
        # Check for conflicting information in similar responses
        response_groups = self._group_similar_responses(responses, similarity_threshold=0.6)
        
        for group in response_groups:
            if len(group) > 1:
                # Compare each pair in the group
                for i, resp1 in enumerate(group):
                    for resp2 in group[i+1:]:
                        if self._has_conflict(resp1.content, resp2.content):
                            inconsistencies.append(
                                f"Conflict between {resp1.agent_type.value} and {resp2.agent_type.value}"
                            )
        
        return inconsistencies
    
    def _has_conflict(self, text1: str, text2: str) -> bool:
        """Check if two texts contain conflicting information."""
        # Simple conflict detection based on negation and contradictions
        negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor'}
        
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        # Check for presence of negation in one but not the other
        has_negation1 = any(word in negation_words for word in words1)
        has_negation2 = any(word in negation_words for word in words2)
        
        if has_negation1 != has_negation2:
            return True
        
        # Additional conflict detection logic can be added here
        return False
    
    def _combine_responses(self, responses: List[Response]) -> str:
        """Combine multiple responses into a comprehensive result."""
        if not responses:
            return ""
        
        combined = []
        responses_sorted = sorted(responses, key=lambda r: r.confidence_score, reverse=True)
        
        for response in responses_sorted:
            agent_header = f"--- {response.agent_type.value.upper()} (Confidence: {response.confidence_score:.2f}) ---"
            combined.append(f"{agent_header}\n{response.content}")
        
        return "\n\n".join(combined)
    
    def _create_combined_response(self, combined_content: str, original_responses: List[Response], task: Task) -> Response:
        """Create a combined response from multiple agent outputs."""
        # Use the highest confidence response as template
        best_response = max(original_responses, key=lambda r: r.confidence_score)
        
        return Response(
            task_id=task.id,
            agent_type=AgentType.ORCHESTRATOR,
            content=combined_content,
            confidence_score=sum(r.confidence_score for r in original_responses) / len(original_responses),
            execution_time=sum(r.execution_time for r in original_responses),
            tokens_used=sum(r.tokens_used or 0 for r in original_responses),
            success=True,
            metadata={
                "combined_from": [r.agent_type.value for r in original_responses],
                "original_confidences": {r.agent_type.value: r.confidence_score for r in original_responses}
            }
        )
    
    async def calculate_confidence_aggregate(self, responses: List[Response]) -> float:
        """Calculate aggregate confidence score from multiple responses."""
        if not responses:
            return 0.0
        
        valid_responses = [r for r in responses if r.success]
        if not valid_responses:
            return 0.0
        
        # Weighted average by confidence scores
        total_confidence = sum(r.confidence_score for r in valid_responses)
        return total_confidence / len(valid_responses)
    
    def get_available_strategies(self) -> List[ValidationStrategy]:
        """Get list of available validation strategies."""
        return list(self.validation_strategies.keys())


# Global instance for easy access
cross_validator = CrossValidator()