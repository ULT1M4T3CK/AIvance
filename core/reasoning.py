"""
Reasoning Engine

This module provides advanced reasoning capabilities for the AI system,
including response analysis, logical reasoning, and decision making.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import re


logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single reasoning step in the analysis process."""
    
    step_id: str
    step_type: str  # analysis, logic, decision, etc.
    description: str
    input_data: Any
    output_data: Any
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReasoningResult:
    """Result of a reasoning process."""
    
    reasoning_id: str
    steps: List[ReasoningStep]
    final_conclusion: str
    confidence_score: float
    reasoning_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ReasoningEngine:
    """
    Advanced reasoning engine for AI responses.
    
    This engine provides:
    - Response analysis and validation
    - Logical reasoning chains
    - Decision making support
    - Confidence scoring
    - Error detection and correction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ReasoningEngine")
        self.reasoning_cache: Dict[str, ReasoningResult] = {}
    
    async def analyze_response(
        self, 
        user_prompt: str, 
        ai_response: str
    ) -> List[str]:
        """
        Analyze an AI response for quality, relevance, and reasoning.
        
        Returns a list of reasoning steps as strings.
        """
        try:
            reasoning_steps = []
            
            # 1. Relevance analysis
            relevance_score = self._analyze_relevance(user_prompt, ai_response)
            reasoning_steps.append(f"Relevance Analysis: Response relevance score is {relevance_score:.2f}")
            
            # 2. Completeness check
            completeness = self._check_completeness(ai_response)
            reasoning_steps.append(f"Completeness Check: Response is {completeness}")
            
            # 3. Logical consistency
            consistency = self._check_logical_consistency(ai_response)
            reasoning_steps.append(f"Logical Consistency: {consistency}")
            
            # 4. Fact verification (basic)
            fact_check = self._basic_fact_check(ai_response)
            reasoning_steps.append(f"Fact Check: {fact_check}")
            
            # 5. Safety and ethics
            safety_check = self._safety_check(ai_response)
            reasoning_steps.append(f"Safety Check: {safety_check}")
            
            # 6. Response quality assessment
            quality_score = self._assess_response_quality(user_prompt, ai_response)
            reasoning_steps.append(f"Quality Assessment: Overall quality score is {quality_score:.2f}")
            
            return reasoning_steps
        
        except Exception as e:
            self.logger.error(f"Error in response analysis: {e}")
            return ["Error: Could not complete response analysis"]
    
    def _analyze_relevance(self, prompt: str, response: str) -> float:
        """Analyze how relevant the response is to the prompt."""
        try:
            # Simple keyword matching for relevance
            prompt_words = set(prompt.lower().split())
            response_words = set(response.lower().split())
            
            if not prompt_words:
                return 0.0
            
            # Calculate word overlap
            overlap = len(prompt_words.intersection(response_words))
            relevance = overlap / len(prompt_words)
            
            # Boost score if response is substantial
            if len(response_words) > 10:
                relevance = min(relevance * 1.2, 1.0)
            
            return relevance
        
        except Exception as e:
            self.logger.error(f"Error in relevance analysis: {e}")
            return 0.5
    
    def _check_completeness(self, response: str) -> str:
        """Check if the response appears complete."""
        try:
            # Check for common incomplete indicators
            incomplete_indicators = [
                "...",
                "etc.",
                "and so on",
                "more to come",
                "to be continued"
            ]
            
            response_lower = response.lower()
            for indicator in incomplete_indicators:
                if indicator in response_lower:
                    return "Potentially incomplete (contains continuation indicators)"
            
            # Check for proper ending
            if response.strip().endswith(('.', '!', '?')):
                return "Appears complete"
            else:
                return "May be incomplete (no clear ending)"
        
        except Exception as e:
            self.logger.error(f"Error in completeness check: {e}")
            return "Could not determine completeness"
    
    def _check_logical_consistency(self, response: str) -> str:
        """Check for logical consistency in the response."""
        try:
            # Check for contradictions
            contradictions = [
                ("yes", "no"),
                ("true", "false"),
                ("correct", "incorrect"),
                ("agree", "disagree")
            ]
            
            response_lower = response.lower()
            for pos, neg in contradictions:
                if pos in response_lower and neg in response_lower:
                    return f"Potential contradiction detected ({pos} vs {neg})"
            
            # Check for logical connectors
            logical_connectors = ["because", "therefore", "however", "although", "but"]
            connector_count = sum(1 for connector in logical_connectors if connector in response_lower)
            
            if connector_count > 0:
                return f"Good logical flow ({connector_count} logical connectors found)"
            else:
                return "Limited logical structure detected"
        
        except Exception as e:
            self.logger.error(f"Error in logical consistency check: {e}")
            return "Could not assess logical consistency"
    
    def _basic_fact_check(self, response: str) -> str:
        """Perform basic fact checking on the response."""
        try:
            # Check for specific claims that might need verification
            fact_indicators = [
                r"\d{4}",  # Years
                r"\d+%",   # Percentages
                r"\$\d+",  # Dollar amounts
                r"\d+ million",
                r"\d+ billion"
            ]
            
            claims_found = []
            for pattern in fact_indicators:
                matches = re.findall(pattern, response)
                if matches:
                    claims_found.extend(matches)
            
            if claims_found:
                return f"Contains {len(claims_found)} specific claims that may need verification"
            else:
                return "No specific factual claims detected"
        
        except Exception as e:
            self.logger.error(f"Error in fact check: {e}")
            return "Could not perform fact check"
    
    def _safety_check(self, response: str) -> str:
        """Check response for safety and ethical concerns."""
        try:
            # Check for potentially harmful content
            harmful_patterns = [
                r"harmful|dangerous|illegal",
                r"violence|attack|kill",
                r"hate|discrimination|bias",
                r"self-harm|suicide",
                r"explicit|inappropriate"
            ]
            
            response_lower = response.lower()
            safety_issues = []
            
            for pattern in harmful_patterns:
                if re.search(pattern, response_lower):
                    safety_issues.append(pattern)
            
            if safety_issues:
                return f"Potential safety concerns detected: {', '.join(safety_issues)}"
            else:
                return "No obvious safety concerns detected"
        
        except Exception as e:
            self.logger.error(f"Error in safety check: {e}")
            return "Could not perform safety check"
    
    def _assess_response_quality(self, prompt: str, response: str) -> float:
        """Assess overall response quality."""
        try:
            quality_score = 0.0
            
            # Length appropriateness (not too short, not too long)
            response_length = len(response.split())
            if 10 <= response_length <= 500:
                quality_score += 0.2
            elif response_length > 500:
                quality_score += 0.1
            
            # Relevance
            relevance = self._analyze_relevance(prompt, response)
            quality_score += relevance * 0.3
            
            # Completeness
            if "Appears complete" in self._check_completeness(response):
                quality_score += 0.2
            
            # Logical structure
            if "Good logical flow" in self._check_logical_consistency(response):
                quality_score += 0.2
            
            # Safety
            if "No obvious safety concerns" in self._safety_check(response):
                quality_score += 0.1
            
            return min(quality_score, 1.0)
        
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {e}")
            return 0.5
    
    async def perform_logical_reasoning(
        self, 
        premises: List[str], 
        question: str
    ) -> ReasoningResult:
        """
        Perform logical reasoning based on given premises.
        
        Args:
            premises: List of statements to use as premises
            question: The question to answer using logical reasoning
            
        Returns:
            ReasoningResult with the reasoning process and conclusion
        """
        try:
            reasoning_id = f"reasoning_{datetime.utcnow().timestamp()}"
            steps = []
            
            # Step 1: Analyze premises
            step1 = ReasoningStep(
                step_id=f"{reasoning_id}_step1",
                step_type="premise_analysis",
                description="Analyzing given premises for logical structure",
                input_data=premises,
                output_data={"premise_count": len(premises), "premise_types": self._categorize_premises(premises)},
                confidence=0.8
            )
            steps.append(step1)
            
            # Step 2: Identify logical relationships
            relationships = self._identify_logical_relationships(premises)
            step2 = ReasoningStep(
                step_id=f"{reasoning_id}_step2",
                step_type="relationship_analysis",
                description="Identifying logical relationships between premises",
                input_data=premises,
                output_data=relationships,
                confidence=0.7
            )
            steps.append(step2)
            
            # Step 3: Apply logical rules
            logical_conclusions = self._apply_logical_rules(premises, relationships)
            step3 = ReasoningStep(
                step_id=f"{reasoning_id}_step3",
                step_type="logical_inference",
                description="Applying logical rules to derive conclusions",
                input_data={"premises": premises, "relationships": relationships},
                output_data=logical_conclusions,
                confidence=0.6
            )
            steps.append(step3)
            
            # Step 4: Answer the question
            answer = self._answer_question(logical_conclusions, question)
            step4 = ReasoningStep(
                step_id=f"{reasoning_id}_step4",
                step_type="question_answering",
                description="Using derived conclusions to answer the question",
                input_data={"conclusions": logical_conclusions, "question": question},
                output_data=answer,
                confidence=0.5
            )
            steps.append(step4)
            
            # Calculate overall confidence
            overall_confidence = sum(step.confidence for step in steps) / len(steps)
            
            result = ReasoningResult(
                reasoning_id=reasoning_id,
                steps=steps,
                final_conclusion=answer,
                confidence_score=overall_confidence,
                reasoning_type="logical_reasoning"
            )
            
            # Cache the result
            self.reasoning_cache[reasoning_id] = result
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error in logical reasoning: {e}")
            return ReasoningResult(
                reasoning_id="error",
                steps=[],
                final_conclusion="Error: Could not perform logical reasoning",
                confidence_score=0.0,
                reasoning_type="error"
            )
    
    def _categorize_premises(self, premises: List[str]) -> Dict[str, int]:
        """Categorize premises by type."""
        categories = {
            "factual": 0,
            "conditional": 0,
            "definitional": 0,
            "causal": 0
        }
        
        for premise in premises:
            premise_lower = premise.lower()
            
            if any(word in premise_lower for word in ["if", "when", "unless"]):
                categories["conditional"] += 1
            elif any(word in premise_lower for word in ["is", "are", "means", "defined"]):
                categories["definitional"] += 1
            elif any(word in premise_lower for word in ["causes", "leads to", "results in"]):
                categories["causal"] += 1
            else:
                categories["factual"] += 1
        
        return categories
    
    def _identify_logical_relationships(self, premises: List[str]) -> List[Dict[str, Any]]:
        """Identify logical relationships between premises."""
        relationships = []
        
        # Simple relationship detection
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises[i+1:], i+1):
                relationship = self._find_relationship(premise1, premise2)
                if relationship:
                    relationships.append({
                        "premise1_index": i,
                        "premise2_index": j,
                        "relationship_type": relationship
                    })
        
        return relationships
    
    def _find_relationship(self, premise1: str, premise2: str) -> Optional[str]:
        """Find the logical relationship between two premises."""
        p1_lower = premise1.lower()
        p2_lower = premise2.lower()
        
        # Check for shared entities
        p1_words = set(p1_lower.split())
        p2_words = set(p2_lower.split())
        shared_words = p1_words.intersection(p2_words)
        
        if len(shared_words) >= 2:
            return "shared_entities"
        
        # Check for logical connectors
        if any(connector in p1_lower for connector in ["therefore", "thus", "so"]):
            return "conclusion"
        
        return None
    
    def _apply_logical_rules(self, premises: List[str], relationships: List[Dict[str, Any]]) -> List[str]:
        """Apply logical rules to derive conclusions."""
        conclusions = []
        
        # Simple rule application
        for premise in premises:
            if "if" in premise.lower() and "then" in premise.lower():
                # Extract conditional conclusion
                parts = premise.lower().split("then")
                if len(parts) > 1:
                    conclusions.append(parts[1].strip())
        
        return conclusions
    
    def _answer_question(self, conclusions: List[str], question: str) -> str:
        """Answer the question based on derived conclusions."""
        if not conclusions:
            return "No conclusions could be derived from the given premises."
        
        # Simple answer generation
        answer = f"Based on the logical analysis, the answer is: {' '.join(conclusions)}"
        return answer
    
    async def validate_argument(
        self, 
        argument: str
    ) -> Dict[str, Any]:
        """
        Validate the logical structure of an argument.
        
        Args:
            argument: The argument to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_result = {
                "valid": True,
                "issues": [],
                "strength": 0.0,
                "suggestions": []
            }
            
            # Check for logical fallacies
            fallacies = self._detect_logical_fallacies(argument)
            if fallacies:
                validation_result["valid"] = False
                validation_result["issues"].extend(fallacies)
            
            # Check argument structure
            structure_issues = self._check_argument_structure(argument)
            if structure_issues:
                validation_result["issues"].extend(structure_issues)
            
            # Assess argument strength
            validation_result["strength"] = self._assess_argument_strength(argument)
            
            # Generate suggestions
            validation_result["suggestions"] = self._generate_argument_suggestions(argument)
            
            return validation_result
        
        except Exception as e:
            self.logger.error(f"Error in argument validation: {e}")
            return {
                "valid": False,
                "issues": ["Error: Could not validate argument"],
                "strength": 0.0,
                "suggestions": []
            }
    
    def _detect_logical_fallacies(self, argument: str) -> List[str]:
        """Detect common logical fallacies in the argument."""
        fallacies = []
        argument_lower = argument.lower()
        
        # Ad hominem
        if any(word in argument_lower for word in ["stupid", "idiot", "fool", "ignorant"]):
            fallacies.append("Ad hominem fallacy detected")
        
        # Appeal to authority
        if any(phrase in argument_lower for phrase in ["experts say", "scientists agree", "everyone knows"]):
            fallacies.append("Appeal to authority fallacy detected")
        
        # False dichotomy
        if any(phrase in argument_lower for phrase in ["either", "or", "only two options"]):
            fallacies.append("Potential false dichotomy detected")
        
        return fallacies
    
    def _check_argument_structure(self, argument: str) -> List[str]:
        """Check the structure of the argument."""
        issues = []
        
        # Check for premises and conclusion
        if "because" not in argument.lower() and "therefore" not in argument.lower():
            issues.append("Missing clear logical structure (premises/conclusion)")
        
        # Check for evidence
        if not any(word in argument.lower() for word in ["evidence", "data", "study", "research"]):
            issues.append("No supporting evidence mentioned")
        
        return issues
    
    def _assess_argument_strength(self, argument: str) -> float:
        """Assess the strength of the argument."""
        strength = 0.5  # Base strength
        
        # Boost for evidence
        if any(word in argument.lower() for word in ["evidence", "data", "study", "research"]):
            strength += 0.2
        
        # Boost for logical connectors
        if any(word in argument.lower() for word in ["because", "therefore", "thus", "consequently"]):
            strength += 0.1
        
        # Boost for specificity
        if any(char.isdigit() for char in argument):
            strength += 0.1
        
        # Penalty for emotional language
        if any(word in argument.lower() for word in ["amazing", "terrible", "horrible", "wonderful"]):
            strength -= 0.1
        
        return max(0.0, min(1.0, strength))
    
    def _generate_argument_suggestions(self, argument: str) -> List[str]:
        """Generate suggestions for improving the argument."""
        suggestions = []
        
        if "because" not in argument.lower():
            suggestions.append("Add 'because' to clearly state your reasoning")
        
        if not any(word in argument.lower() for word in ["evidence", "data", "study"]):
            suggestions.append("Include supporting evidence or data")
        
        if len(argument.split()) < 20:
            suggestions.append("Provide more detail to strengthen your argument")
        
        return suggestions
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning operations."""
        return {
            "cached_results": len(self.reasoning_cache),
            "engine_status": "active",
            "capabilities": [
                "response_analysis",
                "logical_reasoning", 
                "argument_validation",
                "quality_assessment"
            ]
        }


# Global reasoning engine instance
reasoning_engine = ReasoningEngine() 