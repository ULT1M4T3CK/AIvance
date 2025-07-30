"""
Learning Engine

This module handles continuous learning from user interactions, feedback,
and system performance to improve AI responses over time.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib


logger = logging.getLogger(__name__)


@dataclass
class LearningEvent:
    """A learning event from user interaction."""
    
    event_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    event_type: str  # interaction, feedback, correction, etc.
    user_input: str
    ai_response: str
    user_feedback: Optional[str] = None
    feedback_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Generate event ID if not provided."""
        if not self.event_id:
            content_hash = hashlib.md5(
                f"{self.user_input}{self.ai_response}{self.timestamp.isoformat()}".encode()
            ).hexdigest()
            self.event_id = f"learn_{content_hash}"


@dataclass
class LearningPattern:
    """A learned pattern from user interactions."""
    
    pattern_id: str
    pattern_type: str  # response_style, topic_preference, etc.
    user_id: Optional[str]
    pattern_data: Dict[str, Any]
    confidence: float = 0.5
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LearningInsight:
    """An insight derived from learning data."""
    
    insight_id: str
    insight_type: str  # user_preference, system_improvement, etc.
    description: str
    data: Dict[str, Any]
    confidence: float
    actionable: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)


class LearningStore:
    """Base class for learning data storage."""
    
    async def store_event(self, event: LearningEvent) -> bool:
        """Store a learning event."""
        raise NotImplementedError
    
    async def store_pattern(self, pattern: LearningPattern) -> bool:
        """Store a learning pattern."""
        raise NotImplementedError
    
    async def get_user_events(self, user_id: str, limit: int = 100) -> List[LearningEvent]:
        """Get learning events for a user."""
        raise NotImplementedError
    
    async def get_patterns(self, user_id: Optional[str] = None) -> List[LearningPattern]:
        """Get learning patterns."""
        raise NotImplementedError
    
    async def update_pattern(self, pattern: LearningPattern) -> bool:
        """Update a learning pattern."""
        raise NotImplementedError


class FileLearningStore(LearningStore):
    """File-based learning storage implementation."""
    
    def __init__(self, storage_path: str = "./data/learning"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.events_file = self.storage_path / "events.json"
        self.patterns_file = self.storage_path / "patterns.json"
        self.insights_file = self.storage_path / "insights.json"
        
        self.events: Dict[str, LearningEvent] = {}
        self.patterns: Dict[str, LearningPattern] = {}
        self.insights: Dict[str, LearningInsight] = {}
        
        self.logger = logging.getLogger(f"{__name__}.FileLearningStore")
        self._load_data()
    
    def _load_data(self):
        """Load learning data from files."""
        try:
            if self.events_file.exists():
                with open(self.events_file, 'r') as f:
                    events_data = json.load(f)
                    self.events = {
                        eid: LearningEvent(**edata)
                        for eid, edata in events_data.items()
                    }
            
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    self.patterns = {
                        pid: LearningPattern(**pdata)
                        for pid, pdata in patterns_data.items()
                    }
            
            if self.insights_file.exists():
                with open(self.insights_file, 'r') as f:
                    insights_data = json.load(f)
                    self.insights = {
                        iid: LearningInsight(**idata)
                        for iid, idata in insights_data.items()
                    }
            
            self.logger.info(f"Loaded {len(self.events)} events, {len(self.patterns)} patterns, {len(self.insights)} insights")
        
        except Exception as e:
            self.logger.error(f"Error loading learning data: {e}")
    
    async def _save_data(self):
        """Save learning data to files."""
        try:
            # Save events
            events_data = {
                eid: asdict(event)
                for eid, event in self.events.items()
            }
            with open(self.events_file, 'w') as f:
                json.dump(events_data, f, indent=2, default=str)
            
            # Save patterns
            patterns_data = {
                pid: asdict(pattern)
                for pid, pattern in self.patterns.items()
            }
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            # Save insights
            insights_data = {
                iid: asdict(insight)
                for iid, insight in self.insights.items()
            }
            with open(self.insights_file, 'w') as f:
                json.dump(insights_data, f, indent=2, default=str)
            
            self.logger.debug("Learning data saved successfully")
        
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")
    
    async def store_event(self, event: LearningEvent) -> bool:
        """Store a learning event."""
        try:
            self.events[event.event_id] = event
            await self._save_data()
            self.logger.debug(f"Stored learning event: {event.event_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing learning event: {e}")
            return False
    
    async def store_pattern(self, pattern: LearningPattern) -> bool:
        """Store a learning pattern."""
        try:
            self.patterns[pattern.pattern_id] = pattern
            await self._save_data()
            self.logger.debug(f"Stored learning pattern: {pattern.pattern_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing learning pattern: {e}")
            return False
    
    async def get_user_events(self, user_id: str, limit: int = 100) -> List[LearningEvent]:
        """Get learning events for a user."""
        user_events = [
            event for event in self.events.values()
            if event.user_id == user_id
        ]
        
        # Sort by timestamp (newest first)
        user_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return user_events[:limit]
    
    async def get_patterns(self, user_id: Optional[str] = None) -> List[LearningPattern]:
        """Get learning patterns."""
        if user_id:
            patterns = [
                pattern for pattern in self.patterns.values()
                if pattern.user_id == user_id
            ]
        else:
            patterns = list(self.patterns.values())
        
        # Sort by confidence and usage count
        patterns.sort(key=lambda x: (x.confidence, x.usage_count), reverse=True)
        
        return patterns
    
    async def update_pattern(self, pattern: LearningPattern) -> bool:
        """Update a learning pattern."""
        try:
            if pattern.pattern_id in self.patterns:
                self.patterns[pattern.pattern_id] = pattern
                await self._save_data()
                self.logger.debug(f"Updated learning pattern: {pattern.pattern_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating learning pattern: {e}")
            return False


class LearningEngine:
    """
    Main learning engine that handles continuous learning from interactions.
    
    This engine provides:
    - Learning from user interactions
    - Pattern recognition and extraction
    - User preference learning
    - Response quality improvement
    - Feedback integration
    """
    
    def __init__(self, storage_path: str = "./data/learning"):
        self.store = FileLearningStore(storage_path)
        self.logger = logging.getLogger(f"{__name__}.LearningEngine")
        self.learning_enabled = True
    
    async def learn_from_interaction(
        self, 
        user_input: str, 
        ai_response: str, 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Learn from a user interaction."""
        if not self.learning_enabled:
            return
        
        try:
            # Create learning event
            event = LearningEvent(
                event_id="",
                user_id=user_id,
                session_id=session_id,
                event_type="interaction",
                user_input=user_input,
                ai_response=ai_response,
                metadata=metadata or {}
            )
            
            # Store event
            await self.store.store_event(event)
            
            # Extract patterns
            await self._extract_patterns(event)
            
            # Generate insights
            await self._generate_insights(event)
            
            self.logger.debug(f"Learned from interaction for user: {user_id}")
        
        except Exception as e:
            self.logger.error(f"Error learning from interaction: {e}")
    
    async def learn_from_feedback(
        self,
        user_input: str,
        ai_response: str,
        feedback: str,
        feedback_score: Optional[float] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Learn from user feedback."""
        if not self.learning_enabled:
            return
        
        try:
            # Create learning event with feedback
            event = LearningEvent(
                event_id="",
                user_id=user_id,
                session_id=session_id,
                event_type="feedback",
                user_input=user_input,
                ai_response=ai_response,
                user_feedback=feedback,
                feedback_score=feedback_score
            )
            
            # Store event
            await self.store.store_event(event)
            
            # Extract feedback patterns
            await self._extract_feedback_patterns(event)
            
            # Update response patterns based on feedback
            await self._update_response_patterns(event)
            
            self.logger.debug(f"Learned from feedback for user: {user_id}")
        
        except Exception as e:
            self.logger.error(f"Error learning from feedback: {e}")
    
    async def _extract_patterns(self, event: LearningEvent):
        """Extract patterns from a learning event."""
        try:
            # Extract response style patterns
            response_style = self._analyze_response_style(event.ai_response)
            if response_style:
                pattern = LearningPattern(
                    pattern_id=f"style_{event.user_id}_{hashlib.md5(event.user_id.encode()).hexdigest()[:8]}",
                    pattern_type="response_style",
                    user_id=event.user_id,
                    pattern_data=response_style,
                    confidence=0.6
                )
                await self.store.store_pattern(pattern)
            
            # Extract topic preferences
            topics = self._extract_topics(event.user_input)
            if topics:
                pattern = LearningPattern(
                    pattern_id=f"topics_{event.user_id}_{hashlib.md5(event.user_id.encode()).hexdigest()[:8]}",
                    pattern_type="topic_preference",
                    user_id=event.user_id,
                    pattern_data={"topics": topics},
                    confidence=0.5
                )
                await self.store.store_pattern(pattern)
            
            # Extract communication style
            comm_style = self._analyze_communication_style(event.user_input)
            if comm_style:
                pattern = LearningPattern(
                    pattern_id=f"comm_{event.user_id}_{hashlib.md5(event.user_id.encode()).hexdigest()[:8]}",
                    pattern_type="communication_style",
                    user_id=event.user_id,
                    pattern_data=comm_style,
                    confidence=0.7
                )
                await self.store.store_pattern(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting patterns: {e}")
    
    async def _extract_feedback_patterns(self, event: LearningEvent):
        """Extract patterns from feedback."""
        try:
            if not event.user_feedback:
                return
            
            # Analyze feedback sentiment and content
            feedback_analysis = self._analyze_feedback(event.user_feedback)
            
            pattern = LearningPattern(
                pattern_id=f"feedback_{event.user_id}_{hashlib.md5(event.user_feedback.encode()).hexdigest()[:8]}",
                pattern_type="feedback_pattern",
                user_id=event.user_id,
                pattern_data=feedback_analysis,
                confidence=0.8 if event.feedback_score else 0.5
            )
            
            await self.store.store_pattern(pattern)
        
        except Exception as e:
            self.logger.error(f"Error extracting feedback patterns: {e}")
    
    async def _update_response_patterns(self, event: LearningEvent):
        """Update response patterns based on feedback."""
        try:
            if not event.user_feedback or not event.feedback_score:
                return
            
            # Find existing response patterns for this user
            patterns = await self.store.get_patterns(event.user_id)
            response_patterns = [p for p in patterns if p.pattern_type == "response_style"]
            
            for pattern in response_patterns:
                # Update confidence based on feedback
                if event.feedback_score > 0.7:
                    pattern.confidence = min(pattern.confidence + 0.1, 1.0)
                elif event.feedback_score < 0.3:
                    pattern.confidence = max(pattern.confidence - 0.1, 0.0)
                
                pattern.usage_count += 1
                pattern.last_used = datetime.utcnow()
                
                await self.store.update_pattern(pattern)
        
        except Exception as e:
            self.logger.error(f"Error updating response patterns: {e}")
    
    async def _generate_insights(self, event: LearningEvent):
        """Generate insights from learning events."""
        try:
            # Analyze user behavior patterns
            user_events = await self.store.get_user_events(event.user_id, limit=50)
            
            if len(user_events) >= 5:
                # Generate user preference insights
                preference_insight = self._generate_preference_insight(user_events)
                if preference_insight:
                    await self._store_insight(preference_insight)
                
                # Generate interaction pattern insights
                pattern_insight = self._generate_pattern_insight(user_events)
                if pattern_insight:
                    await self._store_insight(pattern_insight)
        
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
    
    def _analyze_response_style(self, response: str) -> Optional[Dict[str, Any]]:
        """Analyze the style of an AI response."""
        try:
            style_data = {}
            
            # Analyze length
            word_count = len(response.split())
            style_data["length"] = "short" if word_count < 50 else "medium" if word_count < 200 else "long"
            
            # Analyze formality
            formal_indicators = ["therefore", "consequently", "furthermore", "moreover"]
            informal_indicators = ["hey", "cool", "awesome", "great"]
            
            formal_count = sum(1 for indicator in formal_indicators if indicator in response.lower())
            informal_count = sum(1 for indicator in informal_indicators if indicator in response.lower())
            
            if formal_count > informal_count:
                style_data["formality"] = "formal"
            elif informal_count > formal_count:
                style_data["formality"] = "informal"
            else:
                style_data["formality"] = "neutral"
            
            # Analyze structure
            if any(char in response for char in ["•", "-", "1.", "2."]):
                style_data["structure"] = "bulleted"
            elif "\n\n" in response:
                style_data["structure"] = "paragraphs"
            else:
                style_data["structure"] = "continuous"
            
            return style_data
        
        except Exception as e:
            self.logger.error(f"Error analyzing response style: {e}")
            return None
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from user input."""
        try:
            # Simple topic extraction based on keywords
            topic_keywords = {
                "technology": ["tech", "computer", "software", "programming", "ai", "machine learning"],
                "science": ["science", "research", "experiment", "study", "theory"],
                "business": ["business", "company", "market", "finance", "strategy"],
                "health": ["health", "medical", "fitness", "wellness", "diet"],
                "education": ["education", "learning", "teaching", "school", "university"],
                "entertainment": ["movie", "music", "game", "entertainment", "fun"]
            }
            
            text_lower = text.lower()
            found_topics = []
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    found_topics.append(topic)
            
            return found_topics
        
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []
    
    def _analyze_communication_style(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze user communication style."""
        try:
            style_data = {}
            
            # Analyze formality
            formal_words = ["please", "thank you", "would you", "could you"]
            informal_words = ["hey", "hi", "thanks", "cool", "awesome"]
            
            formal_count = sum(1 for word in formal_words if word in text.lower())
            informal_count = sum(1 for word in informal_words if word in text.lower())
            
            if formal_count > informal_count:
                style_data["formality"] = "formal"
            elif informal_count > formal_count:
                style_data["formality"] = "informal"
            else:
                style_data["formality"] = "neutral"
            
            # Analyze directness
            question_count = text.count("?")
            if question_count > 0:
                style_data["directness"] = "questioning"
            elif len(text.split()) < 10:
                style_data["directness"] = "direct"
            else:
                style_data["directness"] = "detailed"
            
            return style_data
        
        except Exception as e:
            self.logger.error(f"Error analyzing communication style: {e}")
            return None
    
    def _analyze_feedback(self, feedback: str) -> Dict[str, Any]:
        """Analyze user feedback."""
        try:
            analysis = {}
            
            # Sentiment analysis (simple)
            positive_words = ["good", "great", "excellent", "helpful", "thanks", "perfect"]
            negative_words = ["bad", "wrong", "incorrect", "useless", "terrible", "awful"]
            
            feedback_lower = feedback.lower()
            positive_count = sum(1 for word in positive_words if word in feedback_lower)
            negative_count = sum(1 for word in negative_words if word in feedback_lower)
            
            if positive_count > negative_count:
                analysis["sentiment"] = "positive"
            elif negative_count > positive_count:
                analysis["sentiment"] = "negative"
            else:
                analysis["sentiment"] = "neutral"
            
            # Feedback type
            if "?" in feedback:
                analysis["type"] = "question"
            elif any(word in feedback_lower for word in ["wrong", "incorrect", "error"]):
                analysis["type"] = "correction"
            elif any(word in feedback_lower for word in ["thanks", "thank you", "good"]):
                analysis["type"] = "appreciation"
            else:
                analysis["type"] = "general"
            
            return analysis
        
        except Exception as e:
            self.logger.error(f"Error analyzing feedback: {e}")
            return {"sentiment": "neutral", "type": "general"}
    
    def _generate_preference_insight(self, events: List[LearningEvent]) -> Optional[LearningInsight]:
        """Generate user preference insights."""
        try:
            if len(events) < 3:
                return None
            
            # Analyze recent events for patterns
            recent_events = events[:10]
            
            # Extract common topics
            all_topics = []
            for event in recent_events:
                topics = self._extract_topics(event.user_input)
                all_topics.extend(topics)
            
            if all_topics:
                topic_counts = {}
                for topic in all_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                preferred_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                
                insight = LearningInsight(
                    insight_id=f"pref_{events[0].user_id}_{datetime.utcnow().timestamp()}",
                    insight_type="user_preference",
                    description=f"User shows preference for topics: {', '.join([topic for topic, _ in preferred_topics])}",
                    data={"preferred_topics": preferred_topics},
                    confidence=0.7
                )
                
                return insight
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error generating preference insight: {e}")
            return None
    
    def _generate_pattern_insight(self, events: List[LearningEvent]) -> Optional[LearningInsight]:
        """Generate interaction pattern insights."""
        try:
            if len(events) < 5:
                return None
            
            # Analyze interaction patterns
            interaction_times = [event.timestamp for event in events]
            time_diffs = []
            
            for i in range(1, len(interaction_times)):
                diff = (interaction_times[i-1] - interaction_times[i]).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs:
                avg_time_diff = sum(time_diffs) / len(time_diffs)
                
                if avg_time_diff < 60:  # Less than 1 minute
                    pattern = "rapid_interaction"
                elif avg_time_diff < 300:  # Less than 5 minutes
                    pattern = "moderate_interaction"
                else:
                    pattern = "sporadic_interaction"
                
                insight = LearningInsight(
                    insight_id=f"pattern_{events[0].user_id}_{datetime.utcnow().timestamp()}",
                    insight_type="interaction_pattern",
                    description=f"User shows {pattern} pattern with average {avg_time_diff:.1f} seconds between interactions",
                    data={"pattern": pattern, "avg_time_diff": avg_time_diff},
                    confidence=0.6
                )
                
                return insight
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error generating pattern insight: {e}")
            return None
    
    async def _store_insight(self, insight: LearningInsight):
        """Store a learning insight."""
        try:
            # Store in memory for now (could be extended to persistent storage)
            self.store.insights[insight.insight_id] = insight
            await self.store._save_data()
            
            self.logger.debug(f"Stored learning insight: {insight.insight_id}")
        
        except Exception as e:
            self.logger.error(f"Error storing insight: {e}")
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned user preferences."""
        try:
            patterns = await self.store.get_patterns(user_id)
            preferences = {}
            
            for pattern in patterns:
                if pattern.pattern_type == "response_style":
                    preferences["response_style"] = pattern.pattern_data
                elif pattern.pattern_type == "topic_preference":
                    preferences["preferred_topics"] = pattern.pattern_data.get("topics", [])
                elif pattern.pattern_type == "communication_style":
                    preferences["communication_style"] = pattern.pattern_data
            
            return preferences
        
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {e}")
            return {}
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        try:
            total_events = len(self.store.events)
            total_patterns = len(self.store.patterns)
            total_insights = len(self.store.insights)
            
            # Count events by type
            event_types = {}
            for event in self.store.events.values():
                event_type = event.event_type
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Count patterns by type
            pattern_types = {}
            for pattern in self.store.patterns.values():
                pattern_type = pattern.pattern_type
                pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            
            return {
                "total_events": total_events,
                "total_patterns": total_patterns,
                "total_insights": total_insights,
                "event_types": event_types,
                "pattern_types": pattern_types,
                "learning_enabled": self.learning_enabled
            }
        
        except Exception as e:
            self.logger.error(f"Error getting learning statistics: {e}")
            return {}
    
    async def save_learned_data(self):
        """Save all learned data."""
        try:
            await self.store._save_data()
            self.logger.info("All learned data saved")
        except Exception as e:
            self.logger.error(f"Error saving learned data: {e}")


# Global learning engine instance
learning_engine = LearningEngine() 