"""
Remediation Learning Module
==========================

This module provides reinforcement learning capabilities for automated remediation.
"""

import numpy as np
import pandas as pd
import json
import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from collections import defaultdict

# Placeholder for ML libraries
# In a real implementation, we would import:
# import tensorflow as tf
# from stable_baselines3 import PPO, A2C, DQN

logger = logging.getLogger(__name__)

class RemediationAction:
    """Represents a remediation action that can be taken."""
    
    def __init__(self, 
                 action_id: str, 
                 name: str, 
                 description: str,
                 target_type: str,
                 parameters: Dict[str, Any] = None,
                 preconditions: List[str] = None,
                 estimated_duration: int = 0,
                 risk_level: int = 0):
        """Initialize a remediation action.
        
        Args:
            action_id: Unique identifier for the action
            name: Human-readable name
            description: Detailed description
            target_type: Type of target (service, node, cluster)
            parameters: Parameters required for the action
            preconditions: Conditions that must be met before action
            estimated_duration: Estimated duration in seconds
            risk_level: Risk level (0-5, where 5 is highest risk)
        """
        self.action_id = action_id
        self.name = name
        self.description = description
        self.target_type = target_type
        self.parameters = parameters or {}
        self.preconditions = preconditions or []
        self.estimated_duration = estimated_duration
        self.risk_level = risk_level
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the action to a dictionary.
        
        Returns:
            Dictionary representation of the action
        """
        return {
            'action_id': self.action_id,
            'name': self.name,
            'description': self.description,
            'target_type': self.target_type,
            'parameters': self.parameters,
            'preconditions': self.preconditions,
            'estimated_duration': self.estimated_duration,
            'risk_level': self.risk_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RemediationAction':
        """Create an action from a dictionary.
        
        Args:
            data: Dictionary representation of the action
            
        Returns:
            RemediationAction instance
        """
        return cls(
            action_id=data['action_id'],
            name=data['name'],
            description=data['description'],
            target_type=data['target_type'],
            parameters=data.get('parameters', {}),
            preconditions=data.get('preconditions', []),
            estimated_duration=data.get('estimated_duration', 0),
            risk_level=data.get('risk_level', 0)
        )


class RemediationState:
    """Represents the state of a system during remediation."""
    
    def __init__(self, 
                 metrics: Dict[str, float] = None,
                 issue_type: str = None,
                 issue_severity: int = 0,
                 affected_services: List[str] = None,
                 service_states: Dict[str, str] = None,
                 previous_actions: List[str] = None):
        """Initialize a remediation state.
        
        Args:
            metrics: Current system metrics
            issue_type: Type of issue being remediated
            issue_severity: Severity of the issue (0-5)
            affected_services: List of affected service IDs
            service_states: Dictionary mapping service IDs to states
            previous_actions: List of previously taken action IDs
        """
        self.metrics = metrics or {}
        self.issue_type = issue_type
        self.issue_severity = issue_severity
        self.affected_services = affected_services or []
        self.service_states = service_states or {}
        self.previous_actions = previous_actions or []
        
    def to_feature_vector(self) -> np.ndarray:
        """Convert the state to a feature vector for ML models.
        
        Returns:
            NumPy array representing the state
        """
        # In a real implementation, we would:
        # 1. Extract relevant features from metrics
        # 2. One-hot encode categorical variables
        # 3. Normalize numerical features
        
        # Placeholder implementation
        features = []
        
        # Add metrics
        for metric_name in sorted(self.metrics.keys()):
            features.append(self.metrics[metric_name])
        
        # Add issue severity
        features.append(self.issue_severity / 5.0)  # Normalize to [0, 1]
        
        # Add number of affected services
        features.append(len(self.affected_services) / 10.0)  # Normalize assuming max 10 services
        
        # Add number of previous actions
        features.append(len(self.previous_actions) / 5.0)  # Normalize assuming max 5 actions
        
        return np.array(features)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        return {
            'metrics': self.metrics,
            'issue_type': self.issue_type,
            'issue_severity': self.issue_severity,
            'affected_services': self.affected_services,
            'service_states': self.service_states,
            'previous_actions': self.previous_actions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RemediationState':
        """Create a state from a dictionary.
        
        Args:
            data: Dictionary representation of the state
            
        Returns:
            RemediationState instance
        """
        return cls(
            metrics=data.get('metrics', {}),
            issue_type=data.get('issue_type'),
            issue_severity=data.get('issue_severity', 0),
            affected_services=data.get('affected_services', []),
            service_states=data.get('service_states', {}),
            previous_actions=data.get('previous_actions', [])
        )


class RemediationExperience:
    """Represents a single remediation experience for learning."""
    
    def __init__(self, 
                 initial_state: RemediationState,
                 action: RemediationAction,
                 next_state: RemediationState,
                 reward: float,
                 timestamp: datetime = None,
                 success: bool = False,
                 notes: str = None):
        """Initialize a remediation experience.
        
        Args:
            initial_state: State before the action
            action: Action taken
            next_state: State after the action
            reward: Reward received for the action
            timestamp: When the experience occurred
            success: Whether the remediation was successful
            notes: Additional notes about the experience
        """
        self.initial_state = initial_state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.timestamp = timestamp or datetime.now()
        self.success = success
        self.notes = notes
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the experience to a dictionary.
        
        Returns:
            Dictionary representation of the experience
        """
        return {
            'initial_state': self.initial_state.to_dict(),
            'action': self.action.to_dict(),
            'next_state': self.next_state.to_dict(),
            'reward': self.reward,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RemediationExperience':
        """Create an experience from a dictionary.
        
        Args:
            data: Dictionary representation of the experience
            
        Returns:
            RemediationExperience instance
        """
        return cls(
            initial_state=RemediationState.from_dict(data['initial_state']),
            action=RemediationAction.from_dict(data['action']),
            next_state=RemediationState.from_dict(data['next_state']),
            reward=data['reward'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            success=data.get('success', False),
            notes=data.get('notes')
        )


class RemediationLearner:
    """Base class for remediation learning algorithms."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the remediation learner.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def train(self, experiences: List[RemediationExperience]) -> None:
        """Train the model using past remediation experiences.
        
        Args:
            experiences: List of remediation experiences
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict_action(self, state: RemediationState, available_actions: List[RemediationAction]) -> Tuple[RemediationAction, float]:
        """Predict the best action for a given state.
        
        Args:
            state: Current system state
            available_actions: List of available actions
            
        Returns:
            Tuple of (best action, confidence score)
        """
        raise NotImplementedError("Subclasses must implement predict_action()")
    
    def update(self, experience: RemediationExperience) -> None:
        """Update the model with a new experience.
        
        Args:
            experience: New remediation experience
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def save_model(self, path: str) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError("Subclasses must implement save_model()")
    
    def load_model(self, path: str) -> None:
        """Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        raise NotImplementedError("Subclasses must implement load_model()")


class RuleLearner(RemediationLearner):
    """Rule-based remediation learner using decision trees."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the rule learner.
        
        Args:
            config: Configuration parameters
                - min_samples: Minimum samples to create a rule
                - confidence_threshold: Minimum confidence for a rule
        """
        super().__init__(config)
        self.min_samples = self.config.get('min_samples', 3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.rules = defaultdict(lambda: defaultdict(list))
        self.rule_confidences = {}
        
    def train(self, experiences: List[RemediationExperience]) -> None:
        """Train the rule-based model using past experiences.
        
        Args:
            experiences: List of remediation experiences
        """
        # Group experiences by issue type and action
        for exp in experiences:
            issue_type = exp.initial_state.issue_type
            action_id = exp.action.action_id
            self.rules[issue_type][action_id].append(exp)
        
        # Calculate success rates for each rule
        for issue_type, actions in self.rules.items():
            for action_id, exps in actions.items():
                if len(exps) >= self.min_samples:
                    successes = sum(1 for exp in exps if exp.success)
                    confidence = successes / len(exps)
                    self.rule_confidences[(issue_type, action_id)] = confidence
        
        self.is_trained = True
        self.logger.info(f"Trained rule learner with {len(experiences)} experiences")
        
    def predict_action(self, state: RemediationState, available_actions: List[RemediationAction]) -> Tuple[RemediationAction, float]:
        """Predict the best action for a given state.
        
        Args:
            state: Current system state
            available_actions: List of available actions
            
        Returns:
            Tuple of (best action, confidence score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        issue_type = state.issue_type
        best_action = None
        best_confidence = 0
        
        # Find the action with the highest confidence for this issue type
        for action in available_actions:
            confidence = self.rule_confidences.get((issue_type, action.action_id), 0)
            if confidence > best_confidence and confidence >= self.confidence_threshold:
                best_confidence = confidence
                best_action = action
        
        # If no good action found, return the lowest risk action
        if best_action is None and available_actions:
            best_action = min(available_actions, key=lambda a: a.risk_level)
            best_confidence = 0.5  # Default confidence
            
        return best_action, best_confidence
    
    def update(self, experience: RemediationExperience) -> None:
        """Update the model with a new experience.
        
        Args:
            experience: New remediation experience
        """
        issue_type = experience.initial_state.issue_type
        action_id = experience.action.action_id
        
        # Add the new experience
        self.rules[issue_type][action_id].append(experience)
        
        # Recalculate confidence
        exps = self.rules[issue_type][action_id]
        if len(exps) >= self.min_samples:
            successes = sum(1 for exp in exps if exp.success)
            confidence = successes / len(exps)
            self.rule_confidences[(issue_type, action_id)] = confidence
            
        self.logger.info(f"Updated rule for issue type {issue_type}, action {action_id}, new confidence: {confidence:.2f}")
    
    def save_model(self, path: str) -> None:
        """Save the rule-based model to disk.
        
        Args:
            path: Path to save the model
        """
        # Convert rules to a serializable format
        serializable_rules = {}
        for issue_type, actions in self.rules.items():
            serializable_rules[issue_type] = {}
            for action_id, exps in actions.items():
                serializable_rules[issue_type][action_id] = [exp.to_dict() for exp in exps]
        
        model_data = {
            'rules': serializable_rules,
            'confidences': {f"{k[0]}:{k[1]}": v for k, v in self.rule_confidences.items()},
            'config': self.config
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f)
            
        self.logger.info(f"Saved rule-based model to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a rule-based model from disk.
        
        Args:
            path: Path to the saved model
        """
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        # Load configuration
        self.config = model_data['config']
        self.min_samples = self.config.get('min_samples', 3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Load rules
        self.rules = defaultdict(lambda: defaultdict(list))
        for issue_type, actions in model_data['rules'].items():
            for action_id, exps in actions.items():
                self.rules[issue_type][action_id] = [RemediationExperience.from_dict(exp) for exp in exps]
        
        # Load confidences
        self.rule_confidences = {}
        for key, value in model_data['confidences'].items():
            issue_type, action_id = key.split(':')
            self.rule_confidences[(issue_type, action_id)] = value
        
        self.is_trained = True
        self.logger.info(f"Loaded rule-based model from {path}")


class ReinforcementLearner(RemediationLearner):
    """Reinforcement learning-based remediation learner."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the reinforcement learner.
        
        Args:
            config: Configuration parameters
                - algorithm: RL algorithm ('dqn', 'ppo', 'a2c')
                - learning_rate: Learning rate for training
                - discount_factor: Discount factor for future rewards
                - exploration_rate: Initial exploration rate
        """
        super().__init__(config)
        self.algorithm = self.config.get('algorithm', 'dqn')
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        
    def train(self, experiences: List[RemediationExperience]) -> None:
        """Train the reinforcement learning model using past experiences.
        
        Args:
            experiences: List of remediation experiences
        """
        # In a real implementation, we would:
        # 1. Convert experiences to state-action-reward-next_state tuples
        # 2. Create a replay buffer
        # 3. Train a DQN, PPO, or A2C model
        
        self.logger.info(f"Training {self.algorithm} model with {len(experiences)} experiences")
        self.is_trained = True
        
    def predict_action(self, state: RemediationState, available_actions: List[RemediationAction]) -> Tuple[RemediationAction, float]:
        """Predict the best action for a given state.
        
        Args:
            state: Current system state
            available_actions: List of available actions
            
        Returns:
            Tuple of (best action, confidence score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # In a real implementation, we would:
        # 1. Convert state to a feature vector
        # 2. Use the trained model to predict Q-values for each action
        # 3. Select the action with the highest Q-value
        
        # Placeholder implementation
        if not available_actions:
            return None, 0
        
        # Simulate Q-values with random scores
        q_values = {action.action_id: np.random.random() for action in available_actions}
        best_action_id = max(q_values, key=q_values.get)
        best_action = next(a for a in available_actions if a.action_id == best_action_id)
        confidence = q_values[best_action_id]
        
        return best_action, confidence
    
    def update(self, experience: RemediationExperience) -> None:
        """Update the model with a new experience.
        
        Args:
            experience: New remediation experience
        """
        # In a real implementation, we would:
        # 1. Add the experience to the replay buffer
        # 2. Perform a training step
        
        self.logger.info(f"Updated RL model with new experience")
    
    def save_model(self, path: str) -> None:
        """Save the RL model to disk.
        
        Args:
            path: Path to save the model
        """
        # In a real implementation, we would save the TensorFlow/PyTorch model
        pass
    
    def load_model(self, path: str) -> None:
        """Load an RL model from disk.
        
        Args:
            path: Path to the saved model
        """
        # In a real implementation, we would load the TensorFlow/PyTorch model
        pass


class RemediationLearningEngine:
    """Engine for coordinating remediation learning."""
    
    def __init__(self):
        """Initialize the remediation learning engine."""
        self.learners = {}
        self.action_library = {}
        self.experience_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def add_learner(self, name: str, learner: RemediationLearner) -> None:
        """Add a remediation learner to the engine.
        
        Args:
            name: Name of the learner
            learner: RemediationLearner instance
        """
        self.learners[name] = learner
        self.logger.info(f"Added learner: {name}")
        
    def remove_learner(self, name: str) -> None:
        """Remove a remediation learner from the engine.
        
        Args:
            name: Name of the learner to remove
        """
        if name in self.learners:
            del self.learners[name]
            self.logger.info(f"Removed learner: {name}")
    
    def register_action(self, action: RemediationAction) -> None:
        """Register a remediation action in the library.
        
        Args:
            action: RemediationAction to register
        """
        self.action_library[action.action_id] = action
        self.logger.info(f"Registered action: {action.name}")
    
    def get_available_actions(self, state: RemediationState) -> List[RemediationAction]:
        """Get available actions for a given state.
        
        Args:
            state: Current system state
            
        Returns:
            List of available actions
        """
        # Filter actions based on preconditions
        available_actions = []
        
        for action in self.action_library.values():
            # Check if action is applicable to the affected services
            if action.target_type in [service.split('/')[0] for service in state.affected_services]:
                # Check preconditions (simplified)
                preconditions_met = True
                for precondition in action.preconditions:
                    # In a real implementation, we would evaluate preconditions
                    pass
                
                if preconditions_met:
                    available_actions.append(action)
        
        return available_actions
    
    def recommend_action(self, state: RemediationState) -> Tuple[RemediationAction, float, str]:
        """Recommend the best action for a given state.
        
        Args:
            state: Current system state
            
        Returns:
            Tuple of (best action, confidence score, learner name)
        """
        available_actions = self.get_available_actions(state)
        
        if not available_actions:
            self.logger.warning("No available actions for the current state")
            return None, 0, None
        
        best_action = None
        best_confidence = 0
        best_learner = None
        
        # Get recommendations from all learners
        for name, learner in self.learners.items():
            try:
                if learner.is_trained:
                    action, confidence = learner.predict_action(state, available_actions)
                    if action and confidence > best_confidence:
                        best_action = action
                        best_confidence = confidence
                        best_learner = name
            except Exception as e:
                self.logger.error(f"Error getting recommendation from learner {name}: {str(e)}")
        
        # If no learner provided a good recommendation, use a simple heuristic
        if best_action is None and available_actions:
            best_action = min(available_actions, key=lambda a: a.risk_level)
            best_confidence = 0.5
            best_learner = "default"
            
        return best_action, best_confidence, best_learner
    
    def record_experience(self, experience: RemediationExperience) -> None:
        """Record a remediation experience and update learners.
        
        Args:
            experience: RemediationExperience to record
        """
        self.experience_history.append(experience)
        
        # Update all learners
        for name, learner in self.learners.items():
            try:
                learner.update(experience)
            except Exception as e:
                self.logger.error(f"Error updating learner {name}: {str(e)}")
        
        self.logger.info(f"Recorded new remediation experience: {experience.action.name}, success: {experience.success}")
    
    def train_all(self) -> None:
        """Train all learners using the experience history."""
        for name, learner in self.learners.items():
            try:
                learner.train(self.experience_history)
                self.logger.info(f"Trained learner: {name}")
            except Exception as e:
                self.logger.error(f"Error training learner {name}: {str(e)}")
    
    def save_all(self, base_path: str) -> None:
        """Save all learner models to disk.
        
        Args:
            base_path: Base path for saving models
        """
        os.makedirs(base_path, exist_ok=True)
        
        for name, learner in self.learners.items():
            try:
                path = f"{base_path}/{name}_model.json"
                learner.save_model(path)
                self.logger.info(f"Saved model for learner: {name}")
            except Exception as e:
                self.logger.error(f"Error saving model for learner {name}: {str(e)}")
        
        # Save action library
        action_library_path = f"{base_path}/action_library.json"
        with open(action_library_path, 'w') as f:
            json.dump({
                action_id: action.to_dict() 
                for action_id, action in self.action_library.items()
            }, f)
        
        # Save experience history
        history_path = f"{base_path}/experience_history.json"
        with open(history_path, 'w') as f:
            json.dump([exp.to_dict() for exp in self.experience_history], f)
    
    def load_all(self, base_path: str) -> None:
        """Load all learner models from disk.
        
        Args:
            base_path: Base path for loading models
        """
        for name, learner in self.learners.items():
            try:
                path = f"{base_path}/{name}_model.json"
                if os.path.exists(path):
                    learner.load_model(path)
                    self.logger.info(f"Loaded model for learner: {name}")
            except Exception as e:
                self.logger.error(f"Error loading model for learner {name}: {str(e)}")
        
        # Load action library
        action_library_path = f"{base_path}/action_library.json"
        if os.path.exists(action_library_path):
            with open(action_library_path, 'r') as f:
                action_data = json.load(f)
                self.action_library = {
                    action_id: RemediationAction.from_dict(data)
                    for action_id, data in action_data.items()
                }
        
        # Load experience history
        history_path = f"{base_path}/experience_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                experience_data = json.load(f)
                self.experience_history = [
                    RemediationExperience.from_dict(data)
                    for data in experience_data
                ]


def create_default_engine() -> RemediationLearningEngine:
    """Create a default remediation learning engine with standard learners.
    
    Returns:
        Configured remediation learning engine
    """
    engine = RemediationLearningEngine()
    
    # Add rule-based learner
    rule_learner = RuleLearner({
        'min_samples': 3,
        'confidence_threshold': 0.7
    })
    engine.add_learner('rules', rule_learner)
    
    # Add reinforcement learner
    rl_learner = ReinforcementLearner({
        'algorithm': 'dqn',
        'learning_rate': 0.001,
        'discount_factor': 0.95
    })
    engine.add_learner('rl', rl_learner)
    
    # Register common remediation actions
    actions = [
        RemediationAction(
            action_id='restart_service',
            name='Restart Service',
            description='Restart a service that is experiencing issues',
            target_type='service',
            parameters={'service_name': 'string'},
            risk_level=2
        ),
        RemediationAction(
            action_id='scale_service',
            name='Scale Service',
            description='Scale a service to handle increased load',
            target_type='service',
            parameters={'service_name': 'string', 'replicas': 'integer'},
            risk_level=1
        ),
        RemediationAction(
            action_id='rollback_deployment',
            name='Rollback Deployment',
            description='Rollback a deployment to a previous version',
            target_type='deployment',
            parameters={'deployment_name': 'string', 'revision': 'integer'},
            risk_level=3
        ),
        RemediationAction(
            action_id='drain_node',
            name='Drain Node',
            description='Drain a node to move workloads elsewhere',
            target_type='node',
            parameters={'node_name': 'string'},
            risk_level=3
        ),
        RemediationAction(
            action_id='clear_cache',
            name='Clear Cache',
            description='Clear a service cache to resolve stale data issues',
            target_type='service',
            parameters={'service_name': 'string', 'cache_name': 'string'},
            risk_level=1
        )
    ]
    
    for action in actions:
        engine.register_action(action)
    
    return engine