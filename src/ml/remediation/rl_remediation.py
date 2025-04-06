"""
Reinforcement Learning-based Remediation
==================================

This module provides functionality for remediating issues using reinforcement learning.
"""

import logging
import numpy as np
import gym
from gym import spaces
import random
import pickle
import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import from existing modules
from src.remediation.remediation_engine import RemediationAction, RemediationResult, RemediationStatus
from src.detection import Issue, IssueType
from src.anomaly import AnomalyScore
from src.ml.anomaly.metric_anomaly import MetricAnomalyScore
from src.ml.root_cause.ml_root_cause import RootCauseScore

logger = logging.getLogger(__name__)

class RemediationEnvironment(gym.Env):
    """
    Reinforcement learning environment for remediation actions.
    
    This environment simulates the effects of remediation actions on a system
    and provides rewards based on the success of the actions.
    """
    
    def __init__(self, actions: List[RemediationAction], 
                state_size: int = 10, 
                max_steps: int = 5):
        """
        Initialize the remediation environment.
        
        Args:
            actions: List of available remediation actions
            state_size: Size of the state vector
            max_steps: Maximum number of steps per episode
        """
        super().__init__()
        
        self.actions = actions
        self.state_size = state_size
        self.max_steps = max_steps
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_size,), dtype=np.float32
        )
        
        # Initialize state
        self.state = np.zeros(state_size, dtype=np.float32)
        self.steps = 0
        self.episode_reward = 0
        self.history = []
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial state
        """
        # Reset state to random values
        self.state = np.random.uniform(low=0.3, high=0.7, size=self.state_size).astype(np.float32)
        
        # Add some anomalies to simulate issues
        num_anomalies = random.randint(1, 3)
        anomaly_indices = random.sample(range(self.state_size), num_anomalies)
        
        for idx in anomaly_indices:
            # Set to a high value to indicate an anomaly
            self.state[idx] = random.uniform(0.8, 1.0)
        
        self.steps = 0
        self.episode_reward = 0
        self.history = []
        
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment by executing an action.
        
        Args:
            action: Index of the action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if action < 0 or action >= len(self.actions):
            raise ValueError(f"Invalid action: {action}")
        
        # Get the selected action
        remediation_action = self.actions[action]
        
        # Simulate the effect of the action on the state
        next_state = self._apply_action_effect(action)
        
        # Calculate reward
        reward = self._calculate_reward(next_state)
        
        # Update state
        self.state = next_state
        
        # Increment step counter
        self.steps += 1
        
        # Check if episode is done
        done = self.steps >= self.max_steps or np.all(next_state < 0.7)
        
        # Update episode reward
        self.episode_reward += reward
        
        # Store history
        self.history.append({
            'action': action,
            'action_name': remediation_action.name,
            'state': self.state.copy(),
            'reward': reward,
            'done': done
        })
        
        # Return step information
        info = {
            'action_name': remediation_action.name,
            'episode_reward': self.episode_reward,
            'steps': self.steps
        }
        
        return next_state, reward, done, info
    
    def _apply_action_effect(self, action: int) -> np.ndarray:
        """
        Apply the effect of an action on the state.
        
        Args:
            action: Index of the action to apply
            
        Returns:
            New state after applying the action
        """
        # Get the selected action
        remediation_action = self.actions[action]
        
        # Create a copy of the current state
        next_state = self.state.copy()
        
        # Simulate the effect of the action
        # This is a simplified simulation - in a real system, the effect would depend on the actual action
        
        # Find the indices of the highest values (anomalies)
        anomaly_indices = np.argsort(next_state)[-3:]
        
        # Apply different effects based on the action
        if 'restart' in remediation_action.name.lower():
            # Restart actions have a high chance of fixing the issue
            for idx in anomaly_indices:
                if next_state[idx] > 0.7:  # If it's an anomaly
                    # 80% chance of fixing the issue
                    if random.random() < 0.8:
                        next_state[idx] = random.uniform(0.3, 0.6)
                    else:
                        # 20% chance of making it slightly better
                        next_state[idx] = max(0.7, next_state[idx] * 0.9)
        
        elif 'scale' in remediation_action.name.lower():
            # Scaling actions are good for load-related issues
            for idx in anomaly_indices:
                if next_state[idx] > 0.7:  # If it's an anomaly
                    # 60% chance of fixing the issue
                    if random.random() < 0.6:
                        next_state[idx] = random.uniform(0.3, 0.6)
                    else:
                        # 40% chance of making it slightly better
                        next_state[idx] = max(0.7, next_state[idx] * 0.95)
        
        else:
            # Other actions have a moderate chance of fixing the issue
            for idx in anomaly_indices:
                if next_state[idx] > 0.7:  # If it's an anomaly
                    # 40% chance of fixing the issue
                    if random.random() < 0.4:
                        next_state[idx] = random.uniform(0.3, 0.6)
                    else:
                        # 60% chance of making it slightly better
                        next_state[idx] = max(0.7, next_state[idx] * 0.97)
        
        # Add some random noise to simulate real-world variability
        noise = np.random.normal(0, 0.05, self.state_size)
        next_state += noise
        
        # Ensure values are within bounds
        next_state = np.clip(next_state, 0, 1)
        
        return next_state
    
    def _calculate_reward(self, next_state: np.ndarray) -> float:
        """
        Calculate the reward for transitioning to a new state.
        
        Args:
            next_state: New state after applying an action
            
        Returns:
            Reward value
        """
        # Calculate the improvement in the state
        # Lower values are better (fewer anomalies)
        current_anomaly_score = np.sum(self.state > 0.7)
        next_anomaly_score = np.sum(next_state > 0.7)
        
        # Reward for reducing anomalies
        anomaly_reduction_reward = (current_anomaly_score - next_anomaly_score) * 10
        
        # Penalty for each remaining anomaly
        anomaly_penalty = next_anomaly_score * -2
        
        # Small penalty for taking any action (to encourage efficiency)
        action_penalty = -1
        
        # Bonus for resolving all anomalies
        resolution_bonus = 20 if next_anomaly_score == 0 else 0
        
        # Combine rewards
        reward = anomaly_reduction_reward + anomaly_penalty + action_penalty + resolution_bonus
        
        return reward
    
    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass

class RLRemediationEngine:
    """
    Reinforcement learning-based engine for remediating issues.
    
    This class provides functionality for using reinforcement learning
    to select and execute remediation actions.
    """
    
    def __init__(self, actions: List[RemediationAction]):
        """
        Initialize the RL remediation engine.
        
        Args:
            actions: List of available remediation actions
        """
        self.actions = actions
        self.model = None
        self.env = None
        self.state_size = 10
        self.model_dir = Path("models/rl_remediation")
        self.results = []
        self.max_results = 1000
        self.callbacks = []
        
        # Create the environment
        self._create_environment()
    
    def _create_environment(self) -> None:
        """Create the reinforcement learning environment."""
        self.env = RemediationEnvironment(self.actions, self.state_size)
        
        # Wrap the environment in a DummyVecEnv for compatibility with stable-baselines
        self.env = DummyVecEnv([lambda: self.env])
    
    def train(self, total_timesteps: int = 10000) -> None:
        """
        Train the reinforcement learning model.
        
        Args:
            total_timesteps: Total number of timesteps to train for
        """
        if not self.env:
            self._create_environment()
        
        # Create a PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./tensorboard/rl_remediation/"
        )
        
        # Train the model
        logger.info(f"Training RL remediation model for {total_timesteps} timesteps")
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save the model
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(f"{self.model_dir}/ppo_remediation")
        
        logger.info("Trained and saved RL remediation model")
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the model file (default: self.model_dir/ppo_remediation.zip)
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        path = path or f"{self.model_dir}/ppo_remediation.zip"
        
        try:
            # Create the environment if it doesn't exist
            if not self.env:
                self._create_environment()
            
            # Load the model
            self.model = PPO.load(path, env=self.env)
            
            logger.info(f"Loaded RL remediation model from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RL remediation model: {e}")
            return False
    
    def add_callback(self, callback):
        """
        Add a callback function to be called for each remediation result.
        
        Args:
            callback: Function to call with the remediation result
        """
        self.callbacks.append(callback)
    
    def _extract_state_from_issues(self, issues: List[Issue], 
                                 anomalies: List[AnomalyScore],
                                 metric_anomalies: List[MetricAnomalyScore],
                                 root_causes: List[RootCauseScore]) -> np.ndarray:
        """
        Extract a state vector from issues, anomalies, and root causes.
        
        Args:
            issues: List of issues
            anomalies: List of anomalies
            metric_anomalies: List of metric anomalies
            root_causes: List of root causes
            
        Returns:
            State vector
        """
        # Initialize state vector
        state = np.zeros(self.state_size, dtype=np.float32)
        
        # Fill in state based on issues, anomalies, and root causes
        # This is a simplified mapping - in a real system, you would have a more sophisticated mapping
        
        # State indices:
        # 0: Overall system health
        # 1: Number of issues
        # 2: Average issue severity
        # 3: Number of anomalies
        # 4: Average anomaly score
        # 5: Number of metric anomalies
        # 6: Average metric anomaly score
        # 7: Number of root causes
        # 8: Average root cause score
        # 9: Time since last remediation (normalized)
        
        # Overall system health (inverse of the average of all scores)
        all_scores = []
        if issues:
            all_scores.extend([i.severity / 5.0 for i in issues])
        if anomalies:
            all_scores.extend([a.score for a in anomalies])
        if metric_anomalies:
            all_scores.extend([a.score for a in metric_anomalies])
        if root_causes:
            all_scores.extend([r.score for r in root_causes])
        
        if all_scores:
            state[0] = min(1.0, sum(all_scores) / len(all_scores))
        else:
            state[0] = 0.0
        
        # Number of issues (normalized)
        state[1] = min(1.0, len(issues) / 10.0)
        
        # Average issue severity (normalized)
        if issues:
            state[2] = sum(i.severity for i in issues) / (len(issues) * 5.0)
        
        # Number of anomalies (normalized)
        state[3] = min(1.0, len(anomalies) / 10.0)
        
        # Average anomaly score
        if anomalies:
            state[4] = sum(a.score for a in anomalies) / len(anomalies)
        
        # Number of metric anomalies (normalized)
        state[5] = min(1.0, len(metric_anomalies) / 10.0)
        
        # Average metric anomaly score
        if metric_anomalies:
            state[6] = sum(a.score for a in metric_anomalies) / len(metric_anomalies)
        
        # Number of root causes (normalized)
        state[7] = min(1.0, len(root_causes) / 5.0)
        
        # Average root cause score
        if root_causes:
            state[8] = sum(r.score for r in root_causes) / len(root_causes)
        
        # Time since last remediation (normalized to 0-1, where 1 means a long time)
        # This is a placeholder - in a real system, you would track the actual time
        state[9] = 0.5
        
        return state
    
    def select_action(self, issues: List[Issue], 
                     anomalies: List[AnomalyScore],
                     metric_anomalies: List[MetricAnomalyScore],
                     root_causes: List[RootCauseScore]) -> Optional[RemediationAction]:
        """
        Select a remediation action based on the current state.
        
        Args:
            issues: List of issues
            anomalies: List of anomalies
            metric_anomalies: List of metric anomalies
            root_causes: List of root causes
            
        Returns:
            Selected remediation action, or None if no action should be taken
        """
        if not self.model:
            logger.warning("No RL model available for action selection")
            return None
        
        # Extract state from issues, anomalies, and root causes
        state = self._extract_state_from_issues(issues, anomalies, metric_anomalies, root_causes)
        
        # Use the model to predict the best action
        action, _ = self.model.predict(state, deterministic=True)
        
        # Get the selected action
        return self.actions[action]
    
    def execute_action(self, action: RemediationAction, 
                      issues: List[Issue] = None,
                      anomalies: List[AnomalyScore] = None,
                      metric_anomalies: List[MetricAnomalyScore] = None,
                      root_causes: List[RootCauseScore] = None) -> RemediationResult:
        """
        Execute a remediation action.
        
        Args:
            action: Remediation action to execute
            issues: List of issues that triggered the action
            anomalies: List of anomalies that triggered the action
            metric_anomalies: List of metric anomalies that triggered the action
            root_causes: List of root causes that triggered the action
            
        Returns:
            Result of the remediation action
        """
        # Execute the action
        success, output = action.execute()
        
        # Create a result
        status = RemediationStatus.SUCCESS if success else RemediationStatus.FAILURE
        result = RemediationResult(
            action=action,
            status=status,
            output=output,
            timestamp=datetime.now(),
            issue=issues[0] if issues else None,
            anomaly=anomalies[0] if anomalies else None,
            retry_count=0
        )
        
        # Add to results
        self.results.append(result)
        
        # Trim results if needed
        if len(self.results) > self.max_results:
            self.results = self.results[-self.max_results:]
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in remediation callback: {e}")
        
        return result
    
    def remediate(self, issues: List[Issue] = None,
                anomalies: List[AnomalyScore] = None,
                metric_anomalies: List[MetricAnomalyScore] = None,
                root_causes: List[RootCauseScore] = None) -> Optional[RemediationResult]:
        """
        Select and execute a remediation action based on the current state.
        
        Args:
            issues: List of issues
            anomalies: List of anomalies
            metric_anomalies: List of metric anomalies
            root_causes: List of root causes
            
        Returns:
            Result of the remediation action, or None if no action was taken
        """
        # Initialize empty lists if None
        issues = issues or []
        anomalies = anomalies or []
        metric_anomalies = metric_anomalies or []
        root_causes = root_causes or []
        
        # Select an action
        action = self.select_action(issues, anomalies, metric_anomalies, root_causes)
        
        if not action:
            return None
        
        # Execute the action
        return self.execute_action(action, issues, anomalies, metric_anomalies, root_causes)
