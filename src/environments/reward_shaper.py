"""
Reward Shaper for Highway-env
Implements custom reward functions for autonomous driving training.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable


class RewardShaper(gym.Wrapper):
    """
    Wrapper that implements custom reward shaping for autonomous driving.
    Provides dense rewards that guide the agent towards safe and efficient driving.
    """
    
    def __init__(
        self,
        env: gym.Env,
        collision_reward: float = -100.0,
        lane_keeping_reward: float = 1.0,
        speed_efficiency_reward: float = 0.5,
        rule_violation_reward: float = -5.0,
        lane_change_reward: float = 0.0,
        right_lane_reward: float = 0.1,
        safe_distance_reward: float = 0.2,
        smooth_driving_reward: float = 0.1,
        goal_achievement_reward: float = 10.0,
        use_shaped_rewards: bool = True,
        normalize_rewards: bool = True,
        reward_clipping: Optional[Tuple[float, float]] = (-10, 10)
    ):
        """
        Initialize the reward shaper.
        
        Args:
            env: The environment to wrap
            collision_reward: Reward for collisions (negative)
            lane_keeping_reward: Reward for staying in lane
            speed_efficiency_reward: Reward for maintaining efficient speed
            rule_violation_reward: Reward for traffic rule violations (negative)
            lane_change_reward: Reward for lane changes
            right_lane_reward: Reward for staying in right lane
            safe_distance_reward: Reward for maintaining safe following distance
            smooth_driving_reward: Reward for smooth acceleration/deceleration
            goal_achievement_reward: Reward for reaching goals
            use_shaped_rewards: Whether to use dense reward shaping
            normalize_rewards: Whether to normalize rewards
            reward_clipping: Optional reward clipping bounds
        """
        super().__init__(env)
        
        self.collision_reward = collision_reward
        self.lane_keeping_reward = lane_keeping_reward
        self.speed_efficiency_reward = speed_efficiency_reward
        self.rule_violation_reward = rule_violation_reward
        self.lane_change_reward = lane_change_reward
        self.right_lane_reward = right_lane_reward
        self.safe_distance_reward = safe_distance_reward
        self.smooth_driving_reward = smooth_driving_reward
        self.goal_achievement_reward = goal_achievement_reward
        self.use_shaped_rewards = use_shaped_rewards
        self.normalize_rewards = normalize_rewards
        self.reward_clipping = reward_clipping
        
        # State tracking for reward computation
        self.previous_state = None
        self.episode_rewards = []
        self.reward_components = {
            "collision": 0.0,
            "lane_keeping": 0.0,
            "speed_efficiency": 0.0,
            "rule_violation": 0.0,
            "lane_change": 0.0,
            "right_lane": 0.0,
            "safe_distance": 0.0,
            "smooth_driving": 0.0,
            "goal_achievement": 0.0
        }
        
        # Reward normalization parameters
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and reward tracking."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset state tracking
        self.previous_state = self._extract_state(obs, info)
        self.episode_rewards = []
        self.reward_components = {k: 0.0 for k in self.reward_components.keys()}
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment and apply reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract current state
        current_state = self._extract_state(obs, info)
        
        # Compute shaped reward
        if self.use_shaped_rewards:
            shaped_reward = self._compute_shaped_reward(
                action, current_state, self.previous_state, info, terminated, truncated
            )
        else:
            shaped_reward = reward
            
        # Apply normalization and clipping
        final_reward = self._process_reward(shaped_reward)
        
        # Update state tracking
        self.previous_state = current_state
        self.episode_rewards.append(final_reward)
        
        # Add reward information to info
        info["reward_components"] = self.reward_components.copy()
        info["original_reward"] = reward
        info["shaped_reward"] = shaped_reward
        info["final_reward"] = final_reward
        
        return obs, final_reward, terminated, truncated, info
    
    def _extract_state(self, obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant state information for reward computation."""
        state = {
            "observation": obs,
            "crashed": info.get("crashed", False),
            "speed": info.get("speed", 0.0),
            "on_road": info.get("on_road", True),
            "action": info.get("action", None)
        }
        
        # Extract vehicle information if available
        if hasattr(self.env, 'vehicle') and self.env.vehicle:
            vehicle = self.env.vehicle
            state.update({
                "position": getattr(vehicle, 'position', [0, 0]),
                "speed": getattr(vehicle, 'speed', 0.0),
                "lane_index": getattr(vehicle, 'lane_index', [None, None, 0]),
                "heading": getattr(vehicle, 'heading', 0.0),
            })
            
        # Extract road information
        if hasattr(self.env, 'road') and self.env.road:
            road = self.env.road
            state.update({
                "vehicles_count": len(road.vehicles) if road.vehicles else 0,
                "nearby_vehicles": self._get_nearby_vehicles(state.get("position", [0, 0]))
            })
            
        return state
    
    def _compute_shaped_reward(
        self,
        action: int,
        current_state: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]],
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool
    ) -> float:
        """Compute the shaped reward based on driving behavior."""
        total_reward = 0.0
        
        # Reset reward components
        for key in self.reward_components.keys():
            self.reward_components[key] = 0.0
        
        # 1. Collision penalty
        if current_state["crashed"]:
            collision_reward = self.collision_reward
            self.reward_components["collision"] = collision_reward
            total_reward += collision_reward
        
        # 2. Lane keeping reward
        if current_state["on_road"]:
            lane_reward = self.lane_keeping_reward
            self.reward_components["lane_keeping"] = lane_reward
            total_reward += lane_reward
        else:
            # Off-road penalty
            violation_reward = self.rule_violation_reward
            self.reward_components["rule_violation"] = violation_reward
            total_reward += violation_reward
        
        # 3. Speed efficiency reward
        target_speed = 25.0  # Target speed in m/s
        speed = current_state.get("speed", 0.0)
        
        if 20.0 <= speed <= 30.0:  # Optimal speed range
            speed_reward = self.speed_efficiency_reward
        elif speed < 10.0:  # Too slow
            speed_reward = -0.2
        elif speed > 35.0:  # Too fast
            speed_reward = -0.3
        else:
            # Linear interpolation for intermediate speeds
            if speed < 20.0:
                speed_reward = self.speed_efficiency_reward * (speed - 10.0) / 10.0
            else:  # speed > 30.0
                speed_reward = self.speed_efficiency_reward * (35.0 - speed) / 5.0
        
        self.reward_components["speed_efficiency"] = speed_reward
        total_reward += speed_reward
        
        # 4. Lane change reward/penalty
        if previous_state and action in [0, 2]:  # Lane change actions
            # Check if lane change was successful and safe
            prev_lane = previous_state.get("lane_index", [None, None, 0])[2]
            curr_lane = current_state.get("lane_index", [None, None, 0])[2]
            
            if prev_lane != curr_lane:
                # Successful lane change
                lane_change_reward = self.lane_change_reward
                
                # Bonus for safe lane changes (no nearby vehicles)
                nearby_vehicles = current_state.get("nearby_vehicles", [])
                if len(nearby_vehicles) < 2:
                    lane_change_reward += 0.1
                    
            else:
                # Failed lane change (might indicate blocked lane)
                lane_change_reward = -0.1
                
            self.reward_components["lane_change"] = lane_change_reward
            total_reward += lane_change_reward
        
        # 5. Right lane preference
        lane_index = current_state.get("lane_index", [None, None, 0])[2]
        if lane_index == 0:  # Rightmost lane
            right_lane_reward = self.right_lane_reward
            self.reward_components["right_lane"] = right_lane_reward
            total_reward += right_lane_reward
        
        # 6. Safe following distance
        safe_distance_reward = self._compute_safe_distance_reward(current_state)
        self.reward_components["safe_distance"] = safe_distance_reward
        total_reward += safe_distance_reward
        
        # 7. Smooth driving reward
        if previous_state:
            smooth_reward = self._compute_smooth_driving_reward(
                current_state, previous_state, action
            )
            self.reward_components["smooth_driving"] = smooth_reward
            total_reward += smooth_reward
        
        # 8. Goal achievement reward
        if terminated and not current_state["crashed"]:
            goal_reward = self.goal_achievement_reward
            self.reward_components["goal_achievement"] = goal_reward
            total_reward += goal_reward
        
        return total_reward
    
    def _compute_safe_distance_reward(self, state: Dict[str, Any]) -> float:
        """Compute reward for maintaining safe following distance."""
        nearby_vehicles = state.get("nearby_vehicles", [])
        
        if not nearby_vehicles:
            return self.safe_distance_reward  # Bonus for clear road
        
        min_distance = min(v["distance"] for v in nearby_vehicles)
        safe_distance_threshold = 15.0  # meters
        
        if min_distance >= safe_distance_threshold:
            return self.safe_distance_reward
        elif min_distance < 5.0:  # Very close - dangerous
            return -0.5
        else:
            # Linear interpolation
            return self.safe_distance_reward * (min_distance - 5.0) / 10.0
    
    def _compute_smooth_driving_reward(
        self,
        current_state: Dict[str, Any],
        previous_state: Dict[str, Any],
        action: int
    ) -> float:
        """Compute reward for smooth acceleration and steering."""
        current_speed = current_state.get("speed", 0.0)
        previous_speed = previous_state.get("speed", 0.0)
        
        # Compute acceleration
        acceleration = current_speed - previous_speed
        
        # Penalize harsh acceleration/deceleration
        if abs(acceleration) > 5.0:  # Harsh acceleration/braking
            return -0.2
        elif abs(acceleration) < 1.0:  # Smooth driving
            return self.smooth_driving_reward
        else:
            # Linear penalty for moderate acceleration
            return self.smooth_driving_reward * (5.0 - abs(acceleration)) / 4.0
    
    def _get_nearby_vehicles(self, position: List[float], radius: float = 50.0) -> List[Dict[str, Any]]:
        """Get information about nearby vehicles."""
        nearby_vehicles = []
        
        if not (hasattr(self.env, 'road') and self.env.road and self.env.road.vehicles):
            return nearby_vehicles
        
        ego_position = np.array(position)
        
        for vehicle in self.env.road.vehicles:
            if hasattr(vehicle, 'position'):
                vehicle_position = np.array(vehicle.position)
                distance = np.linalg.norm(vehicle_position - ego_position)
                
                if distance <= radius and distance > 0:  # Exclude ego vehicle
                    nearby_vehicles.append({
                        "distance": distance,
                        "position": vehicle.position,
                        "speed": getattr(vehicle, 'speed', 0.0),
                        "lane_index": getattr(vehicle, 'lane_index', [None, None, 0])[2]
                    })
        
        return nearby_vehicles
    
    def _process_reward(self, reward: float) -> float:
        """Apply normalization and clipping to the reward."""
        processed_reward = reward
        
        # Apply normalization
        if self.normalize_rewards:
            self.reward_history.append(reward)
            
            # Update running statistics
            if len(self.reward_history) > 100:
                recent_rewards = self.reward_history[-100:]
                self.reward_mean = np.mean(recent_rewards)
                self.reward_std = np.std(recent_rewards) + 1e-8
                
                # Normalize
                processed_reward = (reward - self.reward_mean) / self.reward_std
        
        # Apply clipping
        if self.reward_clipping:
            processed_reward = np.clip(
                processed_reward,
                self.reward_clipping[0],
                self.reward_clipping[1]
            )
        
        return processed_reward
    
    def get_episode_reward_summary(self) -> Dict[str, float]:
        """Get summary of rewards for the current episode."""
        if not self.episode_rewards:
            return {}
        
        return {
            "total_reward": sum(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": min(self.episode_rewards),
            "max_reward": max(self.episode_rewards),
            "episode_length": len(self.episode_rewards)
        }


class AdaptiveRewardShaper(RewardShaper):
    """
    Extended reward shaper that adapts reward weights based on agent performance.
    """
    
    def __init__(self, *args, adaptation_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.initial_weights = {
            "collision": self.collision_reward,
            "lane_keeping": self.lane_keeping_reward,
            "speed_efficiency": self.speed_efficiency_reward,
            "safe_distance": self.safe_distance_reward
        }
        
    def _adapt_reward_weights(self, episode_performance: Dict[str, float]):
        """Adapt reward weights based on episode performance."""
        # Adapt collision weight based on crash rate
        if episode_performance.get("crash_rate", 0) > 0.1:
            self.collision_reward *= (1 + self.adaptation_rate)
            
        # Adapt speed weight based on efficiency
        if episode_performance.get("average_speed", 0) < 15:
            self.speed_efficiency_reward *= (1 + self.adaptation_rate)
            
        # Ensure weights don't become too extreme
        self.collision_reward = max(self.collision_reward, -200.0)
        self.speed_efficiency_reward = min(self.speed_efficiency_reward, 2.0)
    
    def reset(self, **kwargs):
        """Reset with potential weight adaptation."""
        # Adapt weights based on recent performance
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            avg_performance = {
                "crash_rate": np.mean([p.get("crashed", 0) for p in recent_performance]),
                "average_speed": np.mean([p.get("average_speed", 20) for p in recent_performance])
            }
            self._adapt_reward_weights(avg_performance)
        
        return super().reset(**kwargs)