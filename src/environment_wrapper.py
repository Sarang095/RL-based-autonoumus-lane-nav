"""
Environment wrapper for highway-env with vision-based observations and domain randomization
"""
import gymnasium as gym
import highway_env
import numpy as np
import cv2
from typing import Any, Dict, Tuple, Optional
import random
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation

from .config import Config


class DomainRandomizationWrapper(gym.Wrapper):
    """Wrapper for domain randomization in highway-env"""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env)
        self.randomization_config = config
        self.randomize_prob = config.get('randomize_probability', 0.3)
        
    def reset(self, **kwargs):
        # Apply domain randomization before reset
        if random.random() < self.randomize_prob:
            self._randomize_environment()
        
        obs, info = self.env.reset(**kwargs)
        return self._add_observation_noise(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_observation_noise(obs), reward, terminated, truncated, info
    
    def _randomize_environment(self):
        """Apply domain randomization to the environment"""
        config = self.env.unwrapped.config
        
        # Randomize traffic density
        density_range = self.randomization_config['traffic_density_range']
        new_density = random.uniform(*density_range)
        config['vehicles_density'] = new_density
        
        # Randomize vehicle speeds
        speed_variance = self.randomization_config['vehicle_speed_variance']
        for vehicle_class in ['IDMVehicle', 'AggressiveVehicle', 'DefensiveVehicle']:
            if hasattr(config, 'vehicles_types'):
                # Add speed variance to vehicle types
                pass
    
    def _add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observations to simulate sensor noise"""
        noise_std = self.randomization_config.get('observation_noise_std', 0.05)
        noise = np.random.normal(0, noise_std, obs.shape)
        return np.clip(obs + noise, 0, 1)


class MultiAgentWrapper(gym.Wrapper):
    """Wrapper to configure multi-agent scenarios"""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        super().__init__(env)
        self.multi_agent_config = config
        
    def reset(self, **kwargs):
        self._configure_multi_agent()
        return self.env.reset(**kwargs)
    
    def _configure_multi_agent(self):
        """Configure multi-agent scenario with diverse vehicle behaviors"""
        config = self.env.unwrapped.config
        
        # Set vehicle types with different behaviors
        aggressive_ratio = self.multi_agent_config['aggressive_vehicles_ratio']
        defensive_ratio = self.multi_agent_config['defensive_vehicles_ratio']
        
        # Configure vehicles_count based on scenario
        max_agents = self.multi_agent_config['max_agents']
        vehicles_count = random.randint(10, max_agents)
        config['vehicles_count'] = vehicles_count


class VisionObservationWrapper(gym.ObservationWrapper):
    """Wrapper to ensure consistent vision-based observations"""
    
    def __init__(self, env: gym.Env, observation_shape: Tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.observation_shape = observation_shape
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(*observation_shape, 1),  # Grayscale
            dtype=np.uint8
        )
    
    def observation(self, obs):
        """Process observation to ensure consistent shape and format"""
        # Handle different observation formats
        if len(obs.shape) == 4:  # Already processed by frame stack
            return obs
        elif len(obs.shape) == 3 and obs.shape[-1] == 3:  # RGB image
            # Convert to grayscale
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 3 and obs.shape[-1] == 1:  # Already grayscale with channel
            obs = obs.squeeze(-1)  # Remove channel dimension for processing
        
        # Resize to target shape if needed
        if obs.shape[:2] != self.observation_shape:
            obs = cv2.resize(obs, self.observation_shape)
        
        # Ensure uint8 and add channel dimension
        obs = obs.astype(np.uint8)
        if len(obs.shape) == 2:
            obs = np.expand_dims(obs, axis=-1)
        
        return obs


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper for custom reward shaping"""
    
    def __init__(self, env: gym.Env, reward_config: Dict[str, float]):
        super().__init__(env)
        self.reward_config = reward_config
        self.prev_obs = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply custom reward shaping
        shaped_reward = self._shape_reward(obs, reward, terminated, info)
        
        self.prev_obs = obs
        return obs, shaped_reward, terminated, truncated, info
    
    def _shape_reward(self, obs, original_reward, terminated, info):
        """Apply custom reward shaping based on config"""
        reward = original_reward
        
        # Collision penalty
        if info.get('crashed', False):
            reward += self.reward_config['collision_penalty']
        
        # Lane keeping reward
        if hasattr(self.env.unwrapped, 'vehicle') and self.env.unwrapped.vehicle:
            vehicle = self.env.unwrapped.vehicle
            if hasattr(vehicle, 'lane_index') and hasattr(vehicle, 'position'):
                # Simple lane keeping check
                lane_center_offset = abs(vehicle.position[1] % 4)  # Assuming 4m lane width
                if lane_center_offset < 1.5:  # Within lane boundaries
                    reward += self.reward_config['lane_keeping_reward']
        
        # Speed efficiency reward
        if hasattr(self.env.unwrapped, 'vehicle') and self.env.unwrapped.vehicle:
            speed = self.env.unwrapped.vehicle.speed
            target_speed = 30  # km/h
            if 25 <= speed <= 35:  # Within reasonable speed range
                reward += self.reward_config['speed_efficiency_reward']
        
        # Time penalty (encourage faster completion)
        reward += self.reward_config['time_penalty']
        
        return reward


def create_environment(env_name: str = 'highway', 
                      enable_domain_randomization: bool = True,
                      enable_multi_agent: bool = True,
                      enable_reward_shaping: bool = True) -> gym.Env:
    """
    Create and configure the highway-env environment with all wrappers
    
    Args:
        env_name: Name of the environment ('highway', 'roundabout', 'parking')
        enable_domain_randomization: Whether to enable domain randomization
        enable_multi_agent: Whether to enable multi-agent scenarios
        enable_reward_shaping: Whether to enable custom reward shaping
        
    Returns:
        Configured environment
    """
    # Create base environment
    env_id = f'{env_name}-v0'
    
    # Get environment-specific config
    env_config = Config.get_env_config(env_name)
    
    # Create environment with visual observations
    env = gym.make(env_id, render_mode='rgb_array')
    
    # Configure for visual observations with stack_size = 1 (we'll handle stacking separately)
    visual_config = {
        'observation': {
            'type': 'GrayscaleObservation',
            'observation_shape': env_config['observation']['observation_shape'],
            'stack_size': 1,  # Use 1 and handle stacking with wrapper
            'weights': env_config['observation']['weights'],
        },
        'vehicles_count': env_config.get('vehicles_count', 50),
        'duration': env_config.get('duration', 40),
        'vehicles_density': env_config.get('vehicles_density', 1),
        'action': env_config.get('action', {'type': 'ContinuousAction'}),
    }
    
    # Apply configuration
    if hasattr(env.unwrapped, 'configure'):
        env.unwrapped.configure(visual_config)
    
    # Apply wrappers in order
    if enable_reward_shaping:
        env = RewardShapingWrapper(env, Config.REWARD_CONFIG)
    
    if enable_multi_agent:
        env = MultiAgentWrapper(env, Config.MULTI_AGENT_CONFIG)
    
    if enable_domain_randomization:
        env = DomainRandomizationWrapper(env, Config.DOMAIN_RANDOMIZATION)
    
    # Frame stacking for temporal information
    stack_size = env_config['observation']['stack_size']
    if stack_size > 1:
        env = FrameStackObservation(env, stack_size)
    
    return env


if __name__ == "__main__":
    # Test environment creation
    env = create_environment('highway')
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Terminated: {terminated}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()