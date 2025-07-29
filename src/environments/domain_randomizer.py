"""
Domain Randomization Wrapper for Highway-env
Randomizes environment parameters to improve generalization.
"""

import gymnasium as gym
import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple


class DomainRandomizer(gym.Wrapper):
    """
    Wrapper that applies domain randomization to highway environments.
    Randomizes traffic density, vehicle behavior, weather conditions, and road layout.
    """
    
    def __init__(
        self,
        env: gym.Env,
        randomize_traffic: bool = True,
        randomize_weather: bool = True,
        randomize_behavior: bool = True,
        randomize_layout: bool = True,
        traffic_density_range: Tuple[int, int] = (10, 50),
        weather_conditions: Optional[List[str]] = None,
        aggressive_ratio_range: Tuple[float, float] = (0.1, 0.5),
        seed: Optional[int] = None
    ):
        """
        Initialize the domain randomizer.
        
        Args:
            env: The environment to wrap
            randomize_traffic: Whether to randomize traffic density
            randomize_weather: Whether to randomize weather conditions
            randomize_behavior: Whether to randomize vehicle behavior
            randomize_layout: Whether to randomize road layout
            traffic_density_range: Range of number of vehicles (min, max)
            weather_conditions: List of weather conditions to sample from
            aggressive_ratio_range: Range of aggressive vehicle ratio
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        
        self.randomize_traffic = randomize_traffic
        self.randomize_weather = randomize_weather
        self.randomize_behavior = randomize_behavior
        self.randomize_layout = randomize_layout
        
        self.traffic_density_range = traffic_density_range
        self.aggressive_ratio_range = aggressive_ratio_range
        
        # Default weather conditions
        if weather_conditions is None:
            self.weather_conditions = [
                "clear", "rain", "fog", "night", "dawn", "dusk"
            ]
        else:
            self.weather_conditions = weather_conditions
            
        # Vehicle behavior parameters
        self.behavior_params = {
            "aggressive": {
                "DISTANCE_WANTED": (2.0, 4.0),
                "TIME_WANTED": (0.5, 1.0),
                "DELTA": (2.0, 6.0),
                "POLITENESS": (0.0, 0.3),
            },
            "normal": {
                "DISTANCE_WANTED": (3.0, 6.0),
                "TIME_WANTED": (1.0, 2.0),
                "DELTA": (3.0, 5.0),
                "POLITENESS": (0.3, 0.7),
            },
            "defensive": {
                "DISTANCE_WANTED": (5.0, 8.0),
                "TIME_WANTED": (1.5, 3.0),
                "DELTA": (2.0, 4.0),
                "POLITENESS": (0.7, 1.0),
            }
        }
        
        # Road layout parameters
        self.layout_params = {
            "lanes_count": [2, 3, 4],
            "initial_lane_id": None,  # Will be set based on lanes_count
            "vehicles_density": None,  # Will be set from traffic_density_range
            "controlled_vehicles": 1,
        }
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def _randomize_traffic_density(self) -> int:
        """Randomize the number of vehicles on the road."""
        return random.randint(*self.traffic_density_range)
    
    def _randomize_weather(self) -> str:
        """Randomize weather conditions."""
        return random.choice(self.weather_conditions)
    
    def _randomize_vehicle_behavior(self) -> Dict[str, Any]:
        """Randomize vehicle behavior parameters."""
        # Choose behavior types
        aggressive_ratio = random.uniform(*self.aggressive_ratio_range)
        defensive_ratio = random.uniform(0.1, 0.4)
        normal_ratio = 1.0 - aggressive_ratio - defensive_ratio
        
        behavior_config = {
            "behavior_ratios": {
                "aggressive": aggressive_ratio,
                "normal": normal_ratio,
                "defensive": defensive_ratio,
            }
        }
        
        # Randomize specific parameters for each behavior type
        for behavior_type, params in self.behavior_params.items():
            behavior_config[f"{behavior_type}_params"] = {}
            for param, (min_val, max_val) in params.items():
                behavior_config[f"{behavior_type}_params"][param] = random.uniform(min_val, max_val)
                
        return behavior_config
    
    def _randomize_layout(self) -> Dict[str, Any]:
        """Randomize road layout parameters."""
        lanes_count = random.choice(self.layout_params["lanes_count"])
        
        layout_config = {
            "lanes_count": lanes_count,
            "initial_lane_id": random.randint(0, lanes_count - 1),
            "right_lane_reward": random.uniform(0.0, 0.3),
            "lane_change_reward": random.uniform(-0.5, 0.5),
        }
        
        return layout_config
    
    def _apply_weather_effects(self, weather: str) -> Dict[str, Any]:
        """Apply visual and behavioral effects for different weather conditions."""
        effects = {}
        
        if weather == "rain":
            effects.update({
                "visibility_range": random.uniform(80, 120),
                "road_friction": random.uniform(0.6, 0.8),
                "reaction_time_multiplier": random.uniform(1.2, 1.5),
            })
        elif weather == "fog":
            effects.update({
                "visibility_range": random.uniform(40, 80),
                "road_friction": random.uniform(0.7, 0.9),
                "reaction_time_multiplier": random.uniform(1.1, 1.3),
            })
        elif weather == "night":
            effects.update({
                "visibility_range": random.uniform(60, 100),
                "road_friction": random.uniform(0.8, 1.0),
                "reaction_time_multiplier": random.uniform(1.0, 1.2),
            })
        elif weather in ["dawn", "dusk"]:
            effects.update({
                "visibility_range": random.uniform(100, 140),
                "road_friction": random.uniform(0.9, 1.0),
                "reaction_time_multiplier": random.uniform(1.0, 1.1),
            })
        else:  # clear
            effects.update({
                "visibility_range": random.uniform(120, 200),
                "road_friction": 1.0,
                "reaction_time_multiplier": 1.0,
            })
            
        return effects
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with randomized parameters."""
        # Build randomized configuration
        config_update = {}
        
        # Randomize traffic density
        if self.randomize_traffic:
            vehicles_count = self._randomize_traffic_density()
            config_update["vehicles_count"] = vehicles_count
            
        # Randomize weather
        if self.randomize_weather:
            weather = self._randomize_weather()
            weather_effects = self._apply_weather_effects(weather)
            config_update.update(weather_effects)
            config_update["weather"] = weather
            
        # Randomize vehicle behavior
        if self.randomize_behavior:
            behavior_config = self._randomize_vehicle_behavior()
            config_update.update(behavior_config)
            
        # Randomize layout
        if self.randomize_layout:
            layout_config = self._randomize_layout()
            config_update.update(layout_config)
            
        # Apply randomization to environment
        if hasattr(self.env, 'config') and config_update:
            self.env.config.update(config_update)
            if hasattr(self.env, 'configure'):
                self.env.configure(self.env.config)
                
        # Reset with randomized configuration
        obs, info = self.env.reset(**kwargs)
        
        # Add randomization info to info dict
        info["domain_randomization"] = config_update
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        return self.env.step(action)


class MultiScenarioRandomizer(DomainRandomizer):
    """
    Extended domain randomizer that switches between different scenario types.
    """
    
    def __init__(
        self,
        env: gym.Env,
        scenario_types: List[str] = None,
        scenario_weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize multi-scenario randomizer.
        
        Args:
            env: The environment to wrap
            scenario_types: List of scenario types to switch between
            scenario_weights: Probability weights for each scenario type
            **kwargs: Additional arguments for DomainRandomizer
        """
        super().__init__(env, **kwargs)
        
        if scenario_types is None:
            self.scenario_types = [
                "highway", "merge", "roundabout", "intersection", "parking"
            ]
        else:
            self.scenario_types = scenario_types
            
        self.scenario_weights = scenario_weights
        
        # Scenario-specific configurations
        self.scenario_configs = {
            "highway": {
                "lanes_count": [3, 4],
                "vehicles_count": (20, 40),
                "duration": 40,
            },
            "merge": {
                "lanes_count": [2, 3],
                "vehicles_count": (15, 30),
                "duration": 30,
            },
            "roundabout": {
                "lanes_count": [1, 2],
                "vehicles_count": (8, 20),
                "duration": 25,
            },
            "intersection": {
                "lanes_count": [2, 3],
                "vehicles_count": (10, 25),
                "duration": 20,
            },
            "parking": {
                "lanes_count": [1, 2],
                "vehicles_count": (5, 15),
                "duration": 30,
            }
        }
    
    def _select_scenario(self) -> str:
        """Select a random scenario type."""
        if self.scenario_weights:
            return np.random.choice(self.scenario_types, p=self.scenario_weights)
        else:
            return random.choice(self.scenario_types)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with randomized scenario and parameters."""
        # Select scenario
        scenario = self._select_scenario()
        scenario_config = self.scenario_configs.get(scenario, {})
        
        # Update traffic density range for this scenario
        if "vehicles_count" in scenario_config:
            self.traffic_density_range = scenario_config["vehicles_count"]
            
        # Call parent reset
        obs, info = super().reset(**kwargs)
        
        # Add scenario info
        info["scenario_type"] = scenario
        info["scenario_config"] = scenario_config
        
        return obs, info