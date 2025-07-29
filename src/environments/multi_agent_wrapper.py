"""
Multi-Agent Wrapper for Highway-env
Implements multi-agent scenarios with diverse vehicle behaviors.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import highway_env


class MultiAgentWrapper(gym.Wrapper):
    """
    Wrapper that enhances highway environments with multi-agent complexity.
    Spawns multiple vehicles with different behavioral patterns.
    """
    
    def __init__(
        self,
        env: gym.Env,
        num_controlled_vehicles: int = 1,
        aggressive_vehicles_ratio: float = 0.3,
        defensive_vehicles_ratio: float = 0.2,
        random_vehicles_ratio: float = 0.1,
        enable_vehicle_interactions: bool = True,
        min_vehicles: int = 10,
        max_vehicles: int = 40
    ):
        """
        Initialize the multi-agent wrapper.
        
        Args:
            env: The environment to wrap
            num_controlled_vehicles: Number of vehicles controlled by the agent
            aggressive_vehicles_ratio: Ratio of aggressive vehicles
            defensive_vehicles_ratio: Ratio of defensive vehicles
            random_vehicles_ratio: Ratio of randomly behaving vehicles
            enable_vehicle_interactions: Whether to enable complex interactions
            min_vehicles: Minimum number of vehicles in the environment
            max_vehicles: Maximum number of vehicles in the environment
        """
        super().__init__(env)
        
        self.num_controlled_vehicles = num_controlled_vehicles
        self.aggressive_vehicles_ratio = aggressive_vehicles_ratio
        self.defensive_vehicles_ratio = defensive_vehicles_ratio
        self.random_vehicles_ratio = random_vehicles_ratio
        self.enable_vehicle_interactions = enable_vehicle_interactions
        self.min_vehicles = min_vehicles
        self.max_vehicles = max_vehicles
        
        # Vehicle behavior parameters
        self.behavior_configs = {
            "aggressive": {
                "DISTANCE_WANTED": 2.0,
                "TIME_WANTED": 0.8,
                "DELTA": 4.0,
                "POLITENESS": 0.1,
                "LANE_CHANGE_MIN_ACC_GAIN": 0.2,
                "LANE_CHANGE_MAX_BRAKING_IMPOSED": 2.0,
            },
            "defensive": {
                "DISTANCE_WANTED": 6.0,
                "TIME_WANTED": 2.5,
                "DELTA": 2.0,
                "POLITENESS": 0.8,
                "LANE_CHANGE_MIN_ACC_GAIN": 0.8,
                "LANE_CHANGE_MAX_BRAKING_IMPOSED": 0.5,
            },
            "normal": {
                "DISTANCE_WANTED": 4.0,
                "TIME_WANTED": 1.5,
                "DELTA": 3.0,
                "POLITENESS": 0.5,
                "LANE_CHANGE_MIN_ACC_GAIN": 0.5,
                "LANE_CHANGE_MAX_BRAKING_IMPOSED": 1.0,
            },
            "random": {
                "DISTANCE_WANTED": np.random.uniform(2.0, 6.0),
                "TIME_WANTED": np.random.uniform(0.5, 3.0),
                "DELTA": np.random.uniform(2.0, 5.0),
                "POLITENESS": np.random.uniform(0.0, 1.0),
            }
        }
        
    def _configure_multi_agent_environment(self):
        """Configure the environment for multi-agent scenarios."""
        # Calculate vehicle counts
        total_vehicles = np.random.randint(self.min_vehicles, self.max_vehicles + 1)
        num_aggressive = int(total_vehicles * self.aggressive_vehicles_ratio)
        num_defensive = int(total_vehicles * self.defensive_vehicles_ratio)
        num_random = int(total_vehicles * self.random_vehicles_ratio)
        num_normal = total_vehicles - num_aggressive - num_defensive - num_random - self.num_controlled_vehicles
        
        # Ensure non-negative counts
        num_normal = max(0, num_normal)
        
        # Build vehicle configuration
        vehicles_config = []
        
        # Add controlled vehicles
        for i in range(self.num_controlled_vehicles):
            vehicles_config.append({
                "type": "highway_env.vehicle.controlled_vehicle.ControlledVehicle",
                "lane_id": np.random.randint(0, 3),  # Random lane
                "position": [0, 0],  # Will be set by environment
                "speed": np.random.uniform(20, 30)
            })
        
        # Add aggressive vehicles
        for i in range(num_aggressive):
            vehicles_config.append({
                "type": "highway_env.vehicle.behavior.AggressiveVehicle",
                "lane_id": np.random.randint(0, 3),
                "position": [np.random.uniform(50, 200), 0],
                "speed": np.random.uniform(25, 35),
                **self.behavior_configs["aggressive"]
            })
        
        # Add defensive vehicles
        for i in range(num_defensive):
            vehicles_config.append({
                "type": "highway_env.vehicle.behavior.DefensiveVehicle", 
                "lane_id": np.random.randint(0, 3),
                "position": [np.random.uniform(50, 200), 0],
                "speed": np.random.uniform(15, 25),
                **self.behavior_configs["defensive"]
            })
        
        # Add normal vehicles
        for i in range(num_normal):
            vehicles_config.append({
                "type": "highway_env.vehicle.behavior.IDMVehicle",
                "lane_id": np.random.randint(0, 3),
                "position": [np.random.uniform(50, 200), 0],
                "speed": np.random.uniform(20, 30),
                **self.behavior_configs["normal"]
            })
        
        # Add random behavior vehicles
        for i in range(num_random):
            random_config = {k: np.random.uniform(*v) if isinstance(v, tuple) else v 
                           for k, v in self.behavior_configs["random"].items()}
            vehicles_config.append({
                "type": "highway_env.vehicle.behavior.IDMVehicle",
                "lane_id": np.random.randint(0, 3),
                "position": [np.random.uniform(50, 200), 0],
                "speed": np.random.uniform(15, 35),
                **random_config
            })
        
        # Update environment configuration
        config_update = {
            "vehicles_count": total_vehicles,
            "controlled_vehicles": self.num_controlled_vehicles,
            "vehicles_density": total_vehicles / 1000,  # Normalize by road length
            "initial_vehicle_spacing": max(10, 200 / total_vehicles),
            "spawn_probability": 0.02,  # Continuous spawning
            "vehicles_config": vehicles_config
        }
        
        return config_update
    
    def _add_interaction_complexity(self):
        """Add complex vehicle interactions and scenarios."""
        if not self.enable_vehicle_interactions:
            return {}
        
        # Add challenging scenarios
        scenarios = [
            "merge_challenge",
            "lane_change_pressure", 
            "traffic_jam",
            "highway_exit",
            "construction_zone"
        ]
        
        selected_scenario = np.random.choice(scenarios)
        
        scenario_configs = {
            "merge_challenge": {
                "merge_vehicles": True,
                "merge_probability": 0.3,
                "merge_aggressive": True
            },
            "lane_change_pressure": {
                "lane_change_frequency": "high",
                "cooperative_behavior": False
            },
            "traffic_jam": {
                "speed_limit_zones": [(100, 150, 15), (200, 250, 10)],
                "high_density_zones": [(80, 180)]
            },
            "highway_exit": {
                "exit_zones": [(180, 200)],
                "exit_probability": 0.4
            },
            "construction_zone": {
                "construction_zones": [(120, 160)],
                "lane_closures": [2],  # Close right lane
                "speed_reduction": 0.7
            }
        }
        
        return {
            "scenario_type": selected_scenario,
            **scenario_configs.get(selected_scenario, {})
        }
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with multi-agent configuration."""
        # Configure multi-agent environment
        multi_agent_config = self._configure_multi_agent_environment()
        interaction_config = self._add_interaction_complexity()
        
        # Combine configurations
        config_update = {**multi_agent_config, **interaction_config}
        
        # Apply configuration to environment
        if hasattr(self.env, 'config'):
            self.env.config.update(config_update)
            if hasattr(self.env, 'configure'):
                self.env.configure(self.env.config)
        
        # Reset environment
        obs, info = self.env.reset(**kwargs)
        
        # Add multi-agent info
        info["multi_agent_config"] = config_update
        info["num_vehicles"] = config_update.get("vehicles_count", 0)
        info["vehicle_behaviors"] = self._get_behavior_distribution(config_update.get("vehicles_count", 0))
        
        return obs, info
    
    def _get_behavior_distribution(self, total_vehicles: int) -> Dict[str, int]:
        """Get the distribution of vehicle behaviors."""
        num_aggressive = int(total_vehicles * self.aggressive_vehicles_ratio)
        num_defensive = int(total_vehicles * self.defensive_vehicles_ratio)
        num_random = int(total_vehicles * self.random_vehicles_ratio)
        num_normal = total_vehicles - num_aggressive - num_defensive - num_random - self.num_controlled_vehicles
        
        return {
            "controlled": self.num_controlled_vehicles,
            "aggressive": num_aggressive,
            "defensive": num_defensive,
            "normal": max(0, num_normal),
            "random": num_random
        }
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment with multi-agent dynamics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add multi-agent specific information
        if hasattr(self.env, 'road') and self.env.road:
            vehicles = self.env.road.vehicles
            if vehicles:
                info["vehicle_count"] = len(vehicles)
                info["nearby_vehicles"] = self._count_nearby_vehicles()
                info["traffic_density"] = len(vehicles) / max(1, len(self.env.road.network.lanes_list()))
                
                # Detect interesting multi-agent events
                info["multi_agent_events"] = self._detect_events(vehicles)
        
        return obs, reward, terminated, truncated, info
    
    def _count_nearby_vehicles(self, radius: float = 50.0) -> int:
        """Count vehicles near the controlled vehicle."""
        if not (hasattr(self.env, 'vehicle') and self.env.vehicle and 
                hasattr(self.env, 'road') and self.env.road):
            return 0
        
        ego_vehicle = self.env.vehicle
        nearby_count = 0
        
        for vehicle in self.env.road.vehicles:
            if vehicle != ego_vehicle:
                distance = np.linalg.norm(
                    np.array(vehicle.position) - np.array(ego_vehicle.position)
                )
                if distance <= radius:
                    nearby_count += 1
                    
        return nearby_count
    
    def _detect_events(self, vehicles: List) -> Dict[str, bool]:
        """Detect interesting multi-agent events."""
        events = {
            "lane_change_detected": False,
            "aggressive_behavior": False,
            "traffic_jam": False,
            "near_collision": False,
            "cooperative_behavior": False
        }
        
        if not vehicles or len(vehicles) < 2:
            return events
        
        # Simple event detection based on vehicle states
        speeds = [getattr(v, 'speed', 20) for v in vehicles if hasattr(v, 'speed')]
        positions = [getattr(v, 'position', [0, 0]) for v in vehicles if hasattr(v, 'position')]
        
        if speeds:
            avg_speed = np.mean(speeds)
            speed_variance = np.var(speeds)
            
            # Traffic jam detection
            events["traffic_jam"] = avg_speed < 15 and len(vehicles) > 20
            
            # Aggressive behavior detection
            events["aggressive_behavior"] = speed_variance > 100 or max(speeds) > 40
        
        if len(positions) >= 2:
            # Near collision detection
            min_distance = float('inf')
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    min_distance = min(min_distance, distance)
            
            events["near_collision"] = min_distance < 8.0
        
        return events