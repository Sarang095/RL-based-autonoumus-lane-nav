"""
Configuration file for the autonomous driving project
"""
import numpy as np
from typing import Dict, Any, Tuple

class Config:
    """Central configuration class for the autonomous driving project"""
    
    # Environment settings
    ENV_CONFIGS = {
        'highway': {
            'observation': {
                'type': 'GrayscaleObservation',
                'observation_shape': (84, 84),
                'stack_size': 4,
                'weights': [0.2989, 0.5870, 0.1140],  # RGB to grayscale conversion
            },
            'action': {
                'type': 'ContinuousAction',
                'longitudinal': True,
                'lateral': True,
            },
            'simulation_frequency': 15,
            'policy_frequency': 5,
            'duration': 40,  # seconds
            'vehicles_count': 50,
            'initial_lane_id': None,
            'ego_spacing': 2,
            'vehicles_density': 1,
        },
        'roundabout': {
            'observation': {
                'type': 'GrayscaleObservation',
                'observation_shape': (84, 84),
                'stack_size': 4,
                'weights': [0.2989, 0.5870, 0.1140],  # RGB to grayscale conversion
            },
            'action': {'type': 'ContinuousAction'},
            'incoming_vehicle_destination': None,
            'vehicles_count': 12,
            'duration': 20,
        },
        'parking': {
            'observation': {
                'type': 'GrayscaleObservation',
                'observation_shape': (84, 84),
                'stack_size': 4,
                'weights': [0.2989, 0.5870, 0.1140],  # RGB to grayscale conversion
            },
            'action': {'type': 'ContinuousAction'},
            'vehicles_count': 10,
            'goal_is_final': True,
        }
    }
    
    # CNN Model settings
    CNN_CONFIG = {
        'input_channels': 4,  # stack_size
        'input_height': 84,
        'input_width': 84,
        'conv_layers': [
            {'out_channels': 32, 'kernel_size': 8, 'stride': 4},
            {'out_channels': 64, 'kernel_size': 4, 'stride': 2},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1},
        ],
        'fc_layers': [512],
        'output_dim': 2,  # continuous actions: [acceleration, steering]
        'dropout_rate': 0.2,
    }
    
    # Imitation Learning settings
    IL_CONFIG = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
        'expert_episodes': 1000,
        'save_model_every': 20,
    }
    
    # PPO settings
    PPO_CONFIG = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'total_timesteps': 1_000_000,
    }
    
    # Reward shaping
    REWARD_CONFIG = {
        'collision_penalty': -100,
        'lane_keeping_reward': 1,
        'speed_efficiency_reward': 0.5,
        'rule_violation_penalty': -5,
        'goal_reached_reward': 100,
        'time_penalty': -0.1,
    }
    
    # Domain randomization settings
    DOMAIN_RANDOMIZATION = {
        'traffic_density_range': (0.5, 2.0),
        'vehicle_speed_variance': 0.2,
        'ego_acceleration_noise': 0.1,
        'observation_noise_std': 0.05,
        'lane_visibility_range': (0.7, 1.0),
        'weather_conditions': ['clear', 'fog', 'rain'],
        'randomize_probability': 0.3,
    }
    
    # Multi-agent settings
    MULTI_AGENT_CONFIG = {
        'aggressive_vehicles_ratio': 0.2,
        'defensive_vehicles_ratio': 0.3,
        'random_vehicles_ratio': 0.5,
        'spawn_probability': 0.8,
        'max_agents': 20,
    }
    
    # Training settings
    TRAINING_CONFIG = {
        'device': 'cuda',  # Will fall back to CPU if CUDA unavailable
        'num_workers': 4,
        'log_interval': 100,
        'eval_interval': 1000,
        'save_interval': 5000,
        'eval_episodes': 10,
    }
    
    # Paths
    PATHS = {
        'data_dir': 'data/',
        'models_dir': 'models/',
        'logs_dir': 'logs/',
        'videos_dir': 'videos/',
        'expert_data': 'data/expert_trajectories.pkl',
        'il_model': 'models/imitation_model.pth',
        'ppo_model': 'models/ppo_model.zip',
    }

    @classmethod
    def get_env_config(cls, env_name: str) -> Dict[str, Any]:
        """Get environment configuration by name"""
        return cls.ENV_CONFIGS.get(env_name, cls.ENV_CONFIGS['highway'])
    
    @classmethod
    def get_observation_shape(cls, env_name: str) -> Tuple[int, ...]:
        """Get observation shape for given environment"""
        config = cls.get_env_config(env_name)
        obs_config = config['observation']
        return (obs_config['stack_size'], *obs_config['observation_shape'])