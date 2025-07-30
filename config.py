"""
Shared Configuration for Autonomous Driving Scenarios
Contains environment settings, training parameters, and utility functions for all scenarios.
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import custom modules
from src.environments.vision_wrapper import HighwayVisionEnv
from src.environments.domain_randomizer import DomainRandomizer
from src.environments.reward_shaper import RewardShaper
from src.environments.multi_agent_wrapper import MultiAgentWrapper
from src.models.ppo_policy import PPOVisionPolicy

# Scenario configurations
SCENARIOS = {
    "highway": "highway-v0",
    "intersection": "intersection-v0", 
    "roundabout": "roundabout-v0",
    "parking": "parking-v0"
}

# Shared environment configuration
ENV_CONFIG = {
    "observation_type": "rgb",
    "image_shape": (84, 84),
    "stack_frames": 4,
    "normalize": True,
    "enhance_contrast": True
}

# Shared PPO training configuration
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1
}

# Scenario-specific reward shaping parameters
REWARD_CONFIGS = {
    "highway": {
        "collision_reward": -100.0,
        "lane_keeping_reward": 1.0,
        "speed_efficiency_reward": 0.5,
        "use_shaped_rewards": True
    },
    "intersection": {
        "collision_reward": -200.0,
        "lane_keeping_reward": 0.5,
        "speed_efficiency_reward": 0.3,
        "use_shaped_rewards": True
    },
    "roundabout": {
        "collision_reward": -150.0,
        "lane_keeping_reward": 0.8,
        "speed_efficiency_reward": 0.4,
        "use_shaped_rewards": True
    },
    "parking": {
        "collision_reward": -50.0,
        "lane_keeping_reward": 0.2,
        "speed_efficiency_reward": 0.1,
        "use_shaped_rewards": True
    }
}

# Scenario-specific domain randomization parameters
DOMAIN_RANDOMIZATION_CONFIGS = {
    "highway": {
        "randomize_traffic": True,
        "randomize_weather": True,
        "randomize_behavior": True,
        "traffic_density_range": (10, 30)
    },
    "intersection": {
        "randomize_traffic": True,
        "randomize_weather": False,
        "randomize_behavior": True,
        "traffic_density_range": (5, 15)
    },
    "roundabout": {
        "randomize_traffic": True,
        "randomize_weather": True,
        "randomize_behavior": True,
        "traffic_density_range": (8, 20)
    },
    "parking": {
        "randomize_traffic": False,
        "randomize_weather": False,
        "randomize_behavior": False,
        "traffic_density_range": (0, 5)
    }
}

# Scenario-specific multi-agent parameters
MULTI_AGENT_CONFIGS = {
    "highway": {
        "aggressive_vehicles_ratio": 0.2,
        "defensive_vehicles_ratio": 0.2,
        "min_vehicles": 15,
        "max_vehicles": 35
    },
    "intersection": {
        "aggressive_vehicles_ratio": 0.1,
        "defensive_vehicles_ratio": 0.3,
        "min_vehicles": 5,
        "max_vehicles": 15
    },
    "roundabout": {
        "aggressive_vehicles_ratio": 0.15,
        "defensive_vehicles_ratio": 0.25,
        "min_vehicles": 8,
        "max_vehicles": 20
    },
    "parking": {
        "aggressive_vehicles_ratio": 0.0,
        "defensive_vehicles_ratio": 0.0,
        "min_vehicles": 0,
        "max_vehicles": 5
    }
}

# Training configurations
TRAINING_CONFIG = {
    "total_timesteps": 500000,
    "n_envs": 4,
    "eval_freq": 10000,
    "save_freq": 50000,
    "eval_episodes": 10
}

# Directory configurations
DIRECTORIES = {
    "models": "models",
    "logs": "logs",
    "results": "results",
    "data": "data"
}


def create_training_env(scenario: str, use_wrappers: bool = True, **kwargs):
    """
    Create a training environment for a specific scenario with all wrappers.
    
    Args:
        scenario: One of "highway", "intersection", "roundabout", "parking"
        use_wrappers: Whether to apply reward shaping, domain randomization, and multi-agent wrappers
        **kwargs: Additional environment parameters
        
    Returns:
        Configured environment for training
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Must be one of {list(SCENARIOS.keys())}")
    
    env_name = SCENARIOS[scenario]
    
    # Create base vision environment
    env = HighwayVisionEnv.create_env(
        env_name=env_name,
        **ENV_CONFIG,
        **kwargs
    )
    
    if use_wrappers:
        # Add reward shaping
        reward_config = REWARD_CONFIGS[scenario]
        env = RewardShaper(env, **reward_config)
        
        # Add domain randomization
        domain_config = DOMAIN_RANDOMIZATION_CONFIGS[scenario]
        env = DomainRandomizer(env, **domain_config)
        
        # Add multi-agent complexity
        multi_agent_config = MULTI_AGENT_CONFIGS[scenario]
        env = MultiAgentWrapper(env, **multi_agent_config)
    
    return env


def create_evaluation_env(scenario: str, render_mode: str = "human", **kwargs):
    """
    Create an evaluation environment for a specific scenario.
    
    Args:
        scenario: One of "highway", "intersection", "roundabout", "parking"
        render_mode: Rendering mode for visualization
        **kwargs: Additional environment parameters
        
    Returns:
        Configured environment for evaluation
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Must be one of {list(SCENARIOS.keys())}")
    
    env_name = SCENARIOS[scenario]
    
    # Create environment with rendering enabled
    env = HighwayVisionEnv.create_env(
        env_name=env_name,
        render_mode=render_mode,
        **ENV_CONFIG,
        **kwargs
    )
    
    return env


def create_ppo_model(env, scenario: str, tensorboard_log: str = None):
    """
    Create a PPO model with scenario-specific configuration.
    
    Args:
        env: Training environment
        scenario: Scenario name for logging
        tensorboard_log: Path for tensorboard logs
        
    Returns:
        Configured PPO model
    """
    # Use CnnPolicy with normalize_images=False for multi-channel stacked frames
    policy_kwargs = {
        "normalize_images": False,
        "features_extractor_kwargs": {"features_dim": 256}
    }
    
    return PPO(
        policy="CnnPolicy",
        env=env,
        tensorboard_log=tensorboard_log or f"logs/{scenario}_tensorboard/",
        policy_kwargs=policy_kwargs,
        **PPO_CONFIG
    )


def setup_callbacks(scenario: str, eval_env, model_save_path: str):
    """
    Setup training callbacks for evaluation and checkpointing.
    
    Args:
        scenario: Scenario name
        eval_env: Environment for evaluation
        model_save_path: Base path for saving models
        
    Returns:
        List of configured callbacks
    """
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_save_path}_best/",
        log_path=f"logs/{scenario}_eval/",
        eval_freq=TRAINING_CONFIG["eval_freq"],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=f"{model_save_path}_checkpoints/",
        name_prefix=f"ppo_{scenario}"
    )
    
    return [eval_callback, checkpoint_callback]


def ensure_directories():
    """Create necessary directories for training."""
    for directory in DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)


def get_model_path(scenario: str):
    """Get the model file path for a scenario."""
    return f"models/ppo_{scenario}_agent.zip"


def get_best_model_path(scenario: str):
    """Get the best model file path for a scenario."""
    return f"models/ppo_{scenario}_agent_best/best_model.zip"


def evaluate_agent(model_path: str, scenario: str, num_episodes: int = 5):
    """
    Evaluate a trained agent on a specific scenario.
    
    Args:
        model_path: Path to the saved model
        scenario: Scenario to evaluate on
        num_episodes: Number of episodes to run
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"🎯 Evaluating {scenario} agent for {num_episodes} episodes...")
    
    # Create evaluation environment
    env = create_evaluation_env(scenario)
    
    # Load model
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"❌ Model not found at {model_path}")
        return None
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    crash_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        crashed = False
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if info.get("crashed", False):
                crashed = True
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if crashed:
            crash_count += 1
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}, Crashed={crashed}")
    
    # Calculate and print results
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    crash_rate = crash_count / num_episodes
    success_rate = 1 - crash_rate
    
    print(f"\n📊 Evaluation Results for {scenario}:")
    print(f"   Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   Average Episode Length: {avg_length:.1f}")
    print(f"   Crash Rate: {crash_count}/{num_episodes} ({100*crash_rate:.1f}%)")
    print(f"   Success Rate: {100*success_rate:.1f}%")
    
    env.close()
    
    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "crash_rate": crash_rate,
        "success_rate": success_rate,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }