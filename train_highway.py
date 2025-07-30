#!/usr/bin/env python3
"""
Training Script for Highway-v0 Autonomous Driving Agent
Uses PPO with CnnPolicy for vision-based highway driving.
"""

import os
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv
from config import (
    create_training_env, create_ppo_model, setup_callbacks, 
    ensure_directories, get_model_path, TRAINING_CONFIG
)


def train_highway_agent(total_timesteps: int = None, model_save_path: str = None):
    """
    Train a PPO agent for the highway-v0 scenario.
    
    Args:
        total_timesteps: Number of training timesteps
        model_save_path: Path to save the trained model
    """
    scenario = "highway"
    
    # Use default values from config if not provided
    if total_timesteps is None:
        total_timesteps = TRAINING_CONFIG["total_timesteps"]
    if model_save_path is None:
        model_save_path = get_model_path(scenario)
    
    print(f"🚗 Training Highway-v0 Agent")
    print("=" * 50)
    print(f"Scenario: {scenario}")
    print(f"Training timesteps: {total_timesteps:,}")
    print(f"Model save path: {model_save_path}")
    print("=" * 50)
    
    # Ensure directories exist
    ensure_directories()
    
    # Create training environments
    def make_train_env():
        return create_training_env(scenario, use_wrappers=True)
    
    def make_eval_env():
        return create_training_env(scenario, use_wrappers=True)
    
    # Create vectorized environments
    n_envs = TRAINING_CONFIG["n_envs"]
    train_env = DummyVecEnv([make_train_env for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_eval_env for _ in range(2)])
    
    print(f"✅ Created {n_envs} training environments")
    print(f"✅ Created 2 evaluation environments")
    
    # Create PPO model
    model = create_ppo_model(
        env=train_env,
        scenario=scenario,
        tensorboard_log=f"logs/{scenario}_tensorboard/"
    )
    
    print(f"✅ Created PPO model with CnnPolicy")
    
    # Setup training callbacks
    base_save_path = model_save_path.replace(".zip", "")
    callbacks = setup_callbacks(scenario, eval_env, base_save_path)
    
    print(f"✅ Setup evaluation and checkpoint callbacks")
    
    # Train the model
    print(f"\n🚀 Starting training for {total_timesteps:,} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        model.save(model_save_path)
        print(f"✅ Training completed! Model saved to {model_save_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        # Save current model state
        interrupted_path = model_save_path.replace(".zip", "_interrupted.zip")
        model.save(interrupted_path)
        print(f"💾 Saved interrupted model to {interrupted_path}")
        
    finally:
        # Clean up environments
        train_env.close()
        eval_env.close()
    
    print(f"\n📁 Training artifacts saved:")
    print(f"   - Model: {model_save_path}")
    print(f"   - Best model: {base_save_path}_best/best_model.zip")
    print(f"   - Checkpoints: {base_save_path}_checkpoints/")
    print(f"   - TensorBoard logs: logs/{scenario}_tensorboard/")
    print(f"   - Evaluation logs: logs/{scenario}_eval/")


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train Highway-v0 Autonomous Driving Agent")
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=TRAINING_CONFIG["total_timesteps"],
        help=f"Number of training timesteps (default: {TRAINING_CONFIG['total_timesteps']:,})"
    )
    parser.add_argument(
        "--save-path", 
        type=str, 
        default=get_model_path("highway"),
        help="Path to save the trained model"
    )
    
    args = parser.parse_args()
    
    # Train the agent
    train_highway_agent(
        total_timesteps=args.timesteps,
        model_save_path=args.save_path
    )


if __name__ == "__main__":
    main()