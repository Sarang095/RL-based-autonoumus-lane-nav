#!/usr/bin/env python3
"""
Evaluation Script for Autonomous Driving Agents
Flexible script to visually demonstrate any of the four trained agents.
Usage: python evaluate.py <scenario> [options]
"""

import os
import argparse
import time
from stable_baselines3 import PPO
from config import (
    create_evaluation_env, get_model_path, get_best_model_path, 
    evaluate_agent, SCENARIOS
)


def demonstrate_agent(scenario: str, model_path: str = None, num_episodes: int = 5, 
                     render_mode: str = "human", deterministic: bool = True,
                     sleep_time: float = 0.02):
    """
    Demonstrate a trained agent in the specified scenario.
    
    Args:
        scenario: One of "highway", "intersection", "roundabout", "parking"
        model_path: Path to the model file (if None, uses default path)
        num_episodes: Number of episodes to run
        render_mode: Rendering mode ("human" for visual, "rgb_array" for headless)
        deterministic: Whether to use deterministic actions
        sleep_time: Sleep time between steps for visualization
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Must be one of {list(SCENARIOS.keys())}")
    
    # Determine model path
    if model_path is None:
        # Try the regular model path first, then the best model path
        model_path = get_model_path(scenario)
        if not os.path.exists(model_path):
            model_path = get_best_model_path(scenario)
            if not os.path.exists(model_path):
                print(f"❌ No trained model found for {scenario} scenario!")
                print(f"   Expected locations:")
                print(f"   - {get_model_path(scenario)}")
                print(f"   - {get_best_model_path(scenario)}")
                print(f"\n   Train the model first using: python train_{scenario}.py")
                return None
    
    print(f"🎯 Demonstrating {scenario.capitalize()} Agent")
    print("=" * 60)
    print(f"Scenario: {SCENARIOS[scenario]}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render mode: {render_mode}")
    print(f"Deterministic: {deterministic}")
    print("=" * 60)
    
    # Create evaluation environment
    env = create_evaluation_env(scenario, render_mode=render_mode)
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print(f"✅ Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        return None
    
    # Run demonstration episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\n🚀 Starting demonstration...")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        crashed = False
        
        # Run episode
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Check for crash
            if info.get("crashed", False):
                crashed = True
            
            # Add small delay for better visualization
            if render_mode == "human" and sleep_time > 0:
                time.sleep(sleep_time)
            
            # Episode termination
            if terminated or truncated:
                break
        
        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if not crashed:
            success_count += 1
        
        # Print episode summary
        status = "❌ CRASHED" if crashed else "✅ SUCCESS"
        print(f"   {status} | Reward: {episode_reward:.2f} | Length: {episode_length} steps")
        
        # Brief pause between episodes
        if episode < num_episodes - 1 and render_mode == "human":
            time.sleep(1.0)
    
    # Clean up
    env.close()
    
    # Print final results
    import numpy as np
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    print(f"\n📊 Final Demonstration Results:")
    print("=" * 60)
    print(f"   Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   Average Episode Length: {avg_length:.1f} steps")
    print(f"   Success Rate: {success_count}/{num_episodes} ({100*success_rate:.1f}%)")
    print(f"   Total Episodes: {num_episodes}")
    print("=" * 60)
    
    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "success_rate": success_rate,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Demonstrate Autonomous Driving Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py highway              # Demonstrate highway agent
  python evaluate.py intersection         # Demonstrate intersection agent
  python evaluate.py roundabout           # Demonstrate roundabout agent
  python evaluate.py parking              # Demonstrate parking agent
  
  python evaluate.py highway --episodes 10 --model models/custom_highway.zip
  python evaluate.py intersection --no-render --episodes 20
        """
    )
    
    parser.add_argument(
        "scenario",
        choices=list(SCENARIOS.keys()),
        help="Driving scenario to demonstrate"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Path to model file (default: auto-detect)"
    )
    
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    
    parser.add_argument(
        "--no-render", 
        action="store_true",
        help="Run without visual rendering (faster)"
    )
    
    parser.add_argument(
        "--non-deterministic", 
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    
    parser.add_argument(
        "--slow", 
        action="store_true",
        help="Add extra delay for better visualization"
    )
    
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Remove delays for faster execution"
    )
    
    args = parser.parse_args()
    
    # Determine render mode
    render_mode = "rgb_array" if args.no_render else "human"
    
    # Determine sleep time
    if args.fast:
        sleep_time = 0.0
    elif args.slow:
        sleep_time = 0.1
    else:
        sleep_time = 0.02
    
    # Run demonstration
    try:
        results = demonstrate_agent(
            scenario=args.scenario,
            model_path=args.model,
            num_episodes=args.episodes,
            render_mode=render_mode,
            deterministic=not args.non_deterministic,
            sleep_time=sleep_time
        )
        
        if results is None:
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()