"""
Demo script for Vision-Based Autonomous Driving Agent
Showcases trained models in different scenarios
"""
import os
import argparse
import torch
import numpy as np
import time
from typing import Optional

from src.config import Config
from src.models import create_model
from src.environment_wrapper import create_environment


def load_il_model(model_path: str):
    """Load imitation learning model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"IL model not found: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_model('cnn', checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded IL model from: {model_path}")
    return model, device


def load_ppo_model(model_path: str):
    """Load PPO model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    
    print(f"Loaded PPO model from: {model_path}")
    return model


def get_il_action(model, obs, device):
    """Get action from IL model"""
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        if obs_tensor.dtype == torch.uint8:
            obs_tensor = obs_tensor.float() / 255.0
        
        action = model(obs_tensor)
        return action.cpu().numpy().squeeze()


def run_demo_episode(model, model_type: str, env, max_steps: int = 1000, render: bool = True):
    """Run a single demo episode"""
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    
    print(f"\nStarting {model_type} demo episode...")
    
    while not done and step_count < max_steps:
        # Get action based on model type
        if model_type == 'IL':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            action = get_il_action(model, obs, device)
        elif model_type == 'PPO':
            action, _ = model.predict(obs, deterministic=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        step_count += 1
        
        # Render if requested
        if render:
            env.render()
            time.sleep(0.05)  # Slow down for visualization
        
        # Print progress every 50 steps
        if step_count % 50 == 0:
            print(f"  Step {step_count}: Reward = {episode_reward:.2f}")
    
    # Episode summary
    crashed = info.get('crashed', False)
    print(f"\nEpisode completed!")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Steps taken: {step_count}")
    print(f"  Crashed: {crashed}")
    print(f"  Success: {not crashed}")
    
    return {
        'reward': episode_reward,
        'steps': step_count,
        'crashed': crashed,
        'success': not crashed
    }


def run_scenario_demo(args):
    """Run demo for a specific scenario"""
    print(f"Setting up {args.scenario} scenario...")
    
    # Create environment
    env = create_environment(
        args.scenario,
        enable_domain_randomization=False,
        enable_multi_agent=True,
        enable_reward_shaping=True
    )
    
    results = {}
    
    # Demo IL model if available
    if args.il_model and os.path.exists(args.il_model):
        print(f"\n{'='*50}")
        print("IMITATION LEARNING MODEL DEMO")
        print('='*50)
        
        try:
            il_model, device = load_il_model(args.il_model)
            
            il_results = []
            for episode in range(args.episodes):
                print(f"\nIL Episode {episode + 1}/{args.episodes}")
                result = run_demo_episode(il_model, 'IL', env, args.max_steps, args.render)
                il_results.append(result)
            
            # Calculate statistics
            results['IL'] = {
                'episodes': il_results,
                'avg_reward': np.mean([r['reward'] for r in il_results]),
                'avg_steps': np.mean([r['steps'] for r in il_results]),
                'success_rate': np.mean([r['success'] for r in il_results])
            }
            
        except Exception as e:
            print(f"Error loading IL model: {e}")
    
    # Demo PPO model if available
    if args.ppo_model and os.path.exists(args.ppo_model):
        print(f"\n{'='*50}")
        print("PPO MODEL DEMO")
        print('='*50)
        
        try:
            ppo_model = load_ppo_model(args.ppo_model)
            
            ppo_results = []
            for episode in range(args.episodes):
                print(f"\nPPO Episode {episode + 1}/{args.episodes}")
                result = run_demo_episode(ppo_model, 'PPO', env, args.max_steps, args.render)
                ppo_results.append(result)
            
            # Calculate statistics
            results['PPO'] = {
                'episodes': ppo_results,
                'avg_reward': np.mean([r['reward'] for r in ppo_results]),
                'avg_steps': np.mean([r['steps'] for r in ppo_results]),
                'success_rate': np.mean([r['success'] for r in ppo_results])
            }
            
        except Exception as e:
            print(f"Error loading PPO model: {e}")
    
    env.close()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"DEMO SUMMARY - {args.scenario.upper()} SCENARIO")
    print('='*50)
    
    for model_type, stats in results.items():
        print(f"\n{model_type} Model Results:")
        print(f"  Average Reward: {stats['avg_reward']:.2f}")
        print(f"  Average Steps: {stats['avg_steps']:.1f}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
    
    if len(results) > 1:
        print(f"\nComparison:")
        il_reward = results.get('IL', {}).get('avg_reward', 0)
        ppo_reward = results.get('PPO', {}).get('avg_reward', 0)
        
        if il_reward > 0 and ppo_reward > 0:
            improvement = ppo_reward - il_reward
            print(f"  PPO vs IL Reward Improvement: {improvement:.2f}")
            print(f"  Better Model: {'PPO' if ppo_reward > il_reward else 'IL'}")
    
    return results


def interactive_demo(args):
    """Interactive demo where user can choose scenarios and models"""
    print("Interactive Demo Mode")
    print("=" * 50)
    
    # Check available models
    il_available = args.il_model and os.path.exists(args.il_model)
    ppo_available = args.ppo_model and os.path.exists(args.ppo_model)
    
    print(f"Available models:")
    print(f"  Imitation Learning: {'✓' if il_available else '✗'}")
    print(f"  PPO: {'✓' if ppo_available else '✗'}")
    
    if not (il_available or ppo_available):
        print("No models available! Please train models first.")
        return
    
    scenarios = ['highway', 'roundabout', 'parking']
    
    while True:
        print(f"\nAvailable scenarios: {scenarios}")
        print("Enter 'quit' to exit")
        
        scenario = input("Choose scenario (highway/roundabout/parking): ").strip().lower()
        
        if scenario == 'quit':
            break
        
        if scenario not in scenarios:
            print("Invalid scenario. Please choose from: highway, roundabout, parking")
            continue
        
        # Set scenario and run demo
        args.scenario = scenario
        run_scenario_demo(args)
        
        if input("\nRun another demo? (y/n): ").strip().lower() != 'y':
            break


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Demo for Vision-Based Autonomous Driving Agent')
    
    # Model paths
    parser.add_argument('--il-model', type=str, 
                       default=os.path.join(Config.PATHS['models_dir'], 'best_model.pth'),
                       help='Path to IL model')
    parser.add_argument('--ppo-model', type=str,
                       default=os.path.join(Config.PATHS['models_dir'], 'ppo_final_model.zip'),
                       help='Path to PPO model')
    
    # Demo settings
    parser.add_argument('--scenario', type=str, default='highway',
                       choices=['highway', 'roundabout', 'parking'],
                       help='Scenario to demo')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true', default=True,
                       help='Render the environment')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive demo')
    
    args = parser.parse_args()
    
    print("Vision-Based Autonomous Driving Demo")
    print("=" * 50)
    print(f"IL Model: {args.il_model}")
    print(f"PPO Model: {args.ppo_model}")
    
    if args.interactive:
        interactive_demo(args)
    else:
        run_scenario_demo(args)
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()