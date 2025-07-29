"""
Main training pipeline for Vision-Based Autonomous Driving Agent
Orchestrates the complete training process: IL -> PPO -> Evaluation
"""
import os
import argparse
import torch
from typing import Optional

from src.config import Config
from src.training import train_imitation_model, train_ppo_model
from src.evaluation import ModelEvaluator, compare_models


def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        Config.PATHS['data_dir'],
        Config.PATHS['models_dir'],
        Config.PATHS['logs_dir'],
        Config.PATHS['videos_dir'],
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def phase1_imitation_learning(args):
    """Phase 1: Train imitation learning model"""
    print("\n" + "="*60)
    print("PHASE 1: IMITATION LEARNING")
    print("="*60)
    
    print("Training CNN model using expert demonstrations...")
    print(f"Environment: {args.env}")
    print(f"Expert episodes: {args.il_episodes}")
    
    # Train imitation learning model
    il_trainer = train_imitation_model(
        env_name=args.env,
        num_episodes=args.il_episodes,
        force_recollect=args.force_recollect
    )
    
    # Save the model path for next phase
    il_model_path = os.path.join(Config.PATHS['models_dir'], 'best_model.pth')
    
    print(f"\nPhase 1 completed! Model saved to: {il_model_path}")
    return il_model_path


def phase2_reinforcement_learning(args, il_model_path: Optional[str] = None):
    """Phase 2: Train PPO model using RL"""
    print("\n" + "="*60)
    print("PHASE 2: REINFORCEMENT LEARNING (PPO)")
    print("="*60)
    
    print("Training PPO agent with domain randomization and multi-agent scenarios...")
    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.ppo_timesteps}")
    print(f"Parallel environments: {args.n_envs}")
    
    if il_model_path and args.use_pretrained:
        print(f"Using pretrained IL model: {il_model_path}")
    else:
        il_model_path = None
        print("Training PPO from scratch (no IL pretraining)")
    
    # Train PPO model
    ppo_trainer = train_ppo_model(
        env_name=args.env,
        total_timesteps=args.ppo_timesteps,
        n_envs=args.n_envs,
        pretrained_model_path=il_model_path
    )
    
    # Get final model path
    ppo_model_path = os.path.join(Config.PATHS['models_dir'], 'ppo_final_model.zip')
    
    print(f"\nPhase 2 completed! Model saved to: {ppo_model_path}")
    return ppo_model_path


def phase3_evaluation(args, il_model_path: Optional[str] = None, ppo_model_path: Optional[str] = None):
    """Phase 3: Comprehensive evaluation and comparison"""
    print("\n" + "="*60)
    print("PHASE 3: EVALUATION AND COMPARISON")
    print("="*60)
    
    evaluator = ModelEvaluator()
    
    scenarios = ['highway']  # Start with highway, add more as needed
    if args.all_scenarios:
        scenarios = ['highway', 'roundabout', 'parking']
    
    print(f"Evaluating models on scenarios: {scenarios}")
    print(f"Episodes per scenario: {args.eval_episodes}")
    
    # Evaluate IL model if available
    if il_model_path and os.path.exists(il_model_path):
        print("\nEvaluating Imitation Learning model...")
        
        # Load IL model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        il_checkpoint = torch.load(il_model_path, map_location=device)
        
        from src.models import create_model
        il_model = create_model('cnn', il_checkpoint['model_config'])
        il_model.load_state_dict(il_checkpoint['model_state_dict'])
        il_model.to(device)
        
        il_results = evaluator.evaluate_model(
            il_model, 'imitation', scenarios, args.eval_episodes, args.save_videos
        )
    else:
        il_results = None
        print("IL model not found, skipping IL evaluation")
    
    # Evaluate PPO model if available
    if ppo_model_path and os.path.exists(ppo_model_path):
        print("\nEvaluating PPO model...")
        
        from stable_baselines3 import PPO
        ppo_model = PPO.load(ppo_model_path)
        
        ppo_results = evaluator.evaluate_model(
            ppo_model, 'ppo', scenarios, args.eval_episodes, args.save_videos
        )
    else:
        ppo_results = None
        print("PPO model not found, skipping PPO evaluation")
    
    # Compare models if both are available
    if il_results and ppo_results and args.compare_models:
        print("\nComparing IL and PPO models...")
        comparison = compare_models(il_model_path, ppo_model_path, scenarios, args.eval_episodes)
        
        # Print comparison summary
        summary = comparison['comparison_summary']
        print(f"\nComparison Summary:")
        print(f"Better overall model: {summary['better_model']}")
        print(f"Reward improvement (PPO vs IL): {summary['reward_improvement']:.2f}")
        print(f"Success rate improvement: {summary['success_rate_improvement']:.2%}")
    
    print("\nPhase 3 completed!")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Vision-Based Autonomous Driving Training Pipeline')
    
    # General arguments
    parser.add_argument('--env', type=str, default='highway', 
                       choices=['highway', 'roundabout', 'parking'],
                       help='Environment to train on')
    parser.add_argument('--phases', type=str, default='all',
                       choices=['il', 'ppo', 'eval', 'all'],
                       help='Which phases to run')
    parser.add_argument('--force-recollect', action='store_true',
                       help='Force recollection of expert data')
    
    # Imitation Learning arguments
    parser.add_argument('--il-episodes', type=int, default=100,
                       help='Number of expert episodes for IL')
    
    # PPO arguments
    parser.add_argument('--ppo-timesteps', type=int, default=500_000,
                       help='Total timesteps for PPO training')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments for PPO')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                       help='Use IL model to initialize PPO')
    
    # Evaluation arguments
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Number of episodes for evaluation')
    parser.add_argument('--all-scenarios', action='store_true',
                       help='Evaluate on all scenarios (highway, roundabout, parking)')
    parser.add_argument('--save-videos', action='store_true',
                       help='Save video recordings during evaluation')
    parser.add_argument('--compare-models', action='store_true', default=True,
                       help='Compare IL and PPO models')
    
    args = parser.parse_args()
    
    print("Vision-Based Autonomous Driving Training Pipeline")
    print("="*60)
    print(f"Environment: {args.env}")
    print(f"Phases to run: {args.phases}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Setup directories
    setup_directories()
    
    # Initialize paths
    il_model_path = None
    ppo_model_path = None
    
    # Run phases based on arguments
    if args.phases in ['il', 'all']:
        il_model_path = phase1_imitation_learning(args)
    
    if args.phases in ['ppo', 'all']:
        # If IL model exists but wasn't just trained, use existing one
        if not il_model_path:
            potential_il_path = os.path.join(Config.PATHS['models_dir'], 'best_model.pth')
            if os.path.exists(potential_il_path):
                il_model_path = potential_il_path
        
        ppo_model_path = phase2_reinforcement_learning(args, il_model_path)
    
    if args.phases in ['eval', 'all']:
        # If models weren't just trained, look for existing ones
        if not il_model_path:
            potential_il_path = os.path.join(Config.PATHS['models_dir'], 'best_model.pth')
            if os.path.exists(potential_il_path):
                il_model_path = potential_il_path
        
        if not ppo_model_path:
            potential_ppo_path = os.path.join(Config.PATHS['models_dir'], 'ppo_final_model.zip')
            if os.path.exists(potential_ppo_path):
                ppo_model_path = potential_ppo_path
        
        phase3_evaluation(args, il_model_path, ppo_model_path)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print("Check the following directories for results:")
    print(f"  Models: {Config.PATHS['models_dir']}")
    print(f"  Logs: {Config.PATHS['logs_dir']}")
    print(f"  Videos: {Config.PATHS['videos_dir']}")


if __name__ == "__main__":
    main()