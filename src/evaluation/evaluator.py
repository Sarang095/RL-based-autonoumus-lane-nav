"""
Comprehensive evaluation module for autonomous driving models
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import torch
import json
from datetime import datetime
import imageio

from ..models import create_model
from ..environment_wrapper import create_environment
from ..config import Config
from ..training import ImitationLearner, PPOTrainer


class ModelEvaluator:
    """
    Comprehensive evaluator for autonomous driving models
    Tests models across different scenarios and conditions
    """
    
    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir or os.path.join(Config.PATHS['logs_dir'], 'evaluation')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def evaluate_model(self, 
                      model,
                      model_type: str,
                      scenarios: List[str] = None,
                      num_episodes: int = 20,
                      save_videos: bool = False) -> Dict[str, Any]:
        """
        Evaluate a model across multiple scenarios
        
        Args:
            model: The model to evaluate (IL model or PPO model)
            model_type: Type of model ('imitation', 'ppo')
            scenarios: List of scenarios to test ['highway', 'roundabout', 'parking']
            num_episodes: Number of episodes per scenario
            save_videos: Whether to save video recordings
            
        Returns:
            Comprehensive evaluation results
        """
        if scenarios is None:
            scenarios = ['highway', 'roundabout', 'parking']
        
        print(f"Evaluating {model_type} model across {len(scenarios)} scenarios...")
        
        results = {
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'scenarios': {},
            'overall_stats': {}
        }
        
        all_rewards = []
        all_lengths = []
        all_success_rates = []
        
        for scenario in scenarios:
            print(f"\nEvaluating on {scenario} scenario...")
            
            scenario_results = self._evaluate_scenario(
                model, model_type, scenario, num_episodes, save_videos
            )
            
            results['scenarios'][scenario] = scenario_results
            
            # Collect overall statistics
            all_rewards.extend(scenario_results['episode_rewards'])
            all_lengths.extend(scenario_results['episode_lengths'])
            all_success_rates.append(scenario_results['success_rate'])
        
        # Calculate overall statistics
        results['overall_stats'] = {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_length': np.mean(all_lengths),
            'std_length': np.std(all_lengths),
            'overall_success_rate': np.mean(all_success_rates),
            'total_episodes': len(all_rewards)
        }
        
        # Save results
        self._save_results(results, model_type)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        return results
    
    def _evaluate_scenario(self, 
                          model,
                          model_type: str,
                          scenario: str,
                          num_episodes: int,
                          save_videos: bool) -> Dict[str, Any]:
        """Evaluate model on a specific scenario"""
        
        # Create environment
        env = create_environment(
            scenario,
            enable_domain_randomization=False,
            enable_multi_agent=True,
            enable_reward_shaping=True
        )
        
        episode_rewards = []
        episode_lengths = []
        collision_count = 0
        success_count = 0
        
        videos = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            crashed = False
            
            episode_frames = []
            
            while not done:
                # Get action based on model type
                if model_type == 'imitation':
                    action = self._get_il_action(model, obs)
                elif model_type == 'ppo':
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Check for collision
                if info.get('crashed', False):
                    crashed = True
                
                # Save frame for video
                if save_videos:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if crashed:
                collision_count += 1
            else:
                success_count += 1
            
            if save_videos and episode_frames:
                videos.append(episode_frames)
            
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                  f"Length = {episode_length}, Crashed = {crashed}")
        
        # Save videos if requested
        if save_videos and videos:
            self._save_videos(videos, scenario, model_type)
        
        env.close()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / num_episodes,
            'collision_rate': collision_count / num_episodes,
            'num_episodes': num_episodes
        }
    
    def _get_il_action(self, model, obs):
        """Get action from imitation learning model"""
        model.eval()
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            if hasattr(model, 'device'):
                obs_tensor = obs_tensor.to(model.device)
            else:
                obs_tensor = obs_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            if obs_tensor.dtype == torch.uint8:
                obs_tensor = obs_tensor.float() / 255.0
            
            action = model(obs_tensor)
            return action.cpu().numpy().squeeze()
    
    def _save_videos(self, videos: List[List], scenario: str, model_type: str):
        """Save video recordings"""
        videos_dir = os.path.join(self.results_dir, 'videos', f"{model_type}_{scenario}")
        os.makedirs(videos_dir, exist_ok=True)
        
        for i, frames in enumerate(videos[:3]):  # Save first 3 episodes
            video_path = os.path.join(videos_dir, f"episode_{i+1}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved video: {video_path}")
    
    def _save_results(self, results: Dict[str, Any], model_type: str):
        """Save evaluation results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_evaluation_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {filepath}")
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualization plots"""
        model_type = results['model_type']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_type.upper()} Model Evaluation Results', fontsize=16)
        
        scenarios = list(results['scenarios'].keys())
        
        # 1. Reward comparison across scenarios
        rewards_data = []
        for scenario in scenarios:
            rewards_data.extend([
                (scenario, reward) for reward in results['scenarios'][scenario]['episode_rewards']
            ])
        
        scenario_names, reward_values = zip(*rewards_data)
        
        ax = axes[0, 0]
        box_data = [results['scenarios'][scenario]['episode_rewards'] for scenario in scenarios]
        ax.boxplot(box_data, labels=scenarios)
        ax.set_title('Reward Distribution by Scenario')
        ax.set_ylabel('Episode Reward')
        ax.grid(True)
        
        # 2. Episode length comparison
        ax = axes[0, 1]
        length_data = [results['scenarios'][scenario]['episode_lengths'] for scenario in scenarios]
        ax.boxplot(length_data, labels=scenarios)
        ax.set_title('Episode Length Distribution by Scenario')
        ax.set_ylabel('Episode Length')
        ax.grid(True)
        
        # 3. Success rate comparison
        ax = axes[0, 2]
        success_rates = [results['scenarios'][scenario]['success_rate'] for scenario in scenarios]
        bars = ax.bar(scenarios, success_rates, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Success Rate by Scenario')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.2%}', ha='center', va='bottom')
        
        # 4. Reward vs Episode Length scatter
        ax = axes[1, 0]
        colors = ['blue', 'green', 'red']
        for i, scenario in enumerate(scenarios):
            scenario_data = results['scenarios'][scenario]
            ax.scatter(scenario_data['episode_lengths'], 
                      scenario_data['episode_rewards'],
                      alpha=0.6, label=scenario, color=colors[i])
        
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Reward vs Episode Length')
        ax.legend()
        ax.grid(True)
        
        # 5. Performance metrics heatmap
        ax = axes[1, 1]
        metrics_data = []
        metric_names = ['Avg Reward', 'Success Rate', 'Avg Length']
        
        for scenario in scenarios:
            scenario_data = results['scenarios'][scenario]
            metrics_data.append([
                scenario_data['avg_reward'],
                scenario_data['success_rate'],
                scenario_data['avg_length']
            ])
        
        # Normalize metrics for better visualization
        metrics_array = np.array(metrics_data)
        normalized_metrics = (metrics_array - metrics_array.min(axis=0)) / (metrics_array.max(axis=0) - metrics_array.min(axis=0))
        
        sns.heatmap(normalized_metrics.T, 
                   xticklabels=scenarios,
                   yticklabels=metric_names,
                   annot=True, fmt='.2f', 
                   cmap='RdYlGn', ax=ax)
        ax.set_title('Normalized Performance Metrics')
        
        # 6. Overall statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        stats = results['overall_stats']
        stats_text = f"""
        Overall Performance Summary
        
        Average Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}
        Average Length: {stats['avg_length']:.1f} ± {stats['std_length']:.1f}
        Overall Success Rate: {stats['overall_success_rate']:.2%}
        Total Episodes: {stats['total_episodes']}
        
        Model Type: {model_type.upper()}
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{model_type}_evaluation_plots_{timestamp}.png"
        plot_path = os.path.join(self.results_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plots saved to: {plot_path}")


def compare_models(il_model_path: str, 
                  ppo_model_path: str,
                  scenarios: List[str] = None,
                  num_episodes: int = 20) -> Dict[str, Any]:
    """
    Compare imitation learning and PPO models
    
    Args:
        il_model_path: Path to trained IL model
        ppo_model_path: Path to trained PPO model
        scenarios: List of scenarios to test
        num_episodes: Number of episodes per scenario
        
    Returns:
        Comparison results
    """
    evaluator = ModelEvaluator()
    
    # Load IL model
    print("Loading Imitation Learning model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    il_checkpoint = torch.load(il_model_path, map_location=device)
    il_model = create_model('cnn', il_checkpoint['model_config'])
    il_model.load_state_dict(il_checkpoint['model_state_dict'])
    il_model.to(device)
    
    # Load PPO model
    print("Loading PPO model...")
    from stable_baselines3 import PPO
    ppo_model = PPO.load(ppo_model_path)
    
    # Evaluate both models
    il_results = evaluator.evaluate_model(il_model, 'imitation', scenarios, num_episodes)
    ppo_results = evaluator.evaluate_model(ppo_model, 'ppo', scenarios, num_episodes)
    
    # Generate comparison
    comparison = {
        'imitation_learning': il_results,
        'ppo': ppo_results,
        'comparison_summary': _generate_comparison_summary(il_results, ppo_results)
    }
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(evaluator.results_dir, f"model_comparison_{timestamp}.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"Comparison results saved to: {comparison_path}")
    
    return comparison


def _generate_comparison_summary(il_results: Dict[str, Any], 
                               ppo_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary comparison between IL and PPO"""
    
    il_stats = il_results['overall_stats']
    ppo_stats = ppo_results['overall_stats']
    
    return {
        'reward_improvement': ppo_stats['avg_reward'] - il_stats['avg_reward'],
        'success_rate_improvement': ppo_stats['overall_success_rate'] - il_stats['overall_success_rate'],
        'length_difference': ppo_stats['avg_length'] - il_stats['avg_length'],
        'better_model': 'PPO' if ppo_stats['avg_reward'] > il_stats['avg_reward'] else 'Imitation Learning',
        'reward_winner': 'PPO' if ppo_stats['avg_reward'] > il_stats['avg_reward'] else 'IL',
        'success_winner': 'PPO' if ppo_stats['overall_success_rate'] > il_stats['overall_success_rate'] else 'IL'
    }


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # This would require trained models
    # il_model_path = "models/best_model.pth"
    # ppo_model_path = "models/ppo_final_model.zip"
    # 
    # comparison = compare_models(il_model_path, ppo_model_path)
    # print("Model comparison completed!")