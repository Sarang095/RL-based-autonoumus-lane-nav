"""
PPO (Proximal Policy Optimization) Training Module for Autonomous Driving
"""
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from typing import Dict, Any, Optional, Callable
from torch.utils.tensorboard import SummaryWriter

from ..environment_wrapper import create_environment
from ..config import Config
from ..models import DrivingActorCritic


class CustomCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for PPO using our DrivingCNN architecture
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim)
        
        # Use our custom CNN model configuration
        self.cnn_config = Config.CNN_CONFIG.copy()
        self.cnn_config['output_dim'] = features_dim  # Output features instead of actions
        
        # Extract CNN layers from our model
        self.conv_layers = torch.nn.ModuleList()
        in_channels = observation_space.shape[0]  # Number of stacked frames
        
        for layer_config in self.cnn_config['conv_layers']:
            conv_layer = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_config['out_channels'],
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=0
            )
            self.conv_layers.append(conv_layer)
            in_channels = layer_config['out_channels']
        
        # Calculate output size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            conv_output = self._forward_conv(dummy_input)
            conv_output_size = conv_output.numel()
        
        # Fully connected layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_output_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, features_dim),
            torch.nn.ReLU()
        )
    
    def _forward_conv(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers"""
        x = observations.float() / 255.0  # Normalize
        
        for conv_layer in self.conv_layers:
            x = torch.nn.functional.relu(conv_layer(x))
        
        return x
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor"""
        conv_features = self._forward_conv(observations)
        flattened = torch.flatten(conv_features, start_dim=1)
        return self.fc(flattened)


class CustomActorCriticPolicy(ActorCriticCnnPolicy):
    """
    Custom Actor-Critic policy using our CNN feature extractor
    """
    
    def __init__(self, *args, **kwargs):
        # Set our custom feature extractor
        kwargs['features_extractor_class'] = CustomCNNExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 512}
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)


class PPOTrainingCallback(BaseCallback):
    """
    Custom callback for PPO training to log additional metrics
    """
    
    def __init__(self, verbose: int = 0):
        super(PPOTrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                    # Log to tensorboard
                    self.logger.record('train/episode_reward', info['episode']['r'])
                    self.logger.record('train/episode_length', info['episode']['l'])
        
        return True


class PPOTrainer:
    """
    PPO trainer for autonomous driving with vision-based inputs
    """
    
    def __init__(self, 
                 env_name: str = 'highway',
                 ppo_config: Dict[str, Any] = None,
                 n_envs: int = 4,
                 pretrained_model_path: Optional[str] = None):
        
        self.env_name = env_name
        self.ppo_config = ppo_config or Config.PPO_CONFIG
        self.n_envs = n_envs
        self.pretrained_model_path = pretrained_model_path
        
        # Create vectorized environment
        self.env = self._create_vectorized_env()
        
        # Create PPO model
        self.model = self._create_ppo_model()
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
    def _create_vectorized_env(self):
        """Create vectorized environment for parallel training"""
        def make_env(rank: int):
            def _init():
                env = create_environment(
                    self.env_name,
                    enable_domain_randomization=True,
                    enable_multi_agent=True,
                    enable_reward_shaping=True
                )
                # Wrap with Monitor for episode statistics
                env = Monitor(env)
                return env
            return _init
        
        if self.n_envs == 1:
            return DummyVecEnv([make_env(0)])
        else:
            return SubprocVecEnv([make_env(i) for i in range(self.n_envs)])
    
    def _create_ppo_model(self):
        """Create PPO model with custom policy"""
        
        # PPO configuration
        ppo_kwargs = {
            'policy': CustomActorCriticPolicy,
            'env': self.env,
            'learning_rate': self.ppo_config['learning_rate'],
            'n_steps': self.ppo_config['n_steps'],
            'batch_size': self.ppo_config['batch_size'],
            'n_epochs': self.ppo_config['n_epochs'],
            'gamma': self.ppo_config['gamma'],
            'gae_lambda': self.ppo_config['gae_lambda'],
            'clip_range': self.ppo_config['clip_range'],
            'ent_coef': self.ppo_config['ent_coef'],
            'vf_coef': self.ppo_config['vf_coef'],
            'max_grad_norm': self.ppo_config['max_grad_norm'],
            'tensorboard_log': os.path.join(Config.PATHS['logs_dir'], 'ppo'),
            'verbose': 1,
            'device': 'auto'
        }
        
        model = PPO(**ppo_kwargs)
        
        # Load pretrained weights if available
        if self.pretrained_model_path and os.path.exists(self.pretrained_model_path):
            print(f"Loading pretrained model from {self.pretrained_model_path}")
            self._load_pretrained_weights(model)
        
        return model
    
    def _load_pretrained_weights(self, model):
        """Load pretrained weights from imitation learning model"""
        try:
            # Load the imitation learning checkpoint
            checkpoint = torch.load(self.pretrained_model_path, map_location='cpu')
            
            # Extract the state dict
            il_state_dict = checkpoint['model_state_dict']
            
            # Get the PPO policy network
            ppo_state_dict = model.policy.state_dict()
            
            # Map imitation learning weights to PPO policy
            # This is a simplified mapping - in practice you might need more careful alignment
            mapped_weights = {}
            
            for ppo_key, ppo_param in ppo_state_dict.items():
                # Try to find corresponding weight in IL model
                if 'features_extractor' in ppo_key:
                    # Map CNN feature extractor weights
                    il_key = ppo_key.replace('features_extractor.', '')
                    if il_key in il_state_dict:
                        mapped_weights[ppo_key] = il_state_dict[il_key]
                
                # For actor head, we can initialize from IL action head
                elif 'action_net' in ppo_key and 'action_head' in il_state_dict:
                    il_key = ppo_key.replace('action_net', 'action_head')
                    if il_key in il_state_dict:
                        mapped_weights[ppo_key] = il_state_dict[il_key]
            
            # Load mapped weights
            if mapped_weights:
                ppo_state_dict.update(mapped_weights)
                model.policy.load_state_dict(ppo_state_dict, strict=False)
                print(f"Successfully loaded {len(mapped_weights)} pretrained weights")
            else:
                print("No compatible weights found for transfer")
                
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Continuing with random initialization")
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Custom training callback
        callbacks.append(PPOTrainingCallback())
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=Config.TRAINING_CONFIG['save_interval'],
            save_path=Config.PATHS['models_dir'],
            name_prefix='ppo_checkpoint'
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = create_environment(
            self.env_name,
            enable_domain_randomization=False,
            enable_multi_agent=False,
            enable_reward_shaping=True
        )
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=Config.PATHS['models_dir'],
            log_path=os.path.join(Config.PATHS['logs_dir'], 'eval'),
            eval_freq=Config.TRAINING_CONFIG['eval_interval'],
            n_eval_episodes=Config.TRAINING_CONFIG['eval_episodes'],
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        return callbacks
    
    def train(self, total_timesteps: int = None):
        """
        Train the PPO model
        
        Args:
            total_timesteps: Total number of timesteps to train for
        """
        if total_timesteps is None:
            total_timesteps = self.ppo_config['total_timesteps']
        
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        print(f"Environment: {self.env_name}")
        print(f"Number of parallel environments: {self.n_envs}")
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            progress_bar=True
        )
        
        # Save final model
        model_path = os.path.join(Config.PATHS['models_dir'], 'ppo_final_model.zip')
        self.model.save(model_path)
        print(f"Final model saved to {model_path}")
        
        return self.model
    
    def evaluate(self, num_episodes: int = 10, render: bool = False):
        """
        Evaluate the trained model
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating model for {num_episodes} episodes...")
        
        # Create evaluation environment
        eval_env = create_environment(
            self.env_name,
            enable_domain_randomization=False,
            enable_multi_agent=True,
            enable_reward_shaping=True
        )
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check if episode was successful (no crash)
            if not info.get('crashed', False):
                success_count += 1
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Calculate statistics
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_count / num_episodes,
            'num_episodes': num_episodes
        }
        
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Average Length: {results['avg_length']:.2f} ± {results['std_length']:.2f}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        
        eval_env.close()
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = PPO.load(filepath, env=self.env)
        print(f"Model loaded from {filepath}")


def train_ppo_model(env_name: str = 'highway',
                   total_timesteps: int = None,
                   n_envs: int = 4,
                   pretrained_model_path: Optional[str] = None) -> PPOTrainer:
    """
    Convenience function to train a PPO model
    
    Args:
        env_name: Environment name
        total_timesteps: Total timesteps to train
        n_envs: Number of parallel environments
        pretrained_model_path: Path to pretrained IL model
        
    Returns:
        Trained PPOTrainer
    """
    trainer = PPOTrainer(
        env_name=env_name,
        n_envs=n_envs,
        pretrained_model_path=pretrained_model_path
    )
    
    trainer.train(total_timesteps=total_timesteps)
    return trainer


if __name__ == "__main__":
    # Test PPO training
    trainer = train_ppo_model(
        env_name='highway',
        total_timesteps=100_000,  # Reduced for testing
        n_envs=2,
        pretrained_model_path=None  # Set path to IL model if available
    )
    
    # Evaluate the trained model
    results = trainer.evaluate(num_episodes=5)
    print(f"Final evaluation: {results['avg_reward']:.2f} average reward")