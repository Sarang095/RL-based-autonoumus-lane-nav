"""
PPO Policy for Vision-Based Autonomous Driving
Integration with Stable-Baselines3 PPO for reinforcement learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
import gymnasium as gym

from .cnn_policy import VisionCNN


class VisionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Vision-based feature extractor for PPO using our custom CNN architecture.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        use_attention: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the vision features extractor.
        
        Args:
            observation_space: Observation space of the environment
            features_dim: Number of features extracted by the CNN
            use_attention: Whether to use attention mechanisms
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        if isinstance(observation_space, gym.spaces.Box):
            n_input_channels = observation_space.shape[0]
            height = observation_space.shape[1]
            width = observation_space.shape[2]
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")
        
        # Create vision CNN
        self.cnn = VisionCNN(
            input_channels=n_input_channels,
            input_height=height,
            input_width=width,
            hidden_dim=features_dim,
            use_attention=use_attention,
            dropout_rate=dropout_rate
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Input observations
            
        Returns:
            Extracted features
        """
        return self.cnn(observations)


class PPOVisionPolicy(ActorCriticPolicy):
    """
    PPO policy using vision-based feature extraction.
    Integrates our custom CNN with Stable-Baselines3 PPO.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_attention: bool = True,
        dropout_rate: float = 0.1,
        *args,
        **kwargs
    ):
        """
        Initialize PPO vision policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture for actor/critic heads
            activation_fn: Activation function
            use_attention: Whether to use attention in CNN
            dropout_rate: Dropout rate
        """
        
        # Store custom parameters
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # Default network architecture if not provided
        if net_arch is None:
            net_arch = [dict(pi=[256, 128], vf=[256, 128])]
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """
        Build the feature extractor and MLP layers.
        """
        # Override to use our custom vision feature extractor
        self.features_extractor = VisionFeaturesExtractor(
            self.observation_space,
            features_dim=self.features_dim,
            use_attention=self.use_attention,
            dropout_rate=self.dropout_rate
        )
        
        # Call parent to build MLP layers
        super()._build_mlp_extractor()
    
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations using our CNN.
        
        Args:
            obs: Input observations
            
        Returns:
            Extracted features
        """
        return self.features_extractor(obs)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Args:
            obs: Input observations
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get latent representations
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # Get values
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            obs: Input observations
            actions: Actions to evaluate
            
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get latent representations
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Get values
        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict values for given observations.
        
        Args:
            obs: Input observations
            
        Returns:
            Predicted values
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get critic latent representation
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        # Get values
        return self.value_net(latent_vf)


class MultiHeadPPOVisionPolicy(PPOVisionPolicy):
    """
    Extended PPO policy with auxiliary prediction heads for additional supervision.
    """
    
    def __init__(self, *args, use_auxiliary_heads: bool = True, **kwargs):
        """
        Initialize multi-head PPO policy.
        
        Args:
            use_auxiliary_heads: Whether to use auxiliary prediction heads
        """
        self.use_auxiliary_heads = use_auxiliary_heads
        super().__init__(*args, **kwargs)
        
        # Add auxiliary heads if requested
        if self.use_auxiliary_heads:
            self._build_auxiliary_heads()
    
    def _build_auxiliary_heads(self):
        """Build auxiliary prediction heads."""
        features_dim = self.features_dim
        
        # Speed prediction head
        self.speed_head = nn.Sequential(
            nn.Linear(features_dim, features_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(features_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Lane prediction head
        self.lane_head = nn.Sequential(
            nn.Linear(features_dim, features_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(features_dim // 4, 3)  # Left, center, right lane
        )
        
        # Collision risk prediction head
        self.collision_risk_head = nn.Sequential(
            nn.Linear(features_dim, features_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(features_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward_with_auxiliary(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with auxiliary predictions.
        
        Args:
            obs: Input observations
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary containing all outputs including auxiliary predictions
        """
        # Standard forward pass
        actions, values, log_prob = self.forward(obs, deterministic)
        
        output = {
            "actions": actions,
            "values": values,
            "log_prob": log_prob
        }
        
        # Add auxiliary predictions if enabled
        if self.use_auxiliary_heads:
            features = self.extract_features(obs)
            
            output.update({
                "speed_pred": self.speed_head(features),
                "lane_pred": torch.softmax(self.lane_head(features), dim=-1),
                "collision_risk_pred": self.collision_risk_head(features)
            })
        
        return output
    
    def compute_auxiliary_loss(
        self,
        obs: torch.Tensor,
        speed_targets: Optional[torch.Tensor] = None,
        lane_targets: Optional[torch.Tensor] = None,
        collision_targets: Optional[torch.Tensor] = None,
        aux_loss_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute auxiliary losses for additional supervision.
        
        Args:
            obs: Input observations
            speed_targets: Target speed values
            lane_targets: Target lane indices
            collision_targets: Target collision risk values
            aux_loss_weight: Weight for auxiliary losses
            
        Returns:
            Combined auxiliary loss
        """
        if not self.use_auxiliary_heads:
            return torch.tensor(0.0, device=obs.device)
        
        features = self.extract_features(obs)
        total_aux_loss = 0.0
        
        # Speed prediction loss
        if speed_targets is not None:
            speed_pred = self.speed_head(features)
            speed_loss = nn.MSELoss()(speed_pred.squeeze(), speed_targets)
            total_aux_loss += speed_loss
        
        # Lane prediction loss
        if lane_targets is not None:
            lane_pred = self.lane_head(features)
            lane_loss = nn.CrossEntropyLoss()(lane_pred, lane_targets.long())
            total_aux_loss += lane_loss
        
        # Collision risk prediction loss
        if collision_targets is not None:
            collision_pred = self.collision_risk_head(features)
            collision_loss = nn.BCELoss()(collision_pred.squeeze(), collision_targets.float())
            total_aux_loss += collision_loss
        
        return aux_loss_weight * total_aux_loss


class VisionPPOConfig:
    """Configuration class for Vision PPO training."""
    
    def __init__(
        self,
        # PPO parameters
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        
        # Vision-specific parameters
        use_attention: bool = True,
        dropout_rate: float = 0.1,
        features_dim: int = 512,
        
        # Auxiliary learning parameters
        use_auxiliary_heads: bool = False,
        aux_loss_weight: float = 0.1,
        
        # Training parameters
        total_timesteps: int = 1000000,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        
        # Environment parameters
        n_envs: int = 8,
        env_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Vision PPO configuration.
        
        Args:
            learning_rate: Learning rate for optimization
            n_steps: Number of steps per environment per update
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            use_attention: Whether to use attention in CNN
            dropout_rate: Dropout rate
            features_dim: Feature dimension of CNN
            use_auxiliary_heads: Whether to use auxiliary prediction heads
            aux_loss_weight: Weight for auxiliary losses
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_freq: Model saving frequency
            n_envs: Number of parallel environments
            env_kwargs: Environment configuration
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.features_dim = features_dim
        
        self.use_auxiliary_heads = use_auxiliary_heads
        self.aux_loss_weight = aux_loss_weight
        
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        
        self.n_envs = n_envs
        self.env_kwargs = env_kwargs or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "use_attention": self.use_attention,
            "dropout_rate": self.dropout_rate,
            "features_dim": self.features_dim,
            "use_auxiliary_heads": self.use_auxiliary_heads,
            "aux_loss_weight": self.aux_loss_weight,
            "total_timesteps": self.total_timesteps,
            "eval_freq": self.eval_freq,
            "save_freq": self.save_freq,
            "n_envs": self.n_envs,
            "env_kwargs": self.env_kwargs
        }