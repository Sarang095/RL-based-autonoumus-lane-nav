"""
CNN Model for Vision-to-Action Mapping in Autonomous Driving
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any

from ..config import Config


class DrivingCNN(nn.Module):
    """
    Convolutional Neural Network for mapping visual observations to driving actions
    
    Architecture based on the Atari DQN CNN with modifications for continuous control
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super(DrivingCNN, self).__init__()
        
        if config is None:
            config = Config.CNN_CONFIG
        
        self.config = config
        self.input_channels = config['input_channels']
        self.input_height = config['input_height']
        self.input_width = config['input_width']
        self.output_dim = config['output_dim']
        self.dropout_rate = config['dropout_rate']
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        
        for layer_config in config['conv_layers']:
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_config['out_channels'],
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=0
            )
            self.conv_layers.append(conv_layer)
            in_channels = layer_config['out_channels']
        
        # Calculate the size of flattened features after conv layers
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = self.conv_output_size
        
        for fc_size in config['fc_layers']:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            prev_size = fc_size
        
        # Output layer for continuous actions
        self.action_head = nn.Linear(prev_size, self.output_dim)
        
        # Value head for actor-critic architectures (optional)
        self.value_head = nn.Linear(prev_size, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            x = dummy_input
            
            for conv_layer in self.conv_layers:
                x = F.relu(conv_layer(x))
            
            return x.numel()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, return_value: bool = False) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            return_value: Whether to return value estimate (for actor-critic)
            
        Returns:
            actions: Predicted actions of shape (batch_size, action_dim)
            value: State value estimate (if return_value=True)
        """
        # Ensure input is float and normalized
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Convolutional layers with ReLU activation
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = F.relu(fc_layer(x))
            x = self.dropout(x)
        
        # Action prediction
        actions = self.action_head(x)
        
        # Apply tanh activation for bounded continuous actions
        # Steering: [-1, 1], Acceleration: [-1, 1]
        actions = torch.tanh(actions)
        
        if return_value:
            value = self.value_head(x)
            return actions, value
        
        return actions
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the network (before action head)"""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = F.relu(fc_layer(x))
        
        return x


class DrivingActorCritic(nn.Module):
    """
    Actor-Critic architecture for PPO training
    Shares convolutional features between actor and critic
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super(DrivingActorCritic, self).__init__()
        
        if config is None:
            config = Config.CNN_CONFIG
        
        self.config = config
        
        # Shared feature extractor
        self.feature_extractor = DrivingCNN(config)
        
        # Get feature size
        feature_size = config['fc_layers'][-1]
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(feature_size, config['output_dim']),
            nn.Tanh()
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(feature_size, 1)
        )
        
        # Action standard deviation (for stochastic policy)
        self.log_std = nn.Parameter(torch.zeros(config['output_dim']))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both action distribution and value
        
        Returns:
            action_mean: Mean of action distribution
            value: State value estimate
        """
        features = self.feature_extractor.get_features(x)
        
        action_mean = self.actor(features)
        value = self.critic(features)
        
        return action_mean, value
    
    def get_action_distribution(self, x: torch.Tensor):
        """Get action distribution for sampling"""
        action_mean, _ = self.forward(x)
        action_std = torch.exp(self.log_std)
        return torch.distributions.Normal(action_mean, action_std)
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO training"""
        action_mean, value = self.forward(x)
        action_std = torch.exp(self.log_std)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value.squeeze()


def create_model(model_type: str = 'cnn', config: Dict[str, Any] = None) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('cnn', 'actor_critic')
        config: Model configuration
        
    Returns:
        Initialized model
    """
    if model_type == 'cnn':
        return DrivingCNN(config)
    elif model_type == 'actor_critic':
        return DrivingActorCritic(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model('cnn')
    model.to(device)
    
    # Test input
    batch_size = 4
    input_tensor = torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8).to(device)
    
    # Forward pass
    with torch.no_grad():
        actions = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {actions.shape}")
        print(f"Actions range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    
    # Test actor-critic model
    print("\nTesting Actor-Critic model:")
    ac_model = create_model('actor_critic')
    ac_model.to(device)
    
    with torch.no_grad():
        action_mean, value = ac_model(input_tensor)
        print(f"Action mean shape: {action_mean.shape}")
        print(f"Value shape: {value.shape}")
        
        # Test action distribution
        dist = ac_model.get_action_distribution(input_tensor)
        sampled_actions = dist.sample()
        print(f"Sampled actions shape: {sampled_actions.shape}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")