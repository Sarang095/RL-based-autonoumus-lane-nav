"""
CNN Policy Network for Vision-Based Autonomous Driving
Implements a sophisticated CNN architecture with attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important road regions."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate attention weights
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature selection."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)


class ConvBlock(nn.Module):
    """Convolutional block with attention and residual connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.channel_attention = ChannelAttention(out_channels) if use_attention else None
        self.spatial_attention = SpatialAttention(out_channels) if use_attention else None
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention
        if self.channel_attention:
            out = self.channel_attention(out)
        if self.spatial_attention:
            out = self.spatial_attention(out)
            
        # Residual connection
        if self.use_residual:
            out += identity
            
        out = self.relu(out)
        return out


class VisionCNN(nn.Module):
    """
    Advanced CNN architecture for processing vision-based driving observations.
    Designed to extract hierarchical features from camera images.
    """
    
    def __init__(
        self,
        input_channels: int = 12,  # 4 frames * 3 RGB channels
        input_height: int = 84,
        input_width: int = 84,
        hidden_dim: int = 512,
        use_attention: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the vision CNN.
        
        Args:
            input_channels: Number of input channels (frames * RGB channels)
            input_height: Input image height
            input_width: Input image width
            hidden_dim: Hidden layer dimension
            use_attention: Whether to use attention mechanisms
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block: Extract low-level features
            ConvBlock(input_channels, 32, kernel_size=5, stride=2, padding=2, use_attention=use_attention),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block: Extract mid-level features
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1, use_attention=use_attention),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, use_attention=use_attention),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block: Extract high-level features
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, use_attention=use_attention),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, use_attention=use_attention),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block: Final feature extraction
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1, use_attention=use_attention),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        
        # Calculate feature dimension after convolutions
        self.feature_dim = 256 * 3 * 3
        
        # Feature projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision CNN.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, hidden_dim)
        """
        # Extract features
        features = self.features(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Project to hidden dimension
        features = self.projection(features)
        
        return features


class CNNPolicy(nn.Module):
    """
    CNN-based policy network for autonomous driving.
    Combines vision processing with action prediction.
    """
    
    def __init__(
        self,
        input_channels: int = 12,
        input_height: int = 84,
        input_width: int = 84,
        action_dim: int = 5,  # Highway-env discrete actions
        hidden_dim: int = 512,
        use_attention: bool = True,
        dropout_rate: float = 0.1,
        action_type: str = "discrete"
    ):
        """
        Initialize the CNN policy.
        
        Args:
            input_channels: Number of input channels
            input_height: Input image height
            input_width: Input image width
            action_dim: Number of discrete actions or continuous action dimension
            hidden_dim: Hidden layer dimension
            use_attention: Whether to use attention mechanisms
            dropout_rate: Dropout rate
            action_type: "discrete" or "continuous"
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.action_type = action_type
        
        # Vision backbone
        self.backbone = VisionCNN(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            hidden_dim=hidden_dim,
            use_attention=use_attention,
            dropout_rate=dropout_rate
        )
        
        # Value function head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Action heads
        if action_type == "discrete":
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        else:  # continuous
            self.action_mean = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Tanh()
            )
            self.action_logstd = nn.Parameter(torch.zeros(action_dim))
            
        # Auxiliary prediction heads for auxiliary tasks
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Lane prediction head - highway-env can have more than 3 lanes
        # Typical highway scenarios can have 4-5 lanes
        self.lane_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 5)  # Support up to 5 lanes (0-4)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_aux: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            x: Input observation tensor
            return_aux: Whether to return auxiliary predictions
            
        Returns:
            Dictionary containing policy outputs
        """
        # Extract features
        features = self.backbone(x)
        
        # Compute value
        value = self.value_head(features)
        
        # Compute action distribution
        if self.action_type == "discrete":
            action_logits = self.action_head(features)
            action_probs = F.softmax(action_logits, dim=-1)
            output = {
                "action_logits": action_logits,
                "action_probs": action_probs,
                "value": value,
                "features": features
            }
        else:  # continuous
            action_mean = self.action_mean(features)
            action_std = torch.exp(self.action_logstd).expand_as(action_mean)
            output = {
                "action_mean": action_mean,
                "action_std": action_std,
                "value": value,
                "features": features
            }
            
        # Auxiliary predictions
        if return_aux:
            output["speed_pred"] = self.speed_head(features)
            output["lane_pred"] = F.softmax(self.lane_head(features), dim=-1)
            
        return output
    
    def get_action(
        self, 
        x: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            x: Input observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_prob)
        """
        with torch.no_grad():
            output = self.forward(x)
            
            if self.action_type == "discrete":
                if deterministic:
                    action = torch.argmax(output["action_probs"], dim=-1)
                    log_prob = torch.log(output["action_probs"].gather(1, action.unsqueeze(1)))
                else:
                    action_dist = torch.distributions.Categorical(output["action_probs"])
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
            else:  # continuous
                action_dist = torch.distributions.Normal(output["action_mean"], output["action_std"])
                if deterministic:
                    action = output["action_mean"]
                else:
                    action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
                
        return action, log_prob
    
    def evaluate_actions(
        self, 
        x: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            x: Input observations
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        output = self.forward(x)
        values = output["value"]
        
        if self.action_type == "discrete":
            action_probs = output["action_probs"]
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze(1)
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        else:  # continuous
            action_dist = torch.distributions.Normal(output["action_mean"], output["action_std"])
            log_probs = action_dist.log_prob(actions).sum(dim=-1)
            entropy = action_dist.entropy().sum(dim=-1)
            
        return log_probs, values, entropy