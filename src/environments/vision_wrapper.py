"""
Vision Wrapper for Highway-env
Converts highway-env observations to vision-based observations for CNN training.
"""

import gymnasium as gym
import numpy as np
import cv2
from typing import Any, Dict, Optional, Tuple
import highway_env

class VisionWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts highway-env observations to vision-based observations.
    Supports both RGB and grayscale image observations with configurable preprocessing.
    """
    
    def __init__(
        self,
        env: gym.Env,
        observation_type: str = "rgb",
        image_shape: Tuple[int, int] = (84, 84),
        stack_frames: int = 4,
        normalize: bool = True,
        enhance_contrast: bool = True
    ):
        """
        Initialize the vision wrapper.
        
        Args:
            env: The highway environment to wrap
            observation_type: "rgb" or "grayscale"
            image_shape: Target image dimensions (height, width)
            stack_frames: Number of frames to stack for temporal information
            normalize: Whether to normalize pixel values to [0, 1]
            enhance_contrast: Whether to apply contrast enhancement
        """
        super().__init__(env)
        
        self.observation_type = observation_type
        self.image_shape = image_shape
        self.stack_frames = stack_frames
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        
        # Don't modify the environment configuration here
        # We'll wrap the kinematics observation and convert to vision
        
        # Set up observation space
        if observation_type == "grayscale":
            channels = stack_frames
        else:
            channels = 3 * stack_frames
            
        self.observation_space = gym.spaces.Box(
            low=0.0 if normalize else 0,
            high=1.0 if normalize else 255,
            shape=(channels, *image_shape),
            dtype=np.float32 if normalize else np.uint8
        )
        
        # Frame buffer for stacking
        self.frame_buffer = []
        
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Process the observation from the environment.
        
        Args:
            observation: Raw observation from highway-env (kinematics)
            
        Returns:
            Processed vision observation ready for CNN input
        """
        # Render the environment to get visual frame
        try:
            frame = self.env.render()
        except Exception:
            # Fallback: create a simple visual representation
            frame = self._create_simple_visual(observation)
            
        # Convert to numpy array if needed
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
            
        # Handle different frame shapes
        if len(frame.shape) == 3:  # RGB or BGR
            if frame.shape[2] == 4:  # RGBA
                frame = frame[:, :, :3]  # Remove alpha channel
            if self.observation_type == "grayscale":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif len(frame.shape) == 2:  # Grayscale
            if self.observation_type == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
        # Resize to target shape
        frame = cv2.resize(frame, self.image_shape, interpolation=cv2.INTER_AREA)
        
        # Enhance contrast if requested
        if self.enhance_contrast:
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.equalizeHist(frame)
            else:  # RGB
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                
        # Normalize if requested
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
            
        # Add to frame buffer
        if len(self.frame_buffer) == 0:
            # Initialize buffer with repeated frames
            self.frame_buffer = [frame] * self.stack_frames
        else:
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.stack_frames:
                self.frame_buffer.pop(0)
                
        # Stack frames
        if self.observation_type == "grayscale":
            # Stack grayscale frames along channel dimension
            stacked = np.stack(self.frame_buffer, axis=0)
        else:
            # Stack RGB frames along channel dimension
            if len(frame.shape) == 2:  # Grayscale but need RGB output
                frame_rgb = np.stack([frame] * 3, axis=2)
                self.frame_buffer[-1] = frame_rgb
            stacked = np.concatenate(self.frame_buffer, axis=2)
            stacked = np.transpose(stacked, (2, 0, 1))  # HWC -> CHW
            
        return stacked.astype(np.float32 if self.normalize else np.uint8)
    
    def _create_simple_visual(self, observation: np.ndarray) -> np.ndarray:
        """Create a simple visual representation from kinematics observation."""
        # Create a simple top-down view from kinematics data
        height, width = self.image_shape
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw road (gray background)
        frame[:, :] = [64, 64, 64]
        
        # Draw lane lines (white)
        for i in range(1, 4):  # 3 lanes
            x = int(width * i / 4)
            frame[:, x-1:x+1] = [255, 255, 255]
        
        # If we have vehicle position data, draw vehicles
        if observation.shape[0] > 0 and observation.shape[1] >= 3:
            for i, vehicle in enumerate(observation):
                if vehicle[0] > 0:  # Vehicle present
                    x = int(width * 0.5 + vehicle[1] * 10)  # Scale position
                    y = int(height * 0.5 - vehicle[2] * 10)
                    x = np.clip(x, 5, width - 5)
                    y = np.clip(y, 5, height - 5)
                    
                    # Draw vehicle (different colors)
                    color = [255, 0, 0] if i == 0 else [0, 255, 0]  # Red for ego, green for others
                    frame[y-2:y+2, x-2:x+2] = color
        
        return frame
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and clear frame buffer."""
        obs, info = self.env.reset(**kwargs)
        self.frame_buffer = []
        processed_obs = self.observation(obs)
        return processed_obs, info

class HighwayVisionEnv:
    """
    Factory class for creating vision-based highway environments.
    """
    
    @staticmethod
    def create_env(
        env_name: str = "highway-v0",
        observation_type: str = "rgb",
        image_shape: Tuple[int, int] = (84, 84),
        stack_frames: int = 4,
        normalize: bool = True,
        **env_kwargs
    ) -> gym.Env:
        """
        Create a vision-wrapped highway environment.
        
        Args:
            env_name: Name of the highway environment
            observation_type: "rgb" or "grayscale"
            image_shape: Target image dimensions
            stack_frames: Number of frames to stack
            normalize: Whether to normalize pixel values
            **env_kwargs: Additional environment configuration
            
        Returns:
            Wrapped environment ready for training
        """
        # Create base environment
        env = gym.make(env_name, render_mode="rgb_array")
        
        # Configure environment with minimal changes
        config = {
            "render_mode": "rgb_array",  # Enable rendering
            "screen_width": 600,
            "screen_height": 150,
            "collision_reward": -100,
            "lane_change_reward": 0,
            "high_speed_reward": 1,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            **env_kwargs
        }
        
        env.configure(config)
        
        # Wrap with vision wrapper
        env = VisionWrapper(
            env,
            observation_type=observation_type,
            image_shape=image_shape,
            stack_frames=stack_frames,
            normalize=normalize
        )
        
        return env