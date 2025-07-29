"""
Simple test to verify basic functionality without complex wrappers
"""
import torch
import numpy as np
import highway_env
import gymnasium as gym
from src.models import create_model
from src.config import Config

def test_basic_setup():
    """Test basic environment and model functionality"""
    print("Testing basic setup...")
    
    # Create simple environment
    env = gym.make('highway-v0', render_mode='rgb_array')
    
    # Configure for vision
    config = {
        'observation': {
            'type': 'GrayscaleObservation',
            'observation_shape': (84, 84),
            'stack_size': 1,
            'weights': [0.2989, 0.5870, 0.1140],
        },
        'action': {'type': 'ContinuousAction'},
        'vehicles_count': 10,
    }
    env.unwrapped.configure(config)
    
    # Test environment
    obs, info = env.reset()
    print(f"✓ Environment observation shape: {obs.shape}")
    
    action = np.array([0.0, 0.0])  # No acceleration, no steering
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step successful, reward: {reward}")
    
    # Test model
    model = create_model('cnn')
    
    # Manually stack 4 frames for the model
    # obs shape is (1, 84, 84), we need (4, 84, 84) for 4-frame stack
    frame = obs[0]  # Shape (84, 84)
    stacked_obs = np.stack([frame, frame, frame, frame], axis=0)  # (4, 84, 84)
    obs_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).float() / 255.0  # (1, 4, 84, 84)
    
    with torch.no_grad():
        action_pred = model(obs_tensor)
    
    print(f"✓ Model prediction shape: {action_pred.shape}")
    print(f"✓ Model prediction range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")
    
    env.close()
    print("✓ Basic setup test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_basic_setup()
        print("\n🎉 BASIC SETUP WORKS!")
        print("The core components are functional.")
        print("Environment configuration issue in wrapper needs fixing,")
        print("but the basic pipeline works!")
    except Exception as e:
        print(f"\n❌ Basic setup failed: {e}")
        import traceback
        traceback.print_exc()