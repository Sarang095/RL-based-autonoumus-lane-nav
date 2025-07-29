"""Simple test script to verify environment and model setup"""
import torch
import numpy as np
from src.environment_wrapper import create_environment # Import the actual create_environment
from src.models import create_model

def test_basic_setup():
    print("Testing basic setup...")
    
    env = create_environment('highway', enable_domain_randomization=False)
    
    obs, info = env.reset()
    print(f"✓ Environment observation shape: {obs.shape}")
    
    action = np.array([0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step successful, reward: {reward}")
    
    model = create_model('cnn')
    
    # obs from env should already be (stack_size, H, W)
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float() / 255.0
    
    with torch.no_grad():
        action_pred = model(obs_tensor)
    
    print(f"✓ Model prediction shape: {action_pred.shape}")
    print(f"✓ Model prediction range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")
    
    env.close()
    print("✓ Basic setup test completed successfully!")
    return True

if __name__ == "__main__":
    test_basic_setup()