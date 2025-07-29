"""
Test script to verify the project setup works correctly
"""
import torch
import numpy as np
from src.environment_wrapper import create_environment
from src.models import create_model
from src.config import Config

def test_environment():
    """Test environment creation and basic functionality"""
    print("Testing environment creation...")
    
    try:
        env = create_environment('highway', enable_domain_randomization=False)
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test reset and step
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Reward: {reward}, Terminated: {terminated}")
        
        env.close()
        print("✓ Environment test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}\n")
        return False

def test_model():
    """Test model creation and forward pass"""
    print("Testing model creation...")
    
    try:
        model = create_model('cnn')
        print(f"✓ CNN model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randint(0, 255, (batch_size, 4, 84, 84), dtype=torch.uint8)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Test actor-critic model
        ac_model = create_model('actor_critic')
        print(f"✓ Actor-Critic model created successfully")
        
        with torch.no_grad():
            action_mean, value = ac_model(dummy_input)
        print(f"  Action mean shape: {action_mean.shape}")
        print(f"  Value shape: {value.shape}")
        
        print("✓ Model test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}\n")
        return False

def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    
    try:
        # Test environment config
        highway_config = Config.get_env_config('highway')
        print(f"✓ Highway config loaded: {len(highway_config)} parameters")
        
        # Test observation shape
        obs_shape = Config.get_observation_shape('highway')
        print(f"✓ Observation shape: {obs_shape}")
        
        # Test all major config sections
        sections = ['CNN_CONFIG', 'IL_CONFIG', 'PPO_CONFIG', 'REWARD_CONFIG']
        for section in sections:
            config = getattr(Config, section)
            print(f"✓ {section}: {len(config)} parameters")
        
        print("✓ Configuration test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}\n")
        return False

def test_integration():
    """Test integration between environment and model"""
    print("Testing environment-model integration...")
    
    try:
        # Create environment and model
        env = create_environment('highway', enable_domain_randomization=False)
        model = create_model('cnn')
        model.eval()
        
        # Test episode
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 10  # Short test
        
        for step in range(max_steps):
            # Get action from model
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            if obs_tensor.dtype == torch.uint8:
                obs_tensor = obs_tensor.float() / 255.0
            
            with torch.no_grad():
                action = model(obs_tensor)
                action = action.cpu().numpy().squeeze()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        env.close()
        
        print(f"✓ Integration test successful")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print("✓ Integration test completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("VISION-BASED AUTONOMOUS DRIVING - SETUP TEST")
    print("="*60)
    
    print(f"Python version: {torch.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    tests = [
        ("Configuration", test_config),
        ("Environment", test_environment),
        ("Model", test_model),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The project setup is working correctly.")
        print("\nYou can now run:")
        print("  python main.py --help           # See training options")
        print("  python demo.py --help           # See demo options")
        print("  python main.py --phases il      # Start with imitation learning")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()