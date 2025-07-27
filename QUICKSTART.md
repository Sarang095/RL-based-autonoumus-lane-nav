# 🚗 Webots Autonomous Driving - Quick Start Guide

Get your autonomous driving simulation running in just a few steps!

## 🎯 Prerequisites

1. **Webots R2023a or later** - Download from [cyberbotics.com](https://cyberbotics.com/)
2. **Python 3.7+** with pip
3. **Git** (if cloning from repository)

## ⚡ Quick Setup (30 seconds)

### Option 1: Automated Setup
```bash
# Run the automated setup script
python setup_project.py

# Test that everything works
python test_setup.py
```

### Option 2: Manual Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p collected_data/{images,actions,sensor_data} training_logs models results
```

## 🚀 Launch the Simulation

### Basic Launch
```bash
python launch_simulation.py
```

### Advanced Options
```bash
# Fast mode (no graphics)
python launch_simulation.py --mode fast

# Check dependencies
python launch_simulation.py --check-deps

# Custom world file
python launch_simulation.py --world path/to/your/world.wbt
```

## 🎮 Controls

Once the simulation is running:

- **W/S**: Throttle forward/backward
- **A/D**: Steering left/right  
- **M**: Toggle between Manual and AI mode
- **Q**: Quit the simulation

## 🤖 Training an AI Agent

### Start RL Training
```bash
python train_rl.py
```

### Start Training with Custom Parameters
```bash
# Edit train_rl.py to modify:
# - Learning rate
# - Training episodes
# - Reward function
# - Network architecture
```

## 📊 Monitor Training

### TensorBoard (Real-time Metrics)
```bash
# In a separate terminal
tensorboard --logdir training_logs
# Open http://localhost:6006 in your browser
```

### Training Logs
- Console output shows episode rewards
- Models saved to `training_logs/`
- Best model: `training_logs/final_model.zip`

## 🎯 Project Workflow

### Phase 1: Data Collection (Imitation Learning)
1. Launch simulation: `python launch_simulation.py`
2. Drive manually using W/A/S/D keys
3. Training data automatically saved to `collected_data/`

### Phase 2: AI Training (Reinforcement Learning)  
1. Run training: `python train_rl.py`
2. Monitor with TensorBoard
3. Wait for training to complete

### Phase 3: AI Testing
1. Launch simulation: `python launch_simulation.py`
2. Press 'M' to switch to AI mode
3. Watch the AI drive autonomously!

## 📁 Project Structure

```
webots-autonomous-driving/
├── worlds/
│   └── autonomous_driving.wbt      # Main world file
├── controllers/
│   ├── agent_driver/
│   │   └── agent_driver.py        # Agent controller
│   └── supervisor_controller/
│       └── supervisor_controller.py # Supervisor controller
├── collected_data/                 # Manual driving data
├── training_logs/                  # RL training results  
├── models/                        # Saved models
├── launch_simulation.py           # Main launcher
├── train_rl.py                    # Training script
└── requirements.txt               # Dependencies
```

## 🔧 Troubleshooting

### Common Issues

**"Webots not found"**
```bash
# Add Webots to your PATH, or install from cyberbotics.com
export PATH="/usr/local/webots:$PATH"  # Linux/Mac
```

**"Module not found"**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**"AGENT_CAR not found"**
- Make sure you're using the provided world file
- The robot must be defined with `DEF AGENT_CAR` in Webots

**Training is slow**
```bash
# Use fast mode
python launch_simulation.py --mode fast

# Or reduce episode length in train_rl.py
max_episode_steps = 500  # Instead of 1000
```

### Performance Tips

1. **Use GPU for training**: Install CUDA-compatible PyTorch
2. **Reduce image resolution**: Edit observation space in `train_rl.py`
3. **Adjust training parameters**: Learning rate, batch size, etc.
4. **Monitor with TensorBoard**: Track training progress

## 🎓 Learning Resources

### Scenarios Available
- **Parking**: Navigate to parking spots
- **Intersection**: Handle traffic lights and turns
- **Roundabout**: Navigate circular intersections

### Extending the Project
- Add new scenarios in `train_rl.py`
- Modify reward function for different behaviors
- Add new sensors to the car model
- Create custom world files

## 📞 Getting Help

1. **Test your setup**: `python test_setup.py`
2. **Check the full README**: `README.md`
3. **Inspect the code**: All files are well-documented
4. **Webots documentation**: [cyberbotics.com/doc](https://cyberbotics.com/doc)

---

## 🎉 Ready to Go!

You should now have a fully functional autonomous driving simulation. Start with manual driving to collect data, then train an AI agent to take over!

**Happy autonomous driving! 🚗💨**