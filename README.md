# Webots Autonomous Driving Simulation

🚗 **A complete autonomous driving project ready for Webots!** 

This project provides a comprehensive autonomous driving simulation environment with reinforcement learning, imitation learning, and data collection capabilities.

## ✅ Project Status
- **Dependencies**: ✅ All fixed (gym → gymnasium, opencv-python)
- **Webots Integration**: ✅ Ready to run
- **Controllers**: ✅ Working (agent + supervisor)
- **Training Pipeline**: ✅ PPO with Stable-Baselines3
- **Test Mode**: ✅ Runs without Webots for development

## 🚀 Quick Start

### Option 1: With Webots (Full Simulation)
```bash
# 1. Install Webots from https://cyberbotics.com/
# 2. Run the launcher
python3 launch_simulation.py
```

### Option 2: Without Webots (Testing Mode)
```bash
# Test controllers without Webots
python3 launch_simulation.py
# Choose 'y' for standalone mode when prompted
```

## 📋 Requirements

### Python Dependencies (✅ Auto-installed)
```bash
pip install -r requirements.txt
```
- numpy, opencv-python, gymnasium
- stable-baselines3, torch, matplotlib, tensorboard

### Webots (Required for full simulation)
- **Download**: https://cyberbotics.com/
- **Versions**: R2023a, R2023b, R2024a+
- **Platforms**: Windows, macOS, Linux

## 📁 Project Structure
```
webots_autonomous_driving/
├── 🌍 worlds/
│   ├── highway_drive.wbt      # Main simulation world
│   └── autonomous_driving.wbt # Alternative world
├── 🎮 controllers/
│   ├── agent_driver/          # Car controller (AI + manual)
│   └── supervisor_controller/ # Environment manager
├── 📊 collected_data/         # Training data
│   ├── images/               # Camera feeds
│   ├── actions/              # Steering/throttle
│   └── sensor_data/          # Distance sensors
├── 🧠 training_logs/          # Models and logs
├── 🚀 launch_simulation.py    # Main launcher
├── 🤖 train_rl.py            # PPO training
└── 📖 WEBOTS_INSTRUCTIONS.md  # Detailed setup guide
```

## 🎮 How to Use

### 1. Manual Data Collection
- Use **W/A/S/D** keys to drive
- Data automatically saved for training
- Press **M** to toggle AI/manual mode

### 2. AI Training
```bash
# Train reinforcement learning model
python3 train_rl.py

# View training progress
tensorboard --logdir training_logs
```

### 3. Autonomous Driving
- Load trained model automatically
- Switch to AI mode with **M** key
- Watch the car drive itself!

## 🔧 Features

### ✅ Reinforcement Learning
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-Baselines3
- **Environment**: Custom Gymnasium wrapper
- **Observations**: Camera + sensors
- **Actions**: Steering + throttle

### ✅ Imitation Learning
- Manual driving data collection
- Behavior cloning from human demonstrations
- Real-time training data export

### ✅ Webots Integration
- Professional physics simulation
- Realistic car dynamics
- Multiple world scenarios
- Sensor simulation (camera, distance, IMU)

### ✅ Development-Friendly
- **Test Mode**: Run without Webots installed
- **Modular Design**: Easy to extend
- **Error Handling**: Comprehensive dependency checking
- **Documentation**: Step-by-step guides

## 🔬 Testing Without Webots

Perfect for development and CI/CD:

```bash
# Test individual components
cd controllers/agent_driver
python3 agent_driver.py --test

cd ../supervisor_controller  
python3 supervisor_controller.py --test

# Test training pipeline
python3 train_rl.py --standalone
```

## 📖 Documentation

- **[WEBOTS_INSTRUCTIONS.md](WEBOTS_INSTRUCTIONS.md)**: Complete setup guide
- **[WEBOTS_SETUP_GUIDE.md](WEBOTS_SETUP_GUIDE.md)**: Installation details
- **Inline comments**: Comprehensive code documentation

## 🐛 Troubleshooting

### Common Issues Fixed ✅

❌ **Old Issue**: "Gym has been unmaintained since 2022"  
✅ **Fixed**: Project now uses `gymnasium`

❌ **Old Issue**: "Missing required packages: opencv-python"  
✅ **Fixed**: Proper import detection (`cv2` vs `opencv-python`)

❌ **Old Issue**: Dependency conflicts  
✅ **Fixed**: Compatible version specifications

### Getting Help

1. Check **WEBOTS_INSTRUCTIONS.md** for detailed setup
2. Verify dependencies: all packages should install without errors
3. Test controllers: `python3 launch_simulation.py` should work
4. For Webots issues: ensure version R2023a+ is installed

## 🏆 Success Criteria

Your setup works when:
- ✅ `python3 launch_simulation.py` runs without dependency errors
- ✅ Controllers pass test mode
- ✅ Webots loads the world file (when installed)
- ✅ Car responds to WASD controls
- ✅ Training pipeline completes successfully

## 🎯 Next Steps

1. **Install Webots** from https://cyberbotics.com/
2. **Run launcher**: `python3 launch_simulation.py`
3. **Drive manually** for 10-15 minutes to collect data
4. **Train AI model**: `python3 train_rl.py`
5. **Test autonomous mode**: Switch to AI with 'M' key

## 🤝 Contributing

This project is ready for:
- Adding new scenarios
- Implementing different RL algorithms
- Extending sensor capabilities
- Creating custom world files

---

**🎉 Happy Autonomous Driving!** 🚗💨

*No dependency errors. No gym warnings. Just working autonomous driving simulation.*