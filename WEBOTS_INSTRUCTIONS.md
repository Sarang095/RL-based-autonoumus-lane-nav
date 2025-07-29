# Webots Autonomous Driving Project - Complete Setup Guide

## 🚗 Project Overview
This project provides a complete autonomous driving simulation environment for Webots, featuring:
- Reinforcement Learning with Stable-Baselines3 PPO
- Imitation Learning from human demonstrations  
- Custom Webots world with highway and city scenarios
- Real-time sensor data collection and processing
- Modular controller architecture

## 📋 Prerequisites

### 1. Install Webots
**Download Webots R2023a or later from:** https://cyberbotics.com/

#### Windows Installation:
```bash
# Download and run the installer
# Default path: C:\Program Files\Cyberbotics\Webots R2024a\
# Add to PATH: C:\Program Files\Cyberbotics\Webots R2024a\msys64\mingw64\bin\
```

#### Linux Installation:
```bash
# Option 1: Snap (Recommended)
sudo snap install webots

# Option 2: Manual installation
wget https://github.com/cyberbotics/webots/releases/download/R2024a/webots_2024a_amd64.deb
sudo dpkg -i webots_2024a_amd64.deb

# Fix dependencies if needed
sudo apt-get install -f
```

#### macOS Installation:
```bash
# Download from website and drag to Applications folder
# Path: /Applications/Webots.app/Contents/MacOS/webots
```

### 2. Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install numpy opencv-python gymnasium stable-baselines3 torch matplotlib tensorboard
```

## 🚀 Quick Start

### Method 1: Automatic Launch (Recommended)
```bash
# Run the launcher script
python3 launch_simulation.py
```

This script will:
- ✅ Check all dependencies
- ✅ Set up data collection directories
- ✅ Find and launch Webots automatically
- ✅ Load the autonomous driving world

### Method 2: Manual Launch
1. **Start Webots**
2. **Open World File**: File → Open World → `worlds/highway_drive.wbt`
3. **Run Simulation**: Press the ▶️ play button

## 🎮 Running the Simulation

### Controller Modes

#### 1. Manual Data Collection Mode (Default)
- Use **W/A/S/D** keys to drive the car
- Data is automatically saved to `collected_data/`
- Press **M** to switch to AI mode
- Press **Q** to quit

#### 2. AI Autonomous Mode
- Press **M** to switch from manual to AI mode
- Car drives automatically using trained model
- Monitor performance in real-time

### Training Options

#### Train RL Model:
```bash
python3 train_rl.py
```

#### Evaluate Model:
```bash
python3 evaluate_model.py
```

## 📁 Project Structure
```
webots_autonomous_driving/
├── worlds/
│   ├── highway_drive.wbt      # Main Webots world file
│   └── autonomous_driving.wbt # Alternative world
├── controllers/
│   ├── agent_driver/          # Main car controller
│   │   └── agent_driver.py
│   └── supervisor_controller/ # Environment manager
│       └── supervisor_controller.py
├── collected_data/            # Training data storage
│   ├── images/
│   ├── actions/
│   └── sensor_data/
├── training_logs/             # Model and logs
├── launch_simulation.py       # Main launcher
├── train_rl.py               # RL training script
└── requirements.txt          # Dependencies
```

## 🔧 Configuration

### Webots Settings
1. **File → Preferences → General**
2. **Set WEBOTS_PROJECT path** to this project directory
3. **Tools → Import PROTO** (if using custom models)

### Controller Path Setup
1. **Tools → Preferences → Projects**
2. **Add controller path**: `[PROJECT_DIR]/controllers`

## 📊 Data Collection

### Automatic Data Collection
- Drive manually with W/A/S/D keys
- Data saved automatically every frame:
  - **Images**: Camera feed (RGB + processed)
  - **Actions**: Steering and throttle values
  - **Sensors**: Distance sensors, IMU, speed

### Data Format
```python
# Saved in collected_data/
{
    "timestamp": "2024-01-01_12:00:00",
    "image": "camera_rgb.png",
    "action": [throttle, steering],
    "sensors": {
        "speed": 15.2,
        "front_distance": 10.5,
        "acceleration": [0.1, 0.0, -9.8]
    }
}
```

## 🧠 Training Models

### 1. Reinforcement Learning (PPO)
```bash
# Train new model
python3 train_rl.py

# Monitor training
tensorboard --logdir training_logs
```

### 2. Imitation Learning
```bash
# First collect manual driving data
python3 launch_simulation.py
# Drive manually for 10-15 minutes

# Then train imitation model
python3 train_il.py --data_dir collected_data
```

## 🔬 Testing Without Webots

If you don't have Webots installed yet, you can test the controllers:

```bash
# Test agent controller
cd controllers/agent_driver
python3 agent_driver.py --test

# Test supervisor controller  
cd controllers/supervisor_controller
python3 supervisor_controller.py --test

# Test training pipeline
python3 train_rl.py --standalone
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Webots executable not found"
**Solution:**
- Check Webots installation path
- Add Webots to system PATH
- Use absolute path in launch script

#### 2. "Missing required packages"
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. "Controller not found in Webots"
**Solution:**
- Check controller path in Webots preferences
- Ensure controller names match in world file
- Verify Python path in Webots

#### 4. "Gym deprecated" warning
**Solution:**
- Already fixed! Project uses `gymnasium` instead of `gym`

### Advanced Configuration

#### Custom World Files
1. Create new `.wbt` file in `worlds/`
2. Add DEF="AGENT_CAR" to vehicle node
3. Set controller field to "agent_driver"
4. Add supervisor with "supervisor_controller"

#### Model Hyperparameters
Edit `train_rl.py` to adjust:
- Learning rate: `learning_rate=3e-4`
- Batch size: `batch_size=64`  
- Training steps: `total_timesteps=100000`

## 📈 Performance Monitoring

### Real-time Metrics
- **Speed**: Current vehicle velocity
- **Steering Angle**: Wheel angle (-1 to 1)
- **Distance Sensors**: Obstacle detection
- **Reward**: RL training progress

### Logs and Visualization
```bash
# View training progress
tensorboard --logdir training_logs

# Plot collected data
python3 plot_data.py --data_dir collected_data
```

## 🎯 Next Steps

1. **Collect Training Data**: Drive manually for 15-20 minutes
2. **Train Models**: Run RL and IL training
3. **Test Performance**: Evaluate on different scenarios
4. **Iterate**: Adjust hyperparameters and retrain

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure Webots version compatibility (R2023a+)
4. Check console output for detailed error messages

## 🏆 Success Criteria

Your setup is working correctly when:
- ✅ Webots launches with the world file
- ✅ Car responds to W/A/S/D controls
- ✅ Data is saved in `collected_data/`
- ✅ Training starts without errors
- ✅ AI mode shows autonomous driving

Happy autonomous driving! 🚗💨