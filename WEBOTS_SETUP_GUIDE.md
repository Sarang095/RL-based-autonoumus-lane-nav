# Webots Autonomous Driving Project - Setup Guide

This guide provides step-by-step instructions to set up and run the autonomous driving simulation in Webots.

## Prerequisites

### 1. Install Webots

Download and install Webots from the official website:
- **Website**: https://cyberbotics.com/
- **Supported versions**: R2023a, R2023b, R2024a or later
- **Platforms**: Windows, macOS, Linux

#### Installation Instructions:

**Windows:**
1. Download the installer from cyberbotics.com
2. Run the installer and follow the setup wizard
3. Default installation path: `C:\Program Files\Cyberbotics\Webots R2024a\`

**Linux:**
```bash
# Option 1: Using Snap (recommended)
sudo snap install webots

# Option 2: Download .deb package from website and install
wget https://github.com/cyberbotics/webots/releases/download/R2024a/webots_2024a_amd64.deb
sudo dpkg -i webots_2024a_amd64.deb
```

**macOS:**
1. Download the .dmg file from cyberbotics.com
2. Mount the disk image and drag Webots to Applications folder

### 2. Install Python Dependencies

Ensure you have Python 3.8 or higher installed, then run:

```bash
# Run the automated setup
python setup_webots_project.py
```

Or manually install dependencies:
```bash
pip install -r requirements.txt
```

## Project Setup

### 1. Automated Setup (Recommended)

Run the setup script to automatically configure everything:

```bash
python setup_webots_project.py
```

This script will:
- Check Python version compatibility
- Install all required dependencies
- Verify package installation
- Create necessary directories
- Check for Webots installation
- Create helper scripts

### 2. Manual Setup

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p collected_data/images
mkdir -p collected_data/actions
mkdir -p collected_data/sensor_data
mkdir -p training_logs
mkdir -p models

# Verify installation
python launch_simulation.py --check-deps
```

## Running the Simulation

### Method 1: Using Launch Script (Recommended)

The launch script provides the easiest way to run the simulation:

```bash
# Normal simulation mode
python launch_simulation.py

# Fast simulation mode (for training)
python launch_simulation.py --mode fast

# Start RL training
python launch_simulation.py --train

# Check dependencies
python launch_simulation.py --check-deps
```

### Method 2: Direct Webots Launch

1. **Start Webots manually**:
   - Open Webots application
   - File → Open World
   - Navigate to your project folder
   - Open `worlds/autonomous_driving.wbt`

2. **Run the simulation**:
   - Click the Play button in Webots
   - The simulation should start automatically

### Method 3: Using Quick Start Script

```bash
python run.py
```

This provides an interactive menu with all available options.

## Webots World Configuration

### Required World Structure

Your Webots world file (`worlds/autonomous_driving.wbt`) should contain:

1. **Robot Node (Agent Car)**:
```
DEF AGENT_CAR Robot {
  controller "agent_driver"
  children [
    # Camera device
    Camera {
      name "camera"
      width 640
      height 480
    }
    # Distance sensors
    DistanceSensor { name "front_sensor" }
    DistanceSensor { name "rear_sensor" }
    DistanceSensor { name "left_sensor" }
    DistanceSensor { name "right_sensor" }
    # Inertial unit
    InertialUnit { name "inertial_unit" }
    # Motors
    HingeJoint { device [ RotationalMotor { name "wheel_left_motor" } ] }
    HingeJoint { device [ RotationalMotor { name "wheel_right_motor" } ] }
    HingeJoint { device [ RotationalMotor { name "steering_motor" } ] }
  ]
}
```

2. **Supervisor Node**:
```
DEF SUPERVISOR Robot {
  supervisor TRUE
  controller "supervisor_controller"
}
```

3. **Optional Traffic Lights**:
```
DEF TRAFFIC_LIGHT_1 LED { name "traffic_light_1" }
DEF TRAFFIC_LIGHT_2 LED { name "traffic_light_2" }
DEF TRAFFIC_LIGHT_3 LED { name "traffic_light_3" }
```

### Controller Assignment

In Webots:
1. Right-click on the AGENT_CAR robot
2. Select "Edit controller"
3. Choose `agent_driver.py` (make sure the file is in the `controllers` directory)
4. Right-click on the SUPERVISOR robot
5. Select "Edit controller"
6. Choose `supervisor_controller.py`

## Usage Guide

### Manual Driving Mode

When the simulation starts, the car is in manual mode by default:

- **W**: Accelerate forward
- **S**: Brake/Reverse
- **A**: Steer left
- **D**: Steer right
- **M**: Toggle between manual and AI mode
- **Q**: Quit the controller

### AI Mode

Press 'M' to switch to AI mode. The car will use the trained model for autonomous driving.

### Data Collection

While in manual mode, the system automatically collects training data:
- Camera images are saved to `collected_data/images/`
- Actions are saved to `collected_data/actions/`
- Sensor data is saved to `collected_data/sensor_data/`

### Training

To train the RL agent:
```bash
python launch_simulation.py --train
```

Monitor training progress:
```bash
tensorboard --logdir training_logs
```

## Troubleshooting

### Common Issues

1. **"Webots executable not found"**
   - Install Webots from https://cyberbotics.com/
   - Add Webots to your system PATH
   - Check installation path in `launch_simulation.py`

2. **"Missing required packages"**
   - Run: `python setup_webots_project.py`
   - Or manually: `pip install -r requirements.txt`

3. **"AGENT_CAR node not found"**
   - Ensure your robot is defined with `DEF AGENT_CAR` in the world file
   - Check that the controller is properly assigned

4. **Device not found warnings**
   - Verify sensor/motor names match those in your robot model
   - Check the world configuration section above

5. **Gymnasium/Gym compatibility issues**
   - The project now uses Gymnasium (maintained replacement for Gym)
   - All imports have been updated automatically

### Performance Tips

- Use `--mode fast` for faster simulation during training
- Reduce camera resolution for better performance
- Monitor system resources during training
- Use GPU acceleration if available

## Project Structure

```
webots-autonomous-driving/
├── worlds/                     # Webots world files
│   └── autonomous_driving.wbt
├── controllers/                # Webots controllers
│   ├── agent_driver.py
│   └── supervisor_controller.py
├── collected_data/             # Training data
│   ├── images/
│   ├── actions/
│   └── sensor_data/
├── training_logs/              # RL training logs
├── models/                     # Saved models
├── launch_simulation.py        # Main launch script
├── train_rl.py                # RL training script
├── setup_webots_project.py    # Setup script
├── run.py                     # Quick start script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Next Steps

1. **Test the setup**: Run `python launch_simulation.py --check-deps`
2. **Start simulation**: Run `python launch_simulation.py`
3. **Collect data**: Drive manually to collect training samples
4. **Train agent**: Run `python launch_simulation.py --train`
5. **Test AI**: Switch to AI mode in the simulation

For more detailed information, see the main `README.md` file.

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Verify all dependencies are installed
3. Ensure Webots is properly configured
4. Check the console output for error messages

The project is designed to work with Webots R2023a and later versions. Make sure you're using a compatible version.