#!/usr/bin/env python3
"""
Launch script for Webots Autonomous Driving Simulation
Provides an easy way to start the simulation with different configurations.
"""

import os
import sys
import subprocess
import argparse
import time


def find_webots_executable():
    """Find the Webots executable on the system."""
    possible_paths = [
        "/usr/local/webots/webots",
        "/opt/webots/webots", 
        "/Applications/Webots.app/Contents/MacOS/webots",
        "C:\\Program Files\\Cyberbotics\\Webots R2023a\\msys64\\mingw64\\bin\\webots.exe",
        "webots"  # Assume it's in PATH
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try to find in PATH
    try:
        result = subprocess.run(["which", "webots"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None


def check_dependencies():
    """Check if required Python dependencies are installed."""
    required_packages = [
        "numpy", "opencv-python", "gym", "stable-baselines3", 
        "torch", "matplotlib", "tensorboard"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def setup_environment():
    """Setup the environment for the simulation."""
    # Create necessary directories
    directories = [
        "collected_data",
        "collected_data/images", 
        "collected_data/actions",
        "collected_data/sensor_data",
        "training_logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def launch_webots(world_file="worlds/autonomous_driving.wbt", mode="normal"):
    """Launch Webots with the specified world file."""
    webots_exe = find_webots_executable()
    
    if not webots_exe:
        print("Error: Webots executable not found!")
        print("Please install Webots or add it to your PATH")
        print("Download from: https://cyberbotics.com/")
        return False
    
    print(f"Found Webots at: {webots_exe}")
    print(f"Launching world: {world_file}")
    
    # Prepare command
    cmd = [webots_exe]
    
    if mode == "fast":
        cmd.append("--mode=fast")
    elif mode == "batch":
        cmd.extend(["--mode=batch", "--minimize"])
    
    cmd.append(world_file)
    
    try:
        print("Starting Webots...")
        print("Command:", " ".join(cmd))
        
        # Start Webots
        process = subprocess.Popen(cmd)
        
        print("\nWebots is starting up...")
        print("Once Webots loads:")
        print("1. The simulation should start automatically")
        print("2. Use W/A/S/D keys to control the car manually")
        print("3. Press 'M' to toggle between manual and AI mode")
        print("4. Press 'Q' to quit the agent controller")
        print("\nPress Ctrl+C to stop this script")
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        process.terminate()
    except Exception as e:
        print(f"Error launching Webots: {e}")
        return False
    
    return True


def start_training():
    """Start the RL training process."""
    print("Starting RL training...")
    
    try:
        subprocess.run([sys.executable, "train_rl.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Launch Webots Autonomous Driving Simulation")
    parser.add_argument("--mode", choices=["normal", "fast", "batch"], default="normal",
                       help="Webots execution mode (default: normal)")
    parser.add_argument("--world", default="worlds/autonomous_driving.wbt",
                       help="World file to load (default: worlds/autonomous_driving.wbt)")
    parser.add_argument("--train", action="store_true",
                       help="Start RL training instead of simulation")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check dependencies and exit")
    parser.add_argument("--setup", action="store_true",
                       help="Setup environment directories and exit")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("All dependencies are installed!")
        sys.exit(0)
    
    # Setup environment if requested
    if args.setup:
        setup_environment()
        print("Environment setup complete!")
        sys.exit(0)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("Please install missing dependencies before running the simulation.")
        sys.exit(1)
    
    # Setup environment
    print("Setting up environment...")
    setup_environment()
    
    # Start training or simulation
    if args.train:
        if not start_training():
            sys.exit(1)
    else:
        if not launch_webots(args.world, args.mode):
            sys.exit(1)


if __name__ == "__main__":
    main()