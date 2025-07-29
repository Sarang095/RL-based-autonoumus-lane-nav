#!/usr/bin/env python3
"""
Launch script for Webots Autonomous Driving Simulation
Handles dependency checking, environment setup, and Webots launching.
"""

import os
import sys
import subprocess
import platform


def find_webots_executable():
    """Find the Webots executable on the system."""
    possible_paths = [
        "/usr/local/webots/webots",
        "/opt/webots/webots", 
        "/Applications/Webots.app/Contents/MacOS/webots",
        "C:\\Program Files\\Cyberbotics\\Webots R2023a\\msys64\\mingw64\\bin\\webots.exe",
        "C:\\Program Files\\Cyberbotics\\Webots R2023b\\msys64\\mingw64\\bin\\webots.exe",
        "C:\\Program Files\\Cyberbotics\\Webots R2024a\\msys64\\mingw64\\bin\\webots.exe",
        "/snap/webots/current/usr/bin/webots",  # Snap install
        "webots"  # Assume it's in PATH
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try to find in PATH
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(["where", "webots"], capture_output=True, text=True)
        else:  # Unix-like
            result = subprocess.run(["which", "webots"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None


def check_dependencies():
    """Check if required Python dependencies are installed."""
    # Package name -> import name mapping
    package_imports = {
        "numpy": "numpy",
        "opencv-python": "cv2",
        "gymnasium": "gymnasium", 
        "stable-baselines3": "stable_baselines3",
        "torch": "torch",
        "matplotlib": "matplotlib",
        "tensorboard": "tensorboard"
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
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


def print_webots_instructions():
    """Print instructions for setting up and running with Webots."""
    print("\n" + "="*60)
    print("WEBOTS SETUP INSTRUCTIONS")
    print("="*60)
    print("\n1. INSTALL WEBOTS:")
    
    system = platform.system()
    if system == "Windows":
        print("   - Download from: https://cyberbotics.com/")
        print("   - Install to default location: C:\\Program Files\\Cyberbotics\\Webots R2024a\\")
        print("   - Add to PATH or use full path to webots.exe")
    elif system == "Darwin":  # macOS
        print("   - Download from: https://cyberbotics.com/")
        print("   - Install to /Applications/Webots.app/")
    else:  # Linux
        print("   Option A - Using Snap (recommended):")
        print("   sudo snap install webots")
        print("\n   Option B - Download .deb package:")
        print("   - Download from: https://cyberbotics.com/")
        print("   - sudo dpkg -i webots_R2024a_amd64.deb")
    
    print("\n2. OPEN WEBOTS PROJECT:")
    print("   - Start Webots application")
    print("   - Open File > Open World...")
    print("   - Navigate to this project folder")
    print("   - Open: worlds/highway_drive.wbt")
    
    print("\n3. SET CONTROLLER PATHS:")
    print("   - In Webots, ensure the controller path includes:")
    print(f"   - {os.path.abspath('controllers')}")
    print("   - Tools > Preferences > General > Projects > WEBOTS_PROJECT")
    
    print("\n4. RUN SIMULATION:")
    print("   - In Webots: Simulation > Run")
    print("   - The car should start driving automatically")
    print("   - Monitor training progress in training_logs/ folder")
    
    print("\n5. TRAINING AND EVALUATION:")
    print("   - Run: python3 train_rl.py")
    print("   - View results: python3 evaluate_model.py")
    
    print("\n" + "="*60)


def launch_simulation_mode():
    """Launch in standalone mode for testing without Webots."""
    print("\n" + "="*50)
    print("STANDALONE MODE - Testing Controllers")
    print("="*50)
    
    print("\nThis mode allows you to test the controllers without Webots.")
    print("Select an option:")
    print("1. Test agent controller logic")
    print("2. Test supervisor controller logic")
    print("3. Train RL model with simulated data")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\nTesting agent controller...")
                os.chdir("controllers/agent_driver")
                subprocess.run([sys.executable, "agent_driver.py", "--test"], check=True)
                break
            elif choice == "2":
                print("\nTesting supervisor controller...")
                os.chdir("controllers/supervisor_controller") 
                subprocess.run([sys.executable, "supervisor_controller.py", "--test"], check=True)
                break
            elif choice == "3":
                print("\nStarting RL training with simulated environment...")
                subprocess.run([sys.executable, "train_rl.py", "--standalone"], check=True)
                break
            elif choice == "4":
                print("Exiting...")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            break


def main():
    """Main launch function."""
    print("Webots Autonomous Driving Simulation Launcher")
    print("============================================")
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies before running the simulation.")
        return
    
    print("✓ All dependencies are installed!")
    
    # Setup environment
    print("\nSetting up environment...")
    setup_environment()
    print("✓ Environment setup complete!")
    
    # Check for Webots
    webots_path = find_webots_executable()
    
    if webots_path:
        print(f"✓ Found Webots at: {webots_path}")
        
        # Launch Webots with the world file
        world_file = os.path.join(os.getcwd(), "worlds", "highway_drive.wbt")
        if os.path.exists(world_file):
            print(f"\nLaunching Webots simulation...")
            print(f"World file: {world_file}")
            try:
                subprocess.run([webots_path, world_file], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error launching Webots: {e}")
        else:
            print(f"⚠ World file not found: {world_file}")
            print("Please ensure the world file exists.")
            
    else:
        print("⚠ Webots executable not found!")
        print_webots_instructions()
        
        # Ask user if they want to run in standalone mode
        print("\nWould you like to run in standalone mode to test the controllers?")
        response = input("Enter 'y' for yes, or any other key to exit: ").lower()
        
        if response in ['y', 'yes']:
            launch_simulation_mode()
        else:
            print("\nInstall Webots and run this script again.")
            print("Visit: https://cyberbotics.com/")


if __name__ == "__main__":
    main()