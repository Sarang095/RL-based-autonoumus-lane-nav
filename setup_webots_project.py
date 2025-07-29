#!/usr/bin/env python3
"""
Setup script for Webots Autonomous Driving Project
Handles dependency installation and Webots integration setup.
"""

import os
import sys
import subprocess
import platform


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    if description:
        print(f"Description: {description}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n=== Installing Python Dependencies ===")
    
    # Upgrade pip first
    print("Upgrading pip...")
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"]):
        print("Warning: Failed to upgrade pip, continuing anyway...")
    
    # Install dependencies
    print("Installing project dependencies...")
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]):
        print("Error: Failed to install dependencies from requirements.txt")
        return False
    
    print("Dependencies installed successfully! ✓")
    return True


def verify_installation():
    """Verify that all required packages are installed."""
    print("\n=== Verifying Installation ===")
    
    required_packages = [
        "numpy", "opencv-python", "gymnasium", "stable-baselines3", 
        "torch", "matplotlib", "tensorboard"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def setup_project_directories():
    """Create necessary project directories."""
    print("\n=== Setting up Project Directories ===")
    
    directories = [
        "collected_data",
        "collected_data/images", 
        "collected_data/actions",
        "collected_data/sensor_data",
        "training_logs",
        "models"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}")
    
    print("Project directories setup complete! ✓")


def find_webots():
    """Find Webots installation."""
    print("\n=== Checking Webots Installation ===")
    
    possible_paths = [
        "/usr/local/webots/webots",
        "/opt/webots/webots", 
        "/Applications/Webots.app/Contents/MacOS/webots",
        "C:\\Program Files\\Cyberbotics\\Webots R2023a\\msys64\\mingw64\\bin\\webots.exe",
        "C:\\Program Files\\Cyberbotics\\Webots R2023b\\msys64\\mingw64\\bin\\webots.exe",
        "C:\\Program Files\\Cyberbotics\\Webots R2024a\\msys64\\mingw64\\bin\\webots.exe",
        "/snap/webots/current/usr/bin/webots",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found Webots at: {path} ✓")
            return path
    
    # Try to find in PATH
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["where", "webots"], capture_output=True, text=True)
        else:
            result = subprocess.run(["which", "webots"], capture_output=True, text=True)
        
        if result.returncode == 0:
            path = result.stdout.strip()
            print(f"Found Webots in PATH: {path} ✓")
            return path
    except:
        pass
    
    print("✗ Webots not found!")
    print("\nPlease install Webots from: https://cyberbotics.com/")
    print("Supported versions: R2023a, R2023b, R2024a or later")
    return None


def create_run_script():
    """Create a simple run script for the project."""
    print("\n=== Creating Run Script ===")
    
    script_content = '''#!/usr/bin/env python3
"""
Quick start script for Webots Autonomous Driving Project
"""

import subprocess
import sys
import os

def main():
    print("Webots Autonomous Driving Project")
    print("=================================")
    print()
    print("Available options:")
    print("1. Run simulation (normal mode)")
    print("2. Run simulation (fast mode)")
    print("3. Start RL training")
    print("4. Check dependencies")
    print("5. Setup environment")
    print()
    
    try:
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            subprocess.run([sys.executable, "launch_simulation.py"])
        elif choice == "2":
            subprocess.run([sys.executable, "launch_simulation.py", "--mode", "fast"])
        elif choice == "3":
            subprocess.run([sys.executable, "launch_simulation.py", "--train"])
        elif choice == "4":
            subprocess.run([sys.executable, "launch_simulation.py", "--check-deps"])
        elif choice == "5":
            subprocess.run([sys.executable, "launch_simulation.py", "--setup"])
        else:
            print("Invalid option. Please select 1-5.")
            
    except KeyboardInterrupt:
        print("\\nExiting...")

if __name__ == "__main__":
    main()
'''
    
    with open("run.py", "w") as f:
        f.write(script_content)
    
    # Make executable on Unix-like systems
    if platform.system() != "Windows":
        os.chmod("run.py", 0o755)
    
    print("Created run.py script ✓")


def print_instructions():
    """Print final instructions for the user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print()
    print("To run the project:")
    print("1. Start Webots application manually")
    print("2. Open the autonomous_driving.wbt world file")
    print("3. Run: python launch_simulation.py")
    print()
    print("Alternative quick start:")
    print("- Run: python run.py")
    print()
    print("For training:")
    print("- Run: python launch_simulation.py --train")
    print()
    print("For help:")
    print("- Run: python launch_simulation.py --help")
    print()
    print("Make sure Webots is installed and accessible!")
    print("Download from: https://cyberbotics.com/")


def main():
    """Main setup function."""
    print("Webots Autonomous Driving Project Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("Some packages failed to install. Please check the errors above.")
        sys.exit(1)
    
    # Setup directories
    setup_project_directories()
    
    # Check Webots
    webots_path = find_webots()
    
    # Create run script
    create_run_script()
    
    # Print final instructions
    print_instructions()
    
    if not webots_path:
        print("\nWARNING: Webots not found. Please install it before running the simulation.")
        sys.exit(1)


if __name__ == "__main__":
    main()