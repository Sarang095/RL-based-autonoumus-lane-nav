#!/usr/bin/env python3
"""
Setup script for Webots Autonomous Driving Project
Installs dependencies, creates directories, and configures the environment.
"""

import os
import sys
import subprocess
import platform


def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"\n{description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
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
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("Error: Python 3.7 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n" + "="*50)
    print("INSTALLING PYTHON DEPENDENCIES")
    print("="*50)
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "Upgrading pip..."):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing project dependencies..."):
        return False
    
    # Verify critical packages
    critical_packages = ["numpy", "cv2", "gym", "stable_baselines3", "torch"]
    
    for package in critical_packages:
        try:
            if package == "cv2":
                import cv2
            else:
                __import__(package)
            print(f"✓ {package} installed successfully")
        except ImportError:
            print(f"✗ {package} failed to install")
            return False
    
    return True


def create_directories():
    """Create necessary project directories."""
    print("\n" + "="*50)
    print("CREATING PROJECT DIRECTORIES")
    print("="*50)
    
    directories = [
        "collected_data",
        "collected_data/images",
        "collected_data/actions", 
        "collected_data/sensor_data",
        "training_logs",
        "models",
        "results",
        "tests"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created: {directory}")
        else:
            print(f"✓ Exists: {directory}")
    
    return True


def check_webots_installation():
    """Check if Webots is installed and accessible."""
    print("\n" + "="*50)
    print("CHECKING WEBOTS INSTALLATION")
    print("="*50)
    
    # Common Webots installation paths
    webots_paths = [
        "/usr/local/webots/webots",
        "/opt/webots/webots",
        "/Applications/Webots.app/Contents/MacOS/webots",
        "C:\\Program Files\\Cyberbotics\\Webots R2023a\\msys64\\mingw64\\bin\\webots.exe"
    ]
    
    webots_found = False
    
    for path in webots_paths:
        if os.path.exists(path):
            print(f"✓ Found Webots at: {path}")
            webots_found = True
            break
    
    # Check if webots is in PATH
    try:
        result = subprocess.run(["webots", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Webots is accessible via PATH")
            print(f"Version info: {result.stdout.strip()}")
            webots_found = True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    if not webots_found:
        print("⚠ Webots not found in common locations")
        print("Please install Webots from: https://cyberbotics.com/")
        print("Supported versions: R2023a or later")
        return False
    
    return True


def create_test_script():
    """Create a simple test script to verify the setup."""
    print("\n" + "="*50)
    print("CREATING TEST SCRIPT")
    print("="*50)
    
    test_script_content = '''#!/usr/bin/env python3
"""
Test script to verify the Webots autonomous driving project setup.
"""

import sys
import os
import traceback


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    modules = [
        ("numpy", "np"),
        ("cv2", "cv2"),
        ("gym", "gym"),
        ("stable_baselines3", "sb3"),
        ("torch", "torch"),
        ("matplotlib.pyplot", "plt"),
        ("json", "json"),
        ("time", "time")
    ]
    
    for module_name, alias in modules:
        try:
            exec(f"import {module_name} as {alias}")
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            return False
    
    return True


def test_directories():
    """Test that all required directories exist."""
    print("\\nTesting directories...")
    
    required_dirs = [
        "worlds",
        "controllers", 
        "controllers/agent_driver",
        "controllers/supervisor_controller",
        "collected_data",
        "training_logs"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} (missing)")
            return False
    
    return True


def test_files():
    """Test that all required files exist."""
    print("\\nTesting files...")
    
    required_files = [
        "worlds/autonomous_driving.wbt",
        "controllers/agent_driver/agent_driver.py",
        "controllers/supervisor_controller/supervisor_controller.py",
        "agent_driver.py",
        "supervisor_controller.py",
        "train_rl.py",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            return False
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("WEBOTS AUTONOMOUS DRIVING PROJECT - SETUP TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Test", test_directories), 
        ("File Test", test_files)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\\n{test_name}:")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            traceback.print_exc()
            all_passed = False
    
    print("\\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The project is ready to use.")
        print("\\nNext steps:")
        print("1. Run: python launch_simulation.py")
        print("2. Or: python train_rl.py (for training)")
    else:
        print("❌ SOME TESTS FAILED! Please fix the issues above.")
    print("="*60)


if __name__ == "__main__":
    main()
'''
    
    with open("test_setup.py", "w") as f:
        f.write(test_script_content)
    
    # Make it executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("test_setup.py", 0o755)
    
    print("✓ Created test_setup.py")
    return True


def make_scripts_executable():
    """Make scripts executable on Unix systems."""
    if platform.system() == "Windows":
        return True
    
    print("\n" + "="*50)
    print("MAKING SCRIPTS EXECUTABLE")
    print("="*50)
    
    scripts = ["launch_simulation.py", "setup_project.py", "test_setup.py"]
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f"✓ Made {script} executable")
    
    return True


def main():
    """Main setup function."""
    print("="*60)
    print("WEBOTS AUTONOMOUS DRIVING PROJECT SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    setup_steps = [
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Checking Webots installation", check_webots_installation),
        ("Creating test script", create_test_script),
        ("Making scripts executable", make_scripts_executable)
    ]
    
    for step_name, step_func in setup_steps:
        print(f"\n🔧 {step_name}...")
        
        try:
            if not step_func():
                print(f"❌ {step_name} failed!")
                sys.exit(1)
        except Exception as e:
            print(f"❌ {step_name} error: {e}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the setup: python test_setup.py")
    print("2. Launch simulation: python launch_simulation.py") 
    print("3. Start training: python train_rl.py")
    print("4. Read the README.md for detailed instructions")
    print("\nFor help: python launch_simulation.py --help")
    print("="*60)


if __name__ == "__main__":
    main()