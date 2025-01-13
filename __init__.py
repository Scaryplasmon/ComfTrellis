import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def is_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None

def setup_packages():
    """Setup required packages if not already installed"""
    # Get the current directory
    cur_dir = Path(__file__).parent.absolute()
    
    # Check if key packages are installed
    required_packages = ['diff-gaussian-rasterization', 'xformers', 'kaolin', 'spconv']
    missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]
    
    if not missing_packages:
        print("All required packages are already installed.")
        return True
        
    print(f"Installing missing packages: {missing_packages}")
    
    # Run setup.py
    setup_script = cur_dir / "setup.py"
    if not setup_script.exists():
        print("Error: setup.py not found")
        return False
        
    try:
        subprocess.check_call([sys.executable, str(setup_script)])
        print("Package installation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during package installation: {e}")
        return False

# Run setup before importing nodes
if setup_packages():
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
else:
    print("Failed to setup required packages. ComfTrellis nodes will not be available.")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}