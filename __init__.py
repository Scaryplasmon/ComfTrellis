import os
import sys
import subprocess
from pathlib import Path

def test_import(package_name):
    """Actually try to import the package and print detailed error if it fails"""
    try:
        print(f"Testing import of {package_name}...")
        __import__(package_name)
        print(f"Successfully imported {package_name}")
        return True
    except ImportError as e:
        print(f"ImportError for {package_name}: {str(e)}")
        return False
    except Exception as e:
        print(f"Error importing {package_name}: {str(e)}")
        return False

def setup_packages():
    """Setup required packages if not already installed"""
    cur_dir = Path(__file__).parent.absolute()
    
    # Package mapping (wheel name -> actual import name)
    package_mapping = {
        'xformers': 'xformers',
        'kaolin': 'kaolin',
        'spconv': 'spconv',
        'nvdiffrast': 'nvdiffrast',
        'utils3d': 'utils3d',
        'vox2seq': 'vox2seq',
        'diffoctreerast': 'diffoctreerast'
    }
    
    # Special packages that need installation but skip verification
    special_packages = ['diff_gaussian_rasterization']
    
    print("\nTesting all required packages...")
    missing_packages = []
    installed_packages = []
    
    # Check regular packages
    for wheel_name, import_name in package_mapping.items():
        if not test_import(import_name):
            print(f"Package {wheel_name} needs installation")
            missing_packages.append(wheel_name)
        else:
            installed_packages.append(wheel_name)
    
    # Add special packages if they're not installed
    for pkg_name in special_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "show", pkg_name], 
                         check=True, capture_output=True)
            installed_packages.append(pkg_name)
        except subprocess.CalledProcessError:
            print(f"Package {pkg_name} needs installation")
            missing_packages.append(pkg_name)
    
    if not missing_packages:
        print("\nAll required packages are properly installed.")
        return True
        
    print(f"\nPackages to install: {missing_packages}")
    print(f"Already working packages: {installed_packages}")
    
    # Run setup.py with list of missing packages
    setup_script = cur_dir / "setup.py"
    if not setup_script.exists():
        print("Error: setup.py not found")
        return False
        
    try:
        print("\nRunning setup.py for missing packages...")
        subprocess.check_call([
            sys.executable, 
            str(setup_script),
            "--packages", *missing_packages
        ])
        
        # Verify installations ONLY for regular packages
        print("\nVerifying installations...")
        still_missing = []
        for pkg_name in missing_packages:
            if pkg_name not in special_packages:  # Skip verification for special packages
                import_name = package_mapping[pkg_name]
                if not test_import(import_name):
                    still_missing.append(pkg_name)
        
        if still_missing:
            print(f"\nFailed to install these packages: {still_missing}")
            return False
            
        print("\nAll packages successfully installed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during package installation: {e}")
        return False

# Run setup before importing nodes
print("\nInitializing ComfTrellis...")
if setup_packages():
    print("\nLoading ComfTrellis nodes...")
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("No CUDA available")
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
else:
    print("\nFailed to setup required packages. ComfTrellis nodes will not be available.")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}