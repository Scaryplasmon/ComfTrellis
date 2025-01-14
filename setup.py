import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request
import argparse

def download_wheel(url):
    """Download wheel file from GitHub and return its path"""
    # Convert GitHub blob URL to raw URL
    raw_url = url.replace("blob/main", "raw/main")
    
    # Extract the original wheel filename from the URL
    wheel_filename = url.split('/')[-1]
    print(f"Downloading {wheel_filename} from {raw_url}...")
    
    try:
        # Create temporary directory if it doesn't exist
        temp_dir = Path(__file__).parent / "temp_wheels"
        temp_dir.mkdir(exist_ok=True)
        
        # Download the wheel file with original filename
        wheel_path = temp_dir / wheel_filename
        urllib.request.urlretrieve(raw_url, wheel_path)
        return wheel_path
    except Exception as e:
        print(f"Error downloading {wheel_filename}: {e}")
        return None

def setup_package():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--packages', nargs='*', help='List of packages to install')
    args = parser.parse_args()

    packages_to_install = set(args.packages) if args.packages else set()
    
    cur_dir = Path(__file__).parent.absolute()
    
    print("Setting up Trellis ComfyUI custom node...")
    
    # Install base requirements first
    requirements_path = cur_dir / "requirements.txt"
    if requirements_path.exists():
        print("Installing base requirements...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_path)
            ])
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
            return False

    # GitHub wheel URLs with their original filenames
    github_wheels = [
        "https://github.com/MrForExample/Comfy3D_Pre_Builds/blob/main/_Build_Wheels/_Wheels_win_py312_torch2.5.1_cu124/nvdiffrast-0.3.3-py3-none-any.whl",
        "https://github.com/MrForExample/Comfy3D_Pre_Builds/blob/main/_Build_Wheels/_Wheels_win_py310_torch2.5.1_cu124/vox2seq-0.0.0-cp310-cp310-win_amd64.whl"

    ]

    # Local wheel paths
    local_wheels = [
        str(Path(__file__).parent / "wheels" / "diff_gaussian_rasterization" / "diff_gaussian_rasterization-0.0.0-cp310-cp310-win_amd64.whl"),
        str(Path(__file__).parent / "wheels" / "diffoctreerast" / "diffoctreerast-0.0.0-cp310-cp310-win_amd64.whl"),
        str(Path(__file__).parent / "wheels" / "utils3d" / "utils3d-0.0.2-py3-none-any.whl")
    ]

    # Install wheels from GitHub
    print("Installing wheels from GitHub...")
    temp_dir = Path(__file__).parent / "temp_wheels"
    
    try:
        # Only install wheels for packages that need installation
        for wheel_url in github_wheels:
            pkg_name = wheel_url.split('/')[-1].split('-')[0]
            if not packages_to_install or pkg_name in packages_to_install:
                wheel_path = download_wheel(wheel_url)
                if wheel_path and wheel_path.exists():
                    print(f"Installing from GitHub wheel: {wheel_path.name}")
                    try:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", 
                            str(wheel_path),
                            "--no-deps"
                        ], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error installing {wheel_path.name}: {e}")
                        return False
                else:
                    print(f"Failed to download wheel from {wheel_url}")
                    return False

        # Install local wheels only if needed
        for wheel_path in local_wheels:
            wheel_path = Path(wheel_path)
            if not wheel_path.exists():
                print(f"Local wheel not found: {wheel_path}")
                return False
                
            # Extract package name and try both variants for verification
            pkg_name = wheel_path.name.split('-')[0]
            pkg_name_hyphen = pkg_name.replace('_', '-')
            pkg_name_underscore = pkg_name.replace('-', '_')
            
            print(f"Checking if {pkg_name} needs installation...")
            
            if not packages_to_install or pkg_name in packages_to_install:
                print(f"Installing from local wheel: {wheel_path.name}")
                try:
                    # Try to verify if package is already installed (check both naming variants)
                    verify_cmd = [sys.executable, "-m", "pip", "show"]
                    result_hyphen = subprocess.run([*verify_cmd, pkg_name_hyphen], 
                                                capture_output=True, text=True)
                    result_underscore = subprocess.run([*verify_cmd, pkg_name_underscore], 
                                                    capture_output=True, text=True)
                    
                    if result_hyphen.returncode != 0 and result_underscore.returncode != 0:
                        # Package not found with either name, install it
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", 
                            str(wheel_path),
                            "--force-reinstall",  # Force reinstall if verification failed
                            "--no-deps"
                        ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error installing {wheel_path.name}: {e}")
                    return False
    finally:
        # Cleanup: remove temporary wheel files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # Check if spconv is already installed
    try:
        import spconv
        print("spconv already installed, skipping installation")
    except ImportError:
        # Install spconv-cu124
        print("Installing spconv-cu124...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "spconv-cu124"
            ])
        except subprocess.CalledProcessError as e:
            print(f"Error installing spconv-cu124: {e}")
            return False

    print("Setup completed successfully!")
    return True

if __name__ == "__main__":
    success = setup_package()
    sys.exit(0 if success else 1)