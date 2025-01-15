import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request
import argparse

def download_wheel(url):
    """Download wheel file from GitHub and return its path"""
    raw_url = url.replace("blob/main", "raw/main")
    
    wheel_filename = url.split('/')[-1]
    print(f"Downloading {wheel_filename} from {raw_url}...")
    
    try:
        temp_dir = Path(__file__).parent / "temp_wheels"
        temp_dir.mkdir(exist_ok=True)
        
        wheel_path = temp_dir / wheel_filename
        urllib.request.urlretrieve(raw_url, wheel_path)
        return wheel_path
    except Exception as e:
        print(f"Error downloading {wheel_filename}: {e}")
        return None

def setup_package():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packages', nargs='*', help='List of packages to install')
    args = parser.parse_args()

    packages_to_install = set(args.packages) if args.packages else set()
    
    cur_dir = Path(__file__).parent.absolute()
    
    print("Setting up Trellis ComfyUI custom node...")
    
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

    github_wheels = [
        "https://github.com/MrForExample/Comfy3D_Pre_Builds/blob/main/_Build_Wheels/_Wheels_win_py312_torch2.5.1_cu124/nvdiffrast-0.3.3-py3-none-any.whl",
        "https://github.com/MrForExample/Comfy3D_Pre_Builds/blob/main/_Build_Wheels/_Wheels_win_py310_torch2.5.1_cu124/vox2seq-0.0.0-cp310-cp310-win_amd64.whl"

    ]

    local_wheels = [
        str(Path(__file__).parent / "wheels" / "diff_gaussian_rasterization" / "diff_gaussian_rasterization-0.0.0-cp310-cp310-win_amd64.whl"),
        str(Path(__file__).parent / "wheels" / "diffoctreerast" / "diffoctreerast-0.0.0-cp310-cp310-win_amd64.whl"),
        str(Path(__file__).parent / "wheels" / "utils3d" / "utils3d-0.0.2-py3-none-any.whl")
    ]

    print("Installing wheels from GitHub...")
    temp_dir = Path(__file__).parent / "temp_wheels"
    
    try:
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

        for wheel_path in local_wheels:
            wheel_path = Path(wheel_path)
            if not wheel_path.exists():
                print(f"Local wheel not found: {wheel_path}")
                return False
                
            pkg_name = wheel_path.name.split('-')[0]
            pkg_name_hyphen = pkg_name.replace('_', '-')
            pkg_name_underscore = pkg_name.replace('-', '_')
            
            print(f"Checking if {pkg_name} needs installation...")
            
            if not packages_to_install or pkg_name in packages_to_install:
                print(f"Installing from local wheel: {wheel_path.name}")
                try:

                    verify_cmd = [sys.executable, "-m", "pip", "show"]
                    result_hyphen = subprocess.run([*verify_cmd, pkg_name_hyphen], 
                                                capture_output=True, text=True)
                    result_underscore = subprocess.run([*verify_cmd, pkg_name_underscore], 
                                                    capture_output=True, text=True)
                    
                    if result_hyphen.returncode != 0 and result_underscore.returncode != 0:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", 
                            str(wheel_path),
                            "--force-reinstall",
                            "--no-deps"
                        ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error installing {wheel_path.name}: {e}")
                    return False
    finally:

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

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