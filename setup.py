import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_package():
    cur_dir = Path(__file__).parent.absolute()
    
    wheels_dir = cur_dir / "wheels"
    
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
    
    wheels = {
        "xformers": "xformers/xformers-0.0.28.post3-py3-none-any.whl",
        "nvdiffrast": "nvdiffrast/nvdiffrast-0.3.3-py3-none-any.whl",
        "diffoctreerast": "diffoctreerast/diffoctreerast-0.0.0-cp310-cp310-win_amd64.whl",
        "diff_gaussian_rasterization": "diff_gaussian_rasterization/diff_gaussian_rasterization-0.0.0-cp310-cp310-win_amd64.whl",
        "utils3d": "utils3d/utils3d-0.0.2-py3-none-any.whl",
        "vox2seq": "vox2seq/vox2seq-0.0.0-cp310-cp310-win_amd64.whl"
    }

    print("Installing custom wheels...")
    for pkg_name, wheel_file in wheels.items():
        wheel_path = wheels_dir / wheel_file
        if wheel_path.exists():
            print(f"Installing {pkg_name} from wheel...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    str(wheel_path)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error installing {pkg_name}: {e}")
                return False
        else:
            print(f"Warning: Wheel file not found for {pkg_name}: {wheel_path}")
            return False

    print("Installing spconv-cu121...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "spconv-cu121"
        ])
    except subprocess.CalledProcessError as e:
        print(f"Error installing spconv-cu121: {e}")
        return False

    print("Setup completed successfully!")
    return True

if __name__ == "__main__":
    success = setup_package()
    sys.exit(0 if success else 1)