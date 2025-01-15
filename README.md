# ComfTrellis (❁´◡`❁) 🍬🎊
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A quick start ComfyUI implementation of [TRELLIS](https://trellis3d.github.io/).

## 🚀 Features
- Single and multi-view image processing
- Optimized for 8GB VRAM
- Fast inference (~10s)
- Pre-compiled wheels for quick installation

    <td align="center">
      <img src="https://github.com/user-attachments/assets/22f713ab-f252-4751-856f-6e85834f867b" width="800"/>
    </td>

## 🔧 Available Nodes

| Node | Description |
|------|-------------|
| **Load Trellis Model** | Downloads and loads the TRELLIS model for inference |
| **Trellis Inference** | Converts images into 3D Gaussians with dynamic support for single and multiview input with integrated rendering options|
| **Save GLB File** | Exports 3D models to GLB format ("fast"/"opt" modes) |
| **Remove Background (Square)** | Uitlity - Preprocesses images with background removal, standardizes the inputs |

## Results
<details>
<summary>Click to see the videos</summary>

https://github.com/user-attachments/assets/f1bd019b-5f1a-4604-94d6-7ccecf61e0cd

https://github.com/user-attachments/assets/3f8b145c-abf3-45c5-bcc9-ba4c34acf40f
</details>

## Requirements

| Framework | ComfyUI |
|-----------|---------------|
| GPU | 8GB+ VRAM |
| CUDA | 12.4 |
| Python | 3.10 |
| OS | Windows 11 |

## Installation

### Quick Start
Simply clone this repository into your ComfyUI's `custom_nodes` folder and launch ComfyUI:

```bash
cd custom_nodes
git clone https://github.com/Scaryplasmon/ComfTrellis.git
cd ..
python main.py
#or comfy launch
```
## 🧌 Example Workflows
> Drag and drop these workflows directly into ComfyUI

### Single View Processing
<table>
  <tr>
    <th width="50%">Low Quality</th>
    <th width="50%">High Quality</th>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/5e7b752a-c08f-4fad-80b6-80d0c1c6a7ac" width="400"/></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/93090e1e-8365-46f9-9b22-b23fcb4ff20a" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><i>LowQuality_SingleView</i></td>
    <td align="center"><i>HighQuality_SingleView</i></td>
  </tr>
</table>

### Multi View Processing
<table>
  <tr>
    <th width="50%">Low Quality</th>
    <th width="50%">High Quality</th>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/b0b9a5a8-cbc2-4f37-85fe-e1a70afdd5f7" width="400"/></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/ea8e4064-57c4-4954-9457-08ca333ac366" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><i>LowQuality_MultiView</i></td>
    <td align="center"><i>HighQuality_MultiView</i></td>
  </tr>
</table>


## ⚡ Performance

The implementation is optimized for both efficiency and quality:

- 🔄 Automatic wheel installation
- 💨 25-second processing time at maximum settings thanks to early stopping.
- 💾 8GB VRAM optimization


## 📈 Ablation Studies
*Coming soon*

## 👏 Credits
<p align="center">
This is a ComfyUI implementation of <a href="https://github.com/microsoft/TRELLIS">TRELLIS</a> by Microsoft Research.<br>
Please refer to their repository for the original implementation and research paper.

---

