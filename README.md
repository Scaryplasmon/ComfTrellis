# ComfTrellis

A ComfyUI implementation of [TRELLIS](https://trellis3d.github.io/) - a powerful 3D asset generation model that converts images into high-quality 3D assets.

## Installation

### Quick Install
Simply clone this repository into your ComfyUI's `custom_nodes` folder and launch ComfyUI:

```bash
cd custom_nodes
git clone https://github.com/yourusername/ComfTrellis.git
```

### Manual Install
If the quick install doesn't work, you can run the setup script:

```bash
cd ComfTrellis
python setup.py install
```

## Available Nodes

This implementation provides several nodes for working with TRELLIS:

- **Load Trellis Model**: Loads the TRELLIS model and prepares it for inference
- **Trellis Inference**: Converts images into 3D assets with various parameters for control
- **Save GLB File**: Exports the generated 3D model to GLB format with customizable settings
- **Remove Background (Square)**: Preprocesses images by removing backgrounds and making them square
- **Multi Image Batch**: Combines multiple images for multi-view 3D generation

## Example Workflow

1. Load your image(s)
2. (Optional) Use RembgSquare to remove background
3. Load the TRELLIS model
4. Run inference
5. Export to GLB format

Here's a basic workflow example:
![Workflow Example](assets/workflow.png)


## Requirements

- ComfyUI
- CUDA124-capable GPU with at least 8GB VRAM
- Python 3.10

## Model Details

This implementation is based on Microsoft's TRELLIS project, which offers:

- High-quality 3D asset generation from images
- Multiple output formats (Radiance Fields, 3D Gaussians, meshes)
- Flexible editing capabilities
- Support for both single and multi-image inputs

For more details about the underlying model, visit the [TRELLIS project page](https://trellis3d.github.io/).


## Credits

This is a ComfyUI implementation of [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft Research. Please refer to their repository for the original implementation and research paper.