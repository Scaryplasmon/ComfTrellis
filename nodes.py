import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Force xformers
os.environ['SPCONV_ALGO'] = 'native'      # Force native

import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from .trellis.pipelines import TrellisImageTo3DPipeline
from .trellis.utils import render_utils, postprocessing_utils

# Setup model paths
dir_trellis = os.path.join(folder_paths.models_dir, "trellis")
os.makedirs(dir_trellis, exist_ok=True)

# Default model info
DEFAULT_MODEL_REPO = "JeffreyXiang/TRELLIS-image-large"
DEFAULT_MODEL_PATH = os.path.join(dir_trellis, "trellis-image-large")

from huggingface_hub import snapshot_download

class LoadTrellisModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": "cuda"})
            }
        }
    
    RETURN_TYPES = ("TRELLIS_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Trellis"

    def load_model(self, device):
        # dtype_map = {
        #     "float32": torch.float32,
        #     "float16": torch.float16,
        #     "bfloat16": torch.bfloat16
        # }
        
        pbar = ProgressBar(2)
        
        try:
            pipeline = TrellisImageTo3DPipeline.from_pretrained(DEFAULT_MODEL_PATH)
        except Exception as e:
            print(f"Model not found locally, downloading from HuggingFace ({DEFAULT_MODEL_REPO})...")
            try:
                local_dir = snapshot_download(
                    repo_id=DEFAULT_MODEL_REPO,
                    local_dir=DEFAULT_MODEL_PATH,
                    local_dir_use_symlinks=False 
                )
                pipeline = TrellisImageTo3DPipeline.from_pretrained(local_dir)
            except Exception as download_error:
                print(f"Error downloading model: {str(download_error)}")
                raise
            
        pbar.update(1)
        
        pipeline.to(device=device)
        pbar.update(1)
        
        return (pipeline,)

class SaveGLBFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "gaussian": ("GAUSSIAN",),
                "filename": ("STRING", {"default": "output.glb"}),
                "simplify": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "texture_size": ("INT", {"default": 512, "min": 512, "max": 4096, "step": 512}),
                "bake_mode": (["fast", "opt"], {"default": "opt"}),
                "precision": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.99, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_glb"
    OUTPUT_NODE = True
    CATEGORY = "Trellis"

    def save_glb(self, mesh, gaussian, filename, simplify, texture_size, bake_mode, precision):
        
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Add .glb extension if not present
        if not filename.lower().endswith('.glb'):
            filename += '.glb'
            

        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "output", filename)
        
        try:
            # Create progress bar
            pbar = ProgressBar(4)
            
            # Step 1: Mesh processing
            pbar.update(1)
            
            # Step 2: UV unwrapping
            pbar.update(1)
            lambda_tv = 1 - precision
            # Step 3: Texture baking
            glb = postprocessing_utils.to_glb(
                gaussian,
                mesh,
                simplify=simplify,
                texture_size=texture_size,
                bake_mode=bake_mode,
                lambda_tv=lambda_tv, 
            )
            pbar.update(1)
            
            # Step 4: GLB export
            glb.export(output_path)
            pbar.update(1)
            
            print(f"GLB file saved to: {output_path}")
            
        except Exception as e:
            print(f"Error during GLB conversion: {str(e)}")
            raise
            
        return ()
    
class RembgSquare:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "white_bg": ("BOOLEAN", {"default": True, "label": "White Background"}),
                "edge_quality": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 20, 
                    "step": 1,
                    "display": "slider",
                    "label": "Edge Quality"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "Trellis"

    def check_transparency(self, img):
        """Check if image has any fully transparent pixels"""
        if img.mode == 'RGBA':
            alpha = np.array(img.split()[3])
            return (alpha == 0).any()  # Return True if any pixel has 0 alpha
        return False

    def process_image(self, image, white_bg, edge_quality):
        i = 255. * image[0].cpu().numpy()
        input_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        w, h = input_image.size
        size = max(w, h)
        

        if input_image.mode == 'RGBA':
            if self.check_transparency(input_image):
                print("Image is mostly transparent, skipping background removal")
                output_image = input_image
            else:
                print("Image has alpha channel, skipping background removal")
                output_image = input_image
        else:
            try:
                import warnings
                from rembg import remove, new_session
                import onnxruntime as ort
                import signal
                from contextlib import contextmanager
                import time
                
                class TimeoutException(Exception): pass
                
                @contextmanager
                def time_limit(seconds):
                    def signal_handler(signum, frame):
                        raise TimeoutException("Timed out!")
                    signal.signal(signal.SIGALRM, signal_handler)
                    signal.alarm(seconds)
                    try:
                        yield
                    finally:
                        signal.alarm(0)
                
                warnings.filterwarnings('ignore')
                ort.set_default_logger_severity(3)
                
                try:
                    session = new_session(providers=['CUDAExecutionProvider'])
                    start_time = time.time()
                    
                    with time_limit(15):
                        output_image = remove(
                            input_image,
                            session=session,
                            alpha_matting=True,
                            alpha_matting_erode_size=edge_quality
                        )
                        
                except (TimeoutException, Exception) as e:
                    print("Background removal taking too long or failed, trying with reduced quality...")
                    try:
                        output_image = remove(
                            input_image,
                            session=session,
                            alpha_matting=False,
                            post_process_mask=True
                        )
                    except Exception as e2:
                        print(f"Background removal failed completely, using original image: {str(e2)}")
                        output_image = input_image.convert('RGBA')
                
            except ImportError:
                print("Please install rembg: pip install rembg")
                raise ImportError("rembg not installed. Please install with: pip install rembg")
        
        bg_color = (255, 255, 255) if white_bg else (0, 0, 0)
        square_bg = Image.new('RGBA', (size, size), bg_color)
        
        paste_x = (size - w) // 2
        paste_y = (size - h) // 2
        
        if output_image.mode == 'RGBA':
            square_bg.paste(output_image, (paste_x, paste_y), output_image.split()[3])
        else:
            square_bg.paste(output_image, (paste_x, paste_y))
        
        square_bg = square_bg.convert('RGB')
        
        output_tensor = torch.from_numpy(np.array(square_bg).astype(np.float32) / 255.0)
        output_tensor = output_tensor.unsqueeze(0)
        
        return (output_tensor,)


class TrellisInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TRELLIS_MODEL",),
                "image1": ("IMAGE",),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "sparse_steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "sparse_cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0}),
                "slat_steps": ("INT", {"default": 16, "min": 1, "max": 50}),
                "slat_cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0}),
                "num_views": ("INT", {"default": 5, "min": 1, "max": 36}),
                "camera_distance": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0}),
                "camera_fov": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
                "mode": (["stochastic", "multidiffusion"], {"default": "stochastic"})

            },
            "optional":{
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("MESH", "GAUSSIAN", "IMAGE")
    RETURN_NAMES = ("mesh", "gaussian", "preview_images")
    FUNCTION = "generate"
    CATEGORY = "Trellis"

    def generate(self, model, image1, image2=None, image3=None, seed=1, 
                sparse_steps=8, sparse_cfg=7.5, slat_steps=16, slat_cfg=4.5, 
                mode="stochastic", num_views=5, camera_distance=2.0, camera_fov=30.0):
        device = mm.get_torch_device()
        model.to(device)
        
        images = []
        for img in [image1, image2, image3]:
            if img is not None:
                i = 255. * img[0].cpu().numpy()
                pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                images.append(pil_img)
        
        if len(images) > 1:
            print("Multiple images mode")
            outputs = model.run_multi_image(
                images,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": sparse_steps,
                    "cfg_strength": sparse_cfg,
                },
                slat_sampler_params={
                    "steps": slat_steps,
                    "cfg_strength": slat_cfg,
                },
                mode=mode
            )
        else:
            print("Single image mode")
            outputs = model.run(
                images[0],
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": sparse_steps,
                    "cfg_strength": sparse_cfg,
                },
                slat_sampler_params={
                    "steps": slat_steps,
                    "cfg_strength": slat_cfg,
                }
            )
        
        mesh, gaussian = outputs['mesh'][0], outputs['gaussian'][0]
        
        preview_images = render_utils.render_n_views(
            gaussian,
            n_views=num_views,
            resolution=512,
            r=camera_distance,
            fov=camera_fov
        )
        
        images_list = []
        for img in preview_images:
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
                
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            images_list.append(img_tensor)
        
        preview_tensor = torch.cat(images_list, dim=0)
        
        return (mesh, gaussian, preview_tensor)

NODE_CLASS_MAPPINGS = {
    "LoadTrellisModel": LoadTrellisModel,
    "TrellisInference": TrellisInference,
    "SaveGLBFile": SaveGLBFile,
    "RembgSquare": RembgSquare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrellisModel": "Load Trellis Model",
    "TrellisInference": "Trellis Inference",
    "SaveGLBFile": "Save GLB File",
    "RembgSquare": "Remove Background (Square)",
}