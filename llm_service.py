import os
from typing import List, Tuple, Optional
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import uvicorn
from PIL import Image
import os
import traceback
import logging
import gc
import time
from utils import decode_image
import cv2
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('service.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variable to store LLM instance
llm_instance = None

# Global variables for attention visualization
model_cache = {
    'model': None,
    'tokenizer': None,
    'device': None,
    'hook_logger': None,
    'model_name': None
}

# Memory management variables
memory_usage_stats = {
    'last_cleanup': time.time(),
    'cleanup_interval': 300,  # Cleanup every 5 minutes
    'max_memory_usage': 0.9,  # 90% of available GPU memory
    'request_count': 0,
    'oom_detected': False,
    'last_oom_time': 0
}

# Model configuration mapping
MODEL_CONFIGS = {
    "qwen2.5-vl-7b": {
        "path": "/media/zhongling/huggingface/Qwen2.5-VL-7B-Instruct",
        "type": "qwen",
        "layer_index": 16,
        "threshold": 0.2,
        "num_layers": 28
    },
    "qwen2-vl-7b": {
        "path": "/media/zhongling/lwm/models/Qwen2-VL-7B-Instruct",
        "type": "qwen",
        "layer_index": 16,
        "threshold": 0.2,
        "num_layers": 28
    },
    "pixtral-12b": {
        "path": "/media/data/lwm/models/pixtral-12b",
        "type": "pixtral",
        "layer_index": 10,
        "threshold": 0.2,
        "num_layers": 40
    },
    "llava-v1.5-13b": {
        "path": "/media/zhongling/lwm/models/llava-v1.5-13b",
        "type": "llava",
        "layer_index": 10,
        "threshold": 0.2,
        "num_layers": 40
    },
    "llava-v1.6-13b": {
        "path": "/media/zhongling/huggingface/llava-v1.6-vicuna-13b",
        "type": "llava",
        "layer_index": 10,
        "threshold": 0.2,
        "num_layers": 40
    },
    "table-llava-v1.5-13b": {
        "path": "/media/zhongling/lwm/models/table-llava-v1.5-13b-hf",
        "type": "table-llava",
        "layer_index": 10,
        "threshold": 0.2,
        "num_layers": 40
    },
    "uitars-1.5-7b": {
        "path": "/media/zhongling/lwm/models/UI-TARS-1.5-7B",
        "type": "uitars",
        "layer_index": 10,
        "threshold": 0.2,
        "num_layers": 28
    },
    "gemma-3-12b": {
        "path": "/media/data/lwm/models/gemma-3-12b-it",
        "type": "gemma",
        "layer_index": 17,
        "threshold": 0.2,
        "num_layers": 48
    },
    "gemma-3-4b": {
        "path": "/root/models/gemma-3-4b-it",
        "type": "gemma",
        "layer_index": 17,
        "threshold": 0.2,
        "num_layers": 34
    },
    "siglip-400m": {
        "path": "google/siglip-so400m-patch14-384",
        "type": "siglip",
        "layer_index": 17,
        "threshold": 0.2,
        "num_layers": 27
    },
    "siglip-base": {
        "path": "google/siglip-base-patch16-224",
        "type": "siglip",
        "layer_index": 11,
        "threshold": 0.2,
        "num_layers": 12
    },
    "vit-l-14-336": {
        "path": "openai/clip-vit-large-patch14-336",#"ViT-L-14-336",
        "type": "vit",
        "layer_index": 22,
        "threshold": 0.3,
        "num_layers": 24
    },
}

# Export MODEL_CONFIGS for use by other modules
__all__ = ['MODEL_CONFIGS', 'create_vlm_instance', 'get_gpu_memory_info', 'clear_gpu_memory', 'is_oom_error', 'handle_oom_error', 'check_memory_and_cleanup', 'generate_token_attention_map']

def create_vlm_instance(model_name: str, temperature: float = 0.1, max_tokens: int = 1024):
    """Create a VLM instance based on model type
    
    Args:
        model_name: Name of the model to create
        temperature: Sampling temperature (default: 0.1)
        max_tokens: Maximum tokens in response (default: 1024)
        
    Returns:
        VLM instance
        
    Raises:
        ValueError: If model type is not supported
    """
    # Get model config
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model_path = config["path"]
    model_type = config["type"]
    
    # Initialize model based on type
    if model_type == "llava":
        from vlm._llava import LLaVA
        instance = LLaVA(model_path, temperature=temperature, max_tokens=max_tokens)
    elif model_type == "table-llava":
        from vlm.table_llava import TableLLaVA
        instance = TableLLaVA(model_path, temperature=temperature, max_tokens=max_tokens)
    elif model_type == "qwen":
        from vlm.qwen import Qwen
        instance = Qwen(model_path, type=model_name, temperature=temperature, max_tokens=max_tokens)
    elif model_type == "pixtral":
        from vlm.pixtral import Pixtral
        instance = Pixtral(model_path, temperature=temperature, max_tokens=max_tokens)
    elif model_type == "uitars":
        from vlm.uitars import UITARS
        instance = UITARS(model_path, temperature=temperature, max_tokens=max_tokens)
    elif model_type == "gemma":
        from vlm.gemma import Gemma
        instance = Gemma(model_path, temperature=temperature, max_tokens=max_tokens)
    elif model_type == "siglip":
        from vlm.siglip import SigLIP
        instance = SigLIP(model_path)
    elif model_type == "vit":
        from vlm.vit import ViT
        instance = ViT(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return instance

def generate_token_attention_map(
    llm_instance,
    model_name: str,
    image_base64: str,
    prompt: str,
    visualize_text: str,
    rm_spikes: bool = False,
    use_averaged_attention: bool = False,
    use_all_layers: bool = False
):
    """Generate token attention map for specific text on image
    
    Args:
        llm_instance: Loaded VLM instance
        model_name: Name of the model
        image_base64: base64 encoded image
        prompt: Question/prompt text
        visualize_text: Text to visualize attention for
        rm_spikes: Whether to remove spikes (default: False)
        use_averaged_attention: Whether to average attention across all visualize tokens (default: False)
        
    Returns:
        Dict containing marked image and contour information
    """
    try:
        # Decode base64 image
        image = decode_image(image_base64)
        llm_instance.target_aspect_ratio = image.width / image.height
        inputs = llm_instance.prepare_inputs(image, prompt)
        input_tokens = llm_instance.processor.tokenizer.convert_ids_to_tokens(
            inputs['input_ids'][0]
        )

        # Get original image dimensions
        original_width, original_height = image.size

        layer_index = MODEL_CONFIGS[model_name]['layer_index']
        if use_all_layers:
            num_layers = MODEL_CONFIGS[model_name]['num_layers']
            # all_layers = list(np.arange(0,num_layers,num_layers/28).astype(int))
            all_layers = list(range(num_layers))
        else:
            all_layers = [layer_index]
        res = {layer_index: {} for layer_index in all_layers}

        for i, layer_index in enumerate(all_layers):
            # Extract attention data
            if model_name.startswith('vit') or model_name.startswith('siglip'):
                attention_map = llm_instance.extract_attention(inputs, layer_idx=layer_index)
            else:
                if i == 0:
                    attentions = llm_instance.extract_attention(inputs, store_both=True)
                
                    # Get token information
                    token_info = llm_instance.get_input_text_tokens(input_tokens)
                    if not token_info:
                        raise Exception("Unable to get token information")
                        
                    input_text_tokens, token_positions = token_info
                    
                    # Tokenize visualize_text to get all tokens
                    visualize_tokens = llm_instance.tokenize_text(visualize_text)
                    
                    # Get layer index
                    print(f"input_text_tokens: {input_text_tokens}")
                    print(f"target_token_text: {visualize_text}")
                    print(f"target_tokens: {visualize_tokens}")
                
                if use_averaged_attention:
                    if i == 0:
                        # Find all target token indices for the visualize_text
                        target_token_indices = []
                        target_token_length = len(visualize_tokens)
                        
                        # Find the sequence of tokens in input_text_tokens
                        for i in range(len(input_text_tokens) - target_token_length + 1):
                            if input_text_tokens[i+1:i+target_token_length] == visualize_tokens[1:]:
                                start_idx = i
                                print('Found target token: ', input_text_tokens[start_idx:start_idx + target_token_length])
                                target_token_indices = list(range(token_positions[start_idx], 
                                                token_positions[start_idx + target_token_length-1]+1))
                                break
                        
                        if not target_token_indices:
                            raise Exception(f"Target text sequence '{visualize_text}' not found in input tokens")
                    
                    # Collect attention maps for all tokens
                    layer_token_maps = []
                    for token_idx in target_token_indices:
                        heatmap = llm_instance.get_input_text_token_attention_heatmap(
                            input_tokens, attentions, layer_index, token_idx, rm_spikes=rm_spikes
                        )
                        layer_token_maps.append(heatmap)
                    
                    # Calculate averaged attention map
                    if not layer_token_maps:
                        raise Exception("No attention maps found for averaging")
                    attention_map = np.mean(layer_token_maps, axis=0)
                    
                else:
                    if i == 0:
                        # Original implementation: use only the last token
                        visualize_token = visualize_tokens[-1]
                        target_token_idx = token_positions[input_text_tokens.index(visualize_token)]
                    
                    # Get token attention map
                    attention_map = llm_instance.get_input_text_token_attention_heatmap(
                        input_tokens, attentions, layer_index, target_token_idx, rm_spikes=rm_spikes
                    )
            
            # Resize attention map to original image dimensions
            attention_map_resized = cv2.resize(attention_map, (original_width, original_height))
            # Find the global maximum point in the attention map and ensure the result is a tuple of Python ints, not numpy types
            max_index = int(np.argmax(attention_map_resized))
            # Swap x and y: np.unravel_index returns (y, x), so we reverse the order for (x, y)
            y, x = np.unravel_index(max_index, attention_map_resized.shape)
            global_max_point = (int(x), int(y))

            if use_all_layers:
                res[layer_index] = {
                    "global_max_point": global_max_point
                }
            else:
                # Binary threshold for contour detection
                threshold_value = MODEL_CONFIGS[model_name]['threshold'] * 255
                binary_mask = cv2.threshold(attention_map_resized, threshold_value, 255, cv2.THRESH_BINARY)[1]
                
                # Find contours
                contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    raise Exception("No contours found")
                
                # Mark contours on original image
                image_with_contours = np.array(image).copy()
                contour_info = []
                
                # Create overlay with alpha channel
                overlay = np.zeros((image_with_contours.shape[0], image_with_contours.shape[1], 4), dtype=np.uint8)
                
                # Generate different random colors for each contour
                np.random.seed(42)  # Set random seed for color consistency
                colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
                
                import math
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = min(original_width, original_height)/500
                font_scale = 0.8*scale
                thickness = math.ceil(2*scale)
                
                for k, contour in enumerate(contours):
                    # Create mask for current contour
                    mask = np.zeros((original_height, original_width), dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    
                    # Get pixels inside contour using mask
                    roi = cv2.bitwise_and(np.array(image), np.array(image), mask=mask)
                    non_zero = roi[roi != 0]  # Get non-black pixels
                    
                    if non_zero.size > 0:  # Make sure ROI is not empty
                        # Calculate variance for each channel where mask is non-zero
                        variance = np.mean([np.var(roi[mask == 255][:, c]) for c in range(3)])
                        if variance < 50:  # Skip if variance is low (background)
                            continue
                    
                    attention_roi = cv2.bitwise_and(attention_map_resized, attention_map_resized, mask=mask)
                    contour_attention = attention_roi[mask == 255]
                    max_attention = np.max(contour_attention)
                    mean_attention = np.mean(contour_attention)

                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        x, y, w, h = cv2.boundingRect(contour)
                        cx, cy = x + w//2, y + h//2
                    
                    # Get color for current contour and create color tuples
                    color = colors[k]
                    color_with_alpha = (int(color[0]), int(color[1]), int(color[2]), int(0.3 * 255))
                    color_edge = (int(color[0]), int(color[1]), int(color[2]), 255)
                        
                    # Fill contour with color and opacity
                    cv2.drawContours(overlay, [contour], -1, color_with_alpha, -1)
                    
                    # Draw contour edge with full opacity
                    cv2.drawContours(overlay, [contour], -1, color_edge, 2)
                    
                    label = f"{mean_attention:.1f}"
                    # Draw black background rectangle for label
                    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                    bg_rect_x = cx - 2
                    bg_rect_y = cy - text_size[1] - 2
                    bg_rect_w = text_size[0] + 4
                    bg_rect_h = text_size[1] + 4
                    cv2.rectangle(overlay, (bg_rect_x, bg_rect_y), 
                                (bg_rect_x + bg_rect_w, bg_rect_y + bg_rect_h), 
                                (0,0,0,255), -1)
                    # Draw white text on black background
                    cv2.putText(overlay, label, (cx, cy), font, font_scale, (255,255,255,255), thickness)
                    
                    # Record contour info
                    x, y, w, h = cv2.boundingRect(contour)
                    info = {
                        "id": k,
                        "center": [int(cx), int(cy)],
                        "bbox": [x, y, x+w, y+h],
                        "max_attention": float(max_attention),
                        "mean_attention": float(mean_attention)
                    }
                    contour_info.append(info)
                
                # Blend overlay with original image using alpha channel
                alpha_mask = overlay[..., 3:] / 255.0
                image_with_contours = image_with_contours * (1 - alpha_mask) + overlay[..., :3] * alpha_mask
                image_with_contours = image_with_contours.astype(np.uint8)
                
                # Convert marked image to base64
                import io
                import base64
                with io.BytesIO() as buffer:
                    Image.fromarray(image_with_contours).save(buffer, format='PNG')
                    buffer.seek(0)
                    marked_image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                return {
                    "marked_image": marked_image_base64,
                    "marks": contour_info,
                    "global_max_point": global_max_point
                }
        return res
        
    except Exception as e:
        raise

def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if not torch.cuda.is_available():
        return {"available": False, "total": 0, "used": 0, "free": 0, "usage_percent": 0}
    
    try:
        # Get memory info for the current device
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - cached_memory
        
        return {
            "available": True,
            "total": total_memory,
            "allocated": allocated_memory,
            "cached": cached_memory,
            "free": free_memory,
            "usage_percent": (cached_memory / total_memory) * 100
        }
    except Exception as e:
        logger.warning(f"Error getting GPU memory info: {e}")
        return {"available": False, "total": 0, "used": 0, "free": 0, "usage_percent": 0}

def should_perform_cleanup():
    """Check if memory cleanup should be performed"""
    current_time = time.time()
    memory_info = get_gpu_memory_info()
    
    # Check time-based cleanup
    time_based = (current_time - memory_usage_stats['last_cleanup']) > memory_usage_stats['cleanup_interval']
    
    # Check memory-based cleanup
    memory_based = False
    if memory_info['available']:
        memory_based = memory_info['usage_percent'] > (memory_usage_stats['max_memory_usage'] * 100)
    
    return time_based or memory_based

def clear_gpu_memory():
    """Clear GPU memory cache with enhanced cleanup"""
    try:
        if torch.cuda.is_available():
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage after cleanup
            memory_info = get_gpu_memory_info()
            
    except Exception as e:
        logger.warning(f"Error clearing GPU memory: {e}")

def perform_memory_cleanup():
    """Perform comprehensive memory cleanup"""
    logger.info("Performing memory cleanup...")
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Reset attention data if model instance exists
    global llm_instance
    if llm_instance is not None and hasattr(llm_instance, 'reset_state'):
        try:
            llm_instance.reset_state()
            logger.info("Model state reset successfully")
        except Exception as e:
            logger.warning(f"Error resetting model state: {e}")
    
    # Update cleanup timestamp
    memory_usage_stats['last_cleanup'] = time.time()
    
    logger.info("Memory cleanup completed")

def check_memory_and_cleanup():
    """Check memory usage and perform cleanup if necessary"""
    if should_perform_cleanup():
        perform_memory_cleanup()

def is_oom_error(error_msg: str) -> bool:
    """Check if error message indicates OOM error"""
    print(f"error_msg: {error_msg}")
    oom_keywords = [
        'out of memory',
        'cuda out of memory',
        'cuda oom',
        'insufficient memory',
        'memory allocation failed',
        'cuda error: out of memory',
        'runtimeerror: cuda out of memory'
    ]
    error_lower = error_msg.lower()
    return any(keyword in error_lower for keyword in oom_keywords)

def handle_oom_error():
    """Handle OOM error by unloading model only"""
    global llm_instance, memory_usage_stats
    
    logger.warning("OOM error detected, unloading model...")
    
    try:
        # Mark OOM detected
        memory_usage_stats['oom_detected'] = True
        memory_usage_stats['last_oom_time'] = time.time()
        
        # Force cleanup
        clear_gpu_memory()
        
        # Unload current model if exists
        if llm_instance is not None:
            try:
                llm_instance.unload_model()
                logger.info("Model unloaded due to OOM")
            except Exception as e:
                logger.error(f"Error unloading model during OOM: {e}")
        
        # Clear model cache
        model_cache['model'] = None
        model_cache['tokenizer'] = None
        model_cache['device'] = None
        model_cache['hook_logger'] = None
        model_cache['model_name'] = None
        
        # Force garbage collection
        gc.collect()
        
        # Wait a bit for memory to be freed
        time.sleep(2)
        
        # Clear GPU memory again
        clear_gpu_memory()
        
        # Set llm_instance to None
        llm_instance = None
        
        logger.info("OOM recovery completed - model unloaded")
        
    except Exception as e:
        logger.error(f"Error during OOM recovery: {e}")

class StartModelRequest(BaseModel):
    model_name: str
    temperature: float = 0
    max_tokens: int = 1024

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: List[MessageContent]

class ChatRequest(BaseModel):
    messages: List[Message]

class AttentionMapRequest(BaseModel):
    image_path: str
    question: str
    layer_index: Optional[int] = 20
    output_path: Optional[str] = None

class TokenAttentionMapRequest(BaseModel):
    model_type: str
    image: str  # base64 encoded image
    prompt: str
    visualize_text: str
    rm_spikes: Optional[bool] = False
    use_averaged_attention: Optional[bool] = False
    use_all_layers: Optional[bool] = False 

@app.post("/get_token_attention_map")
async def get_token_attention_map(request: TokenAttentionMapRequest):
    global llm_instance
    
    try:
        # Check memory and perform cleanup if necessary
        check_memory_and_cleanup()
        
        current_model = model_cache.get('model_name')
        
        if llm_instance is None:
            logger.info(f"No model loaded, creating new instance for {request.model_type}")
            try:
                llm_instance = create_vlm_instance(request.model_type)
                llm_instance.ensure_model_loaded()
                
                # Update model cache
                model_cache['model'] = llm_instance.model
                model_cache['tokenizer'] = llm_instance.tokenizer
                model_cache['device'] = llm_instance.device
                model_cache['model_name'] = request.model_type
                
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to create model instance for {request.model_type}: {str(e)}"
                )
        elif current_model != request.model_type:
            logger.info(f"Different model loaded ({current_model}), switching to {request.model_type}")
            try:
                # Unload current model with enhanced cleanup
                llm_instance.unload_model()
                clear_gpu_memory()
                
                # Create new instance
                llm_instance = create_vlm_instance(request.model_type)
                llm_instance.ensure_model_loaded()
                
                # Update model cache
                model_cache['model'] = llm_instance.model
                model_cache['tokenizer'] = llm_instance.tokenizer
                model_cache['device'] = llm_instance.device
                model_cache['model_name'] = request.model_type
                
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to switch model to {request.model_type}: {str(e)}"
                )
        else:
            logger.info(f"Using already loaded model {current_model}")
        
        # Use the public function to generate token attention map
        return generate_token_attention_map(
            llm_instance=llm_instance,
            model_name=request.model_type,
            image_base64=request.image,
            prompt=request.prompt,
            visualize_text=request.visualize_text,
            rm_spikes=request.rm_spikes,
            use_averaged_attention=request.use_averaged_attention,
            use_all_layers=request.use_all_layers
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error generating token attention map: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Check if this is an OOM error
        if is_oom_error(error_msg):
            logger.warning("OOM error detected in get_token_attention_map")
            handle_oom_error()
            raise HTTPException(
                status_code=503,  # Service Unavailable
                detail="Out of memory error. Model has been unloaded. Please check /health and reload model if needed."
            )
        
        # Perform emergency cleanup on other errors
        try:
            clear_gpu_memory()
            if llm_instance is not None and hasattr(llm_instance, 'reset_state'):
                llm_instance.reset_state()
        except Exception as cleanup_error:
            logger.error(f"Error during emergency cleanup: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_vlm")
async def start_vlm(request: StartModelRequest):
    """Start a new LLM instance with specified parameters"""
    global llm_instance
    
    try:
        # Check memory and perform cleanup if necessary
        check_memory_and_cleanup()
        
        # Check if a different model is already loaded
        if llm_instance is not None:
            current_model = model_cache.get('model_name', 'unknown')
            
            if current_model == request.model_name:
                logger.info(f"Model {request.model_name} is already running")
                return {
                    "status": "success", 
                    "message": f"Model {request.model_name} is already running"
                }
            else:
                # Different model is loaded, need to stop it first
                logger.info(f"Stopping current model {current_model} to load {request.model_name}")
                
                # Stop the current model
                try:
                    # Unload model
                    llm_instance.unload_model()
                    
                    # Clear model cache
                    model_cache['model'] = None
                    model_cache['tokenizer'] = None
                    model_cache['device'] = None
                    model_cache['hook_logger'] = None
                    model_cache['model_name'] = None
                    
                    # Clear GPU memory
                    clear_gpu_memory()
                    
                    llm_instance = None
                    
                    logger.info(f"Successfully stopped model {current_model}")
                    
                except Exception as e:
                    logger.error(f"Error stopping current model: {e}")
                    # Continue anyway to try loading the new model
        
        logger.info(f"Starting model {request.model_name}")
        
        # Create new model instance
        llm_instance = create_vlm_instance(
            model_name=request.model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Load model
        llm_instance.ensure_model_loaded()
        
        # Store in global cache for attention visualization
        model_cache['model'] = llm_instance.model
        model_cache['tokenizer'] = llm_instance.tokenizer
        model_cache['device'] = llm_instance.device
        model_cache['model_name'] = request.model_name
        
        return {
            "status": "success", 
            "message": f"Model {request.model_name} loaded successfully"
        }
            
    except Exception as e:
        logger.error(f"Error in start_vlm: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=str(e)
        )

@app.post("/chat_vlm")
async def chat_vlm(request: ChatRequest):
    """Chat with the loaded VLM model"""
    global llm_instance, memory_usage_stats
    
    try:
        # Increment request counter
        memory_usage_stats['request_count'] += 1
        
        # Check memory and perform cleanup if necessary
        check_memory_and_cleanup()
        
        if llm_instance is None:
            raise HTTPException(
                status_code=400,
                detail="No model loaded. Please call /start_vlm first."
            )
            
        # Convert Pydantic models to dictionaries
        messages = []
        for msg in request.messages:
            message_dict = {
                "role": msg.role,
                "content": [
                    {
                        "type": content.type,
                        **({"text": content.text} if content.text is not None else {}),
                        **({"image_url": content.image_url} if content.image_url is not None else {})
                    }
                    for content in msg.content
                ]
            }
            messages.append(message_dict)
            
        # Call the model with the converted messages
        response = llm_instance(messages)
        
        return {
            "status": "success",
            "response": response
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in chat_vlm: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Check if this is an OOM error
        if is_oom_error(error_msg):
            logger.warning("OOM error detected in chat_vlm")
            handle_oom_error()
            raise HTTPException(
                status_code=503,  # Service Unavailable
                detail="Out of memory error. Model has been unloaded. Please check /health and reload model if needed."
            )
        
        # Perform emergency cleanup on other errors
        try:
            clear_gpu_memory()
            if llm_instance is not None and hasattr(llm_instance, 'reset_state'):
                llm_instance.reset_state()
        except Exception as cleanup_error:
            logger.error(f"Error during emergency cleanup: {cleanup_error}")
        
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.post("/stop_vlm")
async def stop_vlm():
    """Stop the running model and free GPU resources"""
    global llm_instance
    
    if llm_instance is None:
        return {"status": "success", "message": "No model is running"}
    
    try:
        # Unload model through visualizer
        llm_instance.unload_model()
        
        # Clear model cache
        model_cache['model'] = None
        model_cache['tokenizer'] = None
        model_cache['device'] = None
        model_cache['hook_logger'] = None
        model_cache['model_name'] = None
        
        # Clear CUDA cache
        clear_gpu_memory()
            
        llm_instance = None
        return {"status": "success", "message": "Model stopped and resources freed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global llm_instance
    memory_info = get_gpu_memory_info()
    
    # Determine status based on model and OOM state
    if llm_instance is None:
        if memory_usage_stats['oom_detected']:
            status = 'unloaded_due_to_oom'
            message = 'Model was unloaded due to out of memory error'
        else:
            status = 'no_model_loaded'
            message = 'No model is currently loaded'
    else:
        status = 'healthy'
        message = 'Model is loaded and ready'
    
    return {
        'status': status,
        'message': message,
        'model_loaded': llm_instance is not None,
        'model_name': model_cache.get('model_name'),
        'supported_models': list(MODEL_CONFIGS.keys()),
        'gpu_memory': memory_info,
        'request_count': memory_usage_stats['request_count'],
        'last_cleanup': memory_usage_stats['last_cleanup'],
        'oom_detected': memory_usage_stats['oom_detected'],
        'last_oom_time': memory_usage_stats['last_oom_time']
    }

@app.get("/models")
async def list_models():
    """List supported models and their status"""
    global llm_instance
    models_info = {}
    for model_name, config in MODEL_CONFIGS.items():
        models_info[model_name] = {
            "path": config["path"],
            "type": config["type"],
            "exists": os.path.exists(config["path"]),
            "loaded": model_cache.get('model_name') == model_name and llm_instance is not None
        }
    
    return {
        "status": "success",
        "models": models_info,
        "currently_loaded": model_cache.get('model_name') if llm_instance is not None else None
    }

@app.post("/cleanup_memory")
async def cleanup_memory():
    """Manual memory cleanup endpoint"""
    try:
        perform_memory_cleanup()
        return {"status": "success", "message": "Memory cleanup completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_model")
async def reload_model():
    """Manually reload the current model (useful after OOM errors)"""
    global llm_instance, memory_usage_stats
    
    try:
        current_model = model_cache.get('model_name')
        if not current_model:
            raise HTTPException(
                status_code=400,
                detail="No model is currently loaded"
            )
        
        logger.info(f"Manually reloading model {current_model}")
        
        # Unload current model
        if llm_instance is not None:
            try:
                llm_instance.unload_model()
                logger.info("Current model unloaded")
            except Exception as e:
                logger.error(f"Error unloading current model: {e}")
        
        # Clear model cache
        model_cache['model'] = None
        model_cache['tokenizer'] = None
        model_cache['device'] = None
        model_cache['hook_logger'] = None
        model_cache['model_name'] = None
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Wait a bit for memory to be freed
        time.sleep(2)
        
        # Create new model instance
        llm_instance = create_vlm_instance(
            model_name=current_model,
            temperature=0.1,
            max_tokens=1024
        )
        
        # Load model
        llm_instance.ensure_model_loaded()
        
        # Update model cache
        model_cache['model'] = llm_instance.model
        model_cache['tokenizer'] = llm_instance.tokenizer
        model_cache['device'] = llm_instance.device
        model_cache['model_name'] = current_model
        
        # Reset OOM flag
        memory_usage_stats['oom_detected'] = False
        
        return {
            "status": "success",
            "message": f"Model {current_model} reloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_oom_status")
async def reset_oom_status():
    """Reset OOM status after model has been reloaded"""
    global memory_usage_stats
    
    try:
        memory_usage_stats['oom_detected'] = False
        memory_usage_stats['last_oom_time'] = 0
        
        return {
            "status": "success",
            "message": "OOM status reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting OOM status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False  # Disable auto-reload to prevent continuous file change detection
    ) 