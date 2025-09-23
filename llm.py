import requests
import os
import time
import threading
from functools import wraps
from PIL import Image
import json
from utils import resize_image, encode_image
import math

# Fix numpy import issue in CUDA environment
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"


def with_timeout(timeout_seconds):
    """Decorator to add timeout to function calls using threading (Windows compatible)"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                # Thread is still running, timeout occurred
                raise TimeoutError(
                    f"Function call timed out after {timeout_seconds} seconds"
                )

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


class AbstractLLM:
    """
    Abstract class for interacting with various LLM APIs.
    Supports different models through a common interface.
    """

    def __init__(self, model_name, temperature=0.1, max_tokens=4096):
        """
        Initialize LLM instance with specified model and parameters.

        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
        """
        temperature = max(0.1, temperature)
        self.timeout_seconds = 60
        if model_name == "gpt-4o":
            real_name = "gpt-4o-2024-05-13"
            llm = VLMAPI(
                model_name=real_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif model_name == "claude-3.5":
            real_name = "claude-3-5-sonnet-20240620"
            llm = VLMAPI(real_name, temperature=temperature, max_tokens=max_tokens)
        elif model_name == "gemini-2.5-flash":
            llm = VLMAPI(model_name, temperature=temperature, max_tokens=max_tokens)
        elif model_name in [
            "qwen2-vl-7b",
            "gemma-3-12b",
            "gemma-3-4b",
            "siglip-400m",
        ]:
            llm = DirectLocalLLMAPI(
                model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.timeout_seconds = 300
        else:
            raise ValueError(f"Model {model_name} not supported")

        self.model_name = model_name
        self.llm = llm
        if model_name.startswith("gpt-4o") or model_name.startswith("claude-3.5"):
            self.patch_size = 32
        elif model_name.startswith("qwen"):
            self.patch_size = 28
        elif model_name.startswith("gemma") or model_name == 'siglip-400m':
            self.patch_size = 14
        else:
            self.patch_size = 16

    def __call__(self, messages, max_retries=3):
        """
        Call LLM with timeout and retry mechanism

        Args:
            messages: Messages to send to LLM
            max_retries: Maximum number of retry attempts (default: 3)
            timeout_seconds: Timeout in seconds for each attempt (default: 30)

        Returns:
            LLM response or None if all attempts fail
        """
        for attempt in range(max_retries):
            try:
                # if 1:
                @with_timeout(self.timeout_seconds)
                def call_llm():
                    return self.llm(messages)

                response = call_llm()
                return response

            except TimeoutError:
                print(
                    f"Attempt {attempt + 1}/{max_retries}: LLM call timed out after {self.timeout_seconds} seconds"
                )
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Progressive backoff: 5s, 10s, 15s
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

            except Exception as e:
                print(
                    f"Attempt {attempt + 1}/{max_retries}: LLM call failed with error: {e}"
                )
                # print("prompt:")
                # for msg in messages:
                #     print(msg["content"][0]["text"])
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Progressive backoff: 5s, 10s, 15s
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

        print(f"All {max_retries} attempts failed")
        return None

    def get_token_attention_map(self, image_base64, prompt, visualize_text, rm_spikes=False, use_all_layers=False):
        return self.llm.get_token_attention_map(image_base64, prompt, visualize_text, rm_spikes, use_all_layers)

    def resize_image(self, image):
        width, height = image.size

        if self.model_name == "gpt-4o":
            # GPT-4o resize rules (updated):
            # 1. Adjust short edge to 768 pixels (scale up or down)
            # 2. Adjust long edge proportionally, but not exceeding 2048 pixels
            # 3. Image is then divided into 512x512 blocks, each consuming 170 tokens

            # Step 1: Identify short and long edges
            if width <= height:
                short_edge, long_edge = width, height
                is_width_short = True
            else:
                short_edge, long_edge = height, width
                is_width_short = False

            # Step 2: Calculate scale patch_size based on short edge = 768
            scale_factor = 768 / short_edge

            # Step 3: Calculate new dimensions
            if is_width_short:
                new_width = 768
                new_height = math.ceil(height * scale_factor)
            else:
                new_height = 768
                new_width = math.ceil(width * scale_factor)

            # Step 4: Check if long edge exceeds 2048, if so, scale down
            if max(new_width, new_height) > 2048:
                long_edge_scale = 2048 / max(new_width, new_height)
                new_width = math.ceil(new_width * long_edge_scale)
                new_height = math.ceil(new_height * long_edge_scale)

            rw, rh = new_width, new_height
            image = image.resize((rw, rh))

        elif self.model_name == "claude-3.5":
            # Claude-3.5 resize rules
            # Token calculation: (width * height) / 750
            # Max tokens: ~1600, Max dimension: 1568

            estimated_tokens = (width * height) / 750
            needs_token_resize = estimated_tokens > 1600
            needs_dimension_resize = width > 1568 or height > 1568

            if needs_dimension_resize or needs_token_resize:
                if needs_dimension_resize:
                    if width > 1568:
                        dimension_ratio = 1568 / width
                    elif height > 1568:
                        dimension_ratio = 1568 / height
                    else:
                        dimension_ratio = 1.0
                else:
                    dimension_ratio = 1.0

                if needs_token_resize:
                    target_pixels = 1600 * 750
                    current_pixels = width * height
                    token_ratio = math.sqrt(target_pixels / current_pixels)
                else:
                    token_ratio = 1.0

                resize_ratio = min(dimension_ratio, token_ratio)

                rw = max(1, math.ceil(width * resize_ratio))
                rh = max(1, math.ceil(height * resize_ratio))
            else:
                rw, rh = width, height

            image = image.resize((rw, rh))

        elif self.model_name == "qwen2-vl-7b":
            min_pixels = 4 * self.patch_size * self.patch_size
            max_pixels = 1280 * self.patch_size * self.patch_size
            image = self.smart_resize(
                image,
                patch_size=self.patch_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        elif self.model_name == "gemini-2.5-flash":
            image = resize_image(image, max_size=3072)
        elif self.model_name.startswith("gemma-3"):
            image = image.resize((896, 896))
        elif self.model_name == "siglip-400m":
            image = image.resize((384, 384))
        else:
            image = resize_image(image, max_size=1024)

        resized_width, resized_height = image.size
        resize_ratio_width = width / resized_width
        resize_ratio_height = height / resized_height
        print(
            f"resize from {width}x{height} to {resized_width}x{resized_height}, resize_ratio: {1/resize_ratio_width:.2f}x{1/resize_ratio_height:.2f}"
        )
        return image, resize_ratio_width, resize_ratio_height

    def smart_resize(self, img, patch_size, min_pixels, max_pixels):
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'patch_size'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        MAX_RATIO = 200
        width, height = img.size
        if height < patch_size or width < patch_size:
            raise ValueError(f"height: {height} or width: {width} must be larger than patch_size: {patch_size}")
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")

        def round_by_factor(x, f): return round(x / f) * f
        def floor_by_factor(x, f): return math.floor(x / f) * f
        def ceil_by_factor(x, f): return math.ceil(x / f) * f

        h_bar = max(patch_size, round_by_factor(height, patch_size))
        w_bar = max(patch_size, round_by_factor(width, patch_size))

        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(patch_size, floor_by_factor(height / beta, patch_size))
            w_bar = max(patch_size, floor_by_factor(width / beta, patch_size))

        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, patch_size)
            w_bar = ceil_by_factor(width * beta, patch_size)
        img = img.resize((w_bar, h_bar))
        return img


class VLMAPI:
    """
    Client for interacting with OpenAI API and compatible endpoints.
    """

    def __init__(self, model_name, temperature=0, max_tokens=1024):
        """
        Initialize OpenAI API client.

        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/chat/completions"
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages):
        """
        Send question to OpenAI API and get response.

        Args:
            question: Text prompt to send to the model

        Returns:
            Model response text or full response object
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        response = requests.post(self.url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            return result
        else:
            raise Exception(response.json()["error"]["message"])

class DirectLocalLLMAPI:
    """Direct local LLM API that directly loads and manages models without HTTP API calls
    
    This class provides direct access to local models without requiring a separate HTTP service.
    It reuses code from llm_service.py for model creation, memory management, and error handling.
    
    Usage:
        # Create instance
        llm = DirectLocalLLMAPI("qwen2-vl-7b", temperature=0, max_tokens=1024)
        
        # Call model
        response = llm(messages)
        
        # Get attention map
        result = llm.get_token_attention_map(image_base64, prompt, visualize_text, rm_spikes)
        
        # Unload when done
        llm.unload_model()
    """
    
    def __init__(
        self,
        model_name,
        temperature=0,
        max_tokens=1024,
    ):
        """Initialize DirectLocalLLMAPI instance
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (default: 0)
            max_tokens: Maximum tokens in response (default: 1024)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_instance = None
        
        # Import necessary functions from llm_service
        from llm_service import (
            create_vlm_instance, 
            check_memory_and_cleanup, 
            clear_gpu_memory, 
            is_oom_error, 
            handle_oom_error
        )
        self._create_vlm_instance = create_vlm_instance
        self._check_memory_and_cleanup = check_memory_and_cleanup
        self._clear_gpu_memory = clear_gpu_memory
        self._is_oom_error = is_oom_error
        self._handle_oom_error = handle_oom_error

    def _ensure_model_loaded(self):
        """Ensure model is loaded, create if necessary"""
        if self.llm_instance is None:
            self.llm_instance = self._create_vlm_instance(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            self.llm_instance.ensure_model_loaded()

    def __call__(self, messages):
        """Call the model directly with messages
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Model response text
        """
        try:
            self._ensure_model_loaded()
            return self.llm_instance(messages)
        except Exception as e:
            raise e
    
    def get_token_attention_map(self, image_base64, prompt, visualize_text, rm_spikes, use_all_layers):
        """Get token attention map for specific text on image
        
        Args:
            image_base64: base64 encoded image
            prompt: Question/prompt text
            visualize_text: Text to visualize attention for
            rm_spikes: Whether to remove spikes
            
        Returns:
            Dict containing marked image and contour information
        """
        try:
            self._ensure_model_loaded()
            
            # Import and use the public function from llm_service
            from llm_service import generate_token_attention_map
            
            return generate_token_attention_map(
                llm_instance=self.llm_instance,
                model_name=self.model_name,
                image_base64=image_base64,
                prompt=prompt,
                visualize_text=visualize_text,
                rm_spikes=rm_spikes,
                use_all_layers=use_all_layers
            )
            
        except Exception as e:
            raise e

if __name__ == "__main__":
    vlm = AbstractLLM("gemma-3-12b")
    image_path = "data/5/origin.png"
    image = Image.open(image_path).convert("RGB")
    image_path2 = "data/5/grid.png"
    image2 = Image.open(image_path2).convert("RGB")
    question = "What is the difference between the two images?"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image(image)
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image(image2)
                    },
                },
            ],
        }
    ]
    out = vlm(messages)
    print("Response:", out)
