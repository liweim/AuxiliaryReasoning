from .base_vlm import LocalVLM
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from typing import List, Tuple
import math
from utils import decode_image

class Qwen(LocalVLM):
    def __init__(self, model_path, type="qwen2-vl", temperature=0.1, max_tokens=1024):
        super().__init__(model_path, temperature, max_tokens)
        self.vision_token_positions = None
        self.grid_dimensions = None
        self.type = type

    def load_model(self):
        if self.device.type == 'cuda':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        if self.type.startswith("qwen2-vl"):
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                attn_implementation="eager",
                output_attentions=True,
                trust_remote_code=True
            ).to(self.device)
        elif self.type.startswith("qwen2.5-vl"):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                attn_implementation="eager",
                output_attentions=True,
                trust_remote_code=True
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.type}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer 
        self.model.eval()
        
    def compute_vision_token_positions(self, input_tokens: List[str], actual_seq_len: int=None):
        """Compute vision token positions and cache the result for Qwen"""
        if self.vision_token_positions is not None:
            return self.vision_token_positions
        
        # Find vision token positions directly
        image_start = input_tokens.index("<|vision_start|>")+1
        image_end = input_tokens.index("<|vision_end|>")
        text_start = image_end + 1
        text_end = input_tokens[text_start:].index("<|im_end|>") + text_start
        
        # Cache the result
        self.vision_token_positions = {
            'text_start': text_start,
            'text_end': text_end,
            'image_start': image_start,
            'image_end': image_end
        }
        
        if self.verbose:
            # print(f"input tokens: {input_tokens}")
            print(f"  Image tokens: {image_start}:{image_end}")
            print(f"  Text tokens: {text_start}:{text_end}")
        
        return self.vision_token_positions
    
    def __call__(self, messages):
        """Handle Qwen specific inference
        
        Args:
            messages: List of message dictionaries containing role and content
            
        Returns:
            Model response text
        """
        # Convert messages to Qwen format
        self.ensure_model_loaded()

        qwen_messages = []
        for message in messages:
            all_content = []
            for content in message['content']:
                if content['type'] == 'text':
                    all_content.append({"type": "text", "text": content['text']})
                else:
                    # For image content, use the PIL Image directly
                    image = decode_image(content['image_url']['url'])
                    all_content.append({"type": "image", "image": image})
            qwen_messages.append({'role': message['role'], 'content': all_content})

        # Prepare inputs using visualizer
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        
        generated_ids_trimmed = [
            out_ids[len(inputs['input_ids'][i]):] for i, out_ids in enumerate(generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

    def find_best_grid_dimensions(self, n_patches: int, target_aspect_ratio: float) -> Tuple[int, int]:
        """
        Find best grid dimensions considering Qwen2-VL's 2x2 patch token merging.
        The n_patches represents compressed tokens after 2x2 merging.
        We calculate dimensions based on target aspect ratio and then expand.
        """
        # Check if dimensions are already cached for these parameters
        cache_key = (n_patches, target_aspect_ratio)
        if self.grid_dimensions is not None and self.grid_dimensions['key'] == cache_key:
            return self.grid_dimensions['value']
        
        if self.verbose:
            print(f"Finding grid dimensions for {n_patches} compressed tokens")
            print(f"Target aspect ratio: {target_aspect_ratio:.2f}")
        
        # Find the best factorization of n_patches that matches target aspect ratio
        # Try different factorizations
        estimate_h = int(math.sqrt(n_patches/target_aspect_ratio))
        best_w = n_patches // estimate_h
        best_h = estimate_h
        for h in range(estimate_h-2, estimate_h+3):
            if n_patches % h == 0:
                best_w = n_patches // h
                best_h = h
                break
        assert best_h * best_w == n_patches, f"cannot find best grid dimensions: {n_patches} != {best_h} * {best_w}"
        
        # Cache the result
        self.grid_dimensions = {
            'key': cache_key,
            'value': (best_h, best_w)
        }
        
        return (best_h, best_w)

    

    