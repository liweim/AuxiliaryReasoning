from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from .base_vlm import LocalVLM
from typing import List, Tuple
from utils import decode_image
import torch._dynamo
torch._dynamo.config.disable = True


class Gemma(LocalVLM):
    def __init__(self, model_path: str, temperature=0.1, max_tokens=1024):
        super().__init__(model_path, temperature, max_tokens)
        self.vision_token_positions = None
        self.grid_dimensions = None

    def load_model(self):
        if self.device.type == 'cuda':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            attn_implementation="eager",
            output_attentions=True,
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

    def __call__(self, messages):
        """Handle Gemma3 inference
        
        Args:
            messages: List of message dictionaries containing role and content
            
        Returns:
            Model response text
        """
        self.ensure_model_loaded()
        
        new_messages = []
        for message in messages:
            all_content = []
            for content in message['content']:
                if content['type'] == 'text':
                    all_content.append({"type": "text", "text": content['text']})
                else:
                    # For image content, use the PIL Image directly
                    image = decode_image(content['image_url']['url'])
                    all_content.append({"type": "image", "image": image})
            new_messages.append({'role': message['role'], 'content': all_content})
        
        inputs = self.processor.apply_chat_template(
            new_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device, dtype=self.dtype)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, temperature=self.temperature, max_new_tokens=self.max_tokens, do_sample=False)
            generation = generation[0][input_len:]

        output = self.processor.decode(generation, skip_special_tokens=True)
        return output

    def compute_vision_token_positions(self, input_tokens: List[str], actual_seq_len: int=None):
        """Compute vision token positions for Gemma3"""
        if self.vision_token_positions is not None:
            return self.vision_token_positions
    
        image_start = input_tokens.index("<start_of_image>")+1
        image_end = input_tokens.index("<end_of_image>")
        text_start = image_end + 2
        text_end = input_tokens[text_start:].index("<end_of_turn>") + text_start
        
        # Cache the result
        self.vision_token_positions = {
            'text_start': text_start,
            'text_end': text_end,
            'image_start': image_start,
            'image_end': image_end
        }
        
        if self.verbose:
            print(f"  Image tokens: {image_start}:{image_end}")
            print(f"  Text tokens: {text_start}:{text_end}")
        
        return self.vision_token_positions

    def find_best_grid_dimensions(self, n_patches: int, target_aspect_ratio: float) -> Tuple[int, int]:
        """
        Find best grid dimensions considering Qwen2-VL's 2x2 patch token merging.
        The n_patches represents compressed tokens after 2x2 merging.
        We calculate dimensions based on target aspect ratio and then expand.
        """
        assert n_patches == 256, "n_patches != 256"
        return (16, 16)

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize input text using LLaVA's tokenizer.
        
        LLaVA adds special tokens '' and ' ' at the beginning,
        which we need to remove for visualization purposes.
        """
        # Get tokens from base class implementation
        tokens = super().tokenize_text(text)
        
        # Remove special tokens from the beginning ('' and ' ')
        if tokens[0] == '<bos>':
            tokens = tokens[1:]
            
        return tokens
