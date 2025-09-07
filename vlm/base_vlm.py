import os
import json
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from PIL import Image
import torch
from transformers import AutoTokenizer
import cv2

class LocalVLM(ABC):
    """Abstract base class for attention visualizers
    
    This class provides a common interface for different VLM models to extract
    and visualize attention patterns between text tokens and image patches.
    """
    
    def __init__(self, model_path: str, temperature: float = 0.1, max_tokens: int = 1024):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = True
        self.tokenizer = None
        self.model_loaded = False
        self.model = None
        self.processor = None
        
    def is_model_loaded(self) -> bool:
        return self.model_loaded and self.model is not None
    
    def ensure_model_loaded(self):
        if not self.is_model_loaded():
            print(f"Loading model from {self.model_path}...")
            self.load_model()
            self.model_loaded = True
            print("✓ Model loaded successfully")
        else:
            if self.verbose:
                print("✓ Model already loaded")
    
    def unload_model(self):
        if self.is_model_loaded():
            del self.model
            if hasattr(self, 'processor'):
                del self.processor
            self.model = None
            self.processor = None
            self.model_loaded = False
            torch.cuda.empty_cache()
            print("✓ Model unloaded")
    
    def reset_state(self):
        model = self.model if hasattr(self, 'model') else None
        processor = self.processor if hasattr(self, 'processor') else None
        tokenizer = self.tokenizer if hasattr(self, 'tokenizer') else None
        device = self.device
        model_loaded = self.model_loaded
        model_path = self.model_path
        model_type = getattr(self, 'model_type', None)
        verbose = self.verbose
        
        if hasattr(self, 'individual_step_attentions'):
            self.individual_step_attentions = []
        if hasattr(self, 'aggregated_attentions'):
            self.aggregated_attentions = []
        if hasattr(self, 'generated_tokens'):
            self.generated_tokens = []
        
        if hasattr(self, '_vision_token_positions'):
            self.vision_token_positions = None
        if hasattr(self, '_grid_dimensions'):
            self.grid_dimensions = None
        
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.model_loaded = model_loaded
        self.model_path = model_path
        if model_type is not None:
            self.model_type = model_type
        self.verbose = verbose
        
        if self.verbose:
            print("   ✓ Visualizer state reset (model and configuration preserved)")

    def tokenize_text(self, text: str) -> List[str]:
        """Convert input text to tokens"""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return tokens
    
    def get_input_text_token_attention_heatmap(self, input_tokens, attentions, layer_idx: int, token_idx: int, rm_spikes: bool = False) -> np.ndarray:
        layer_attention = attentions[layer_idx].cpu().float().numpy()
        layer_attention = layer_attention[0]
        avg_attention = layer_attention.mean(axis=0)
        
        seq_len = avg_attention.shape[0]
        positions = self.compute_vision_token_positions(input_tokens, seq_len)
        image_start = positions['image_start']
        image_end = positions['image_end']
        text_start = positions['text_start']
        text_end = positions['text_end']
        if 'image_break' in positions:  
            image_break = positions['image_break']
            avg_attention = np.delete(avg_attention, image_break, axis=1)

        if rm_spikes:
            all_layers = []
            for l in range(len(attentions)):
                la = attentions[l].cpu().float().numpy()
                if len(la.shape) == 4:
                    la = la[0].mean(axis=0)
                else:
                    continue
                all_layers.append(la[text_start:text_end, image_start:image_end])
            all_layers = np.stack(all_layers, axis=0)

            attn_per_layer = all_layers[:, token_idx-text_start, :]
            avg_attn = attn_per_layer.mean(axis=0)

            majority_ratio = 0.7
            num_layers = attn_per_layer.shape[0]
            high_attn_mask = attn_per_layer > attn_per_layer.mean(axis=1, keepdims=True)
            high_attn_count = high_attn_mask.sum(axis=0)
            remove_mask = high_attn_count >= int(num_layers * majority_ratio)
            avg_image_attention = avg_attn.copy()
            avg_image_attention[remove_mask] = 0
            token_to_image_attention = avg_attention[token_idx, image_start:image_end].copy()
            token_to_image_attention[remove_mask] = 0
        else:
            token_to_image_attention = avg_attention[token_idx, image_start:image_end]
        
        if len(token_to_image_attention) > 1 and token_to_image_attention.max() > token_to_image_attention.min():
            token_to_image_attention = (token_to_image_attention - token_to_image_attention.min()) / (token_to_image_attention.max() - token_to_image_attention.min())
        else:
            raise ValueError(f"attention is empty")
        
        return self.process_attention_to_heatmap(token_to_image_attention)

    def process_attention_to_heatmap(self, token_to_image_attention: np.ndarray) -> np.ndarray:
        if len(token_to_image_attention) > 1 and token_to_image_attention.max() > token_to_image_attention.min():
            image_attention = (token_to_image_attention - token_to_image_attention.min()) / (token_to_image_attention.max() - token_to_image_attention.min())
        else:
            print(f"    Warning: All attention values are identical, using uniform distribution")
            image_attention = np.ones_like(token_to_image_attention) / len(token_to_image_attention)
        
        if not hasattr(self, 'target_aspect_ratio'):
            raise ValueError("No target aspect ratio provided")
        
        # Find best grid dimensions for the compressed tokens
        best_h, best_w = self.find_best_grid_dimensions(len(image_attention), self.target_aspect_ratio)
        
        # Create attention grid
        attention_grid = np.zeros((best_h, best_w))
        
        # Fill the grid in row-major order
        for idx in range(len(image_attention)):
            row = idx // best_w
            col = idx % best_w
            if row < best_h and col < best_w:
                attention_grid[row, col] = image_attention[idx]
        
        # Apply Gaussian blur for smoother visualization
        attention_grid = cv2.GaussianBlur(attention_grid, (3, 3), 0)
        
        attention_grid = (attention_grid * 255).astype(np.uint8)
        
        return attention_grid
    
    def compute_text_image_similarity_matrix(self, input_tokens, attentions, layer_idx: int, temperature: float = 1.0) -> np.ndarray:
        """
        Compute similarity matrix between text tokens and image patches using Query-Key dot product.
        
        Args:
            layer_idx: Layer index to extract attention from
            temperature: Temperature parameter for softmax normalization
            
        Returns:
            Similarity matrix of shape [num_text_tokens, num_image_patches] after softmax
        """
        if layer_idx >= len(attentions):
            raise ValueError(f"Layer index {layer_idx} out of range. Available layers: 0-{len(attentions)-1}")
        
        layer_attention = attentions[layer_idx]
        
        if len(layer_attention.shape) == 4:
            attention_matrix = layer_attention[0].mean(axis=0).cpu().float().numpy()
        elif len(layer_attention.shape) == 3:
            attention_matrix = layer_attention.mean(axis=0).cpu().float().numpy()
        else:
            attention_matrix = layer_attention.cpu().float().numpy()
        
        seq_len = attention_matrix.shape[0]
        
        positions = self.compute_vision_token_positions(input_tokens, seq_len)
        
        text_start = positions.get('text_start')
        text_end = positions.get('text_end')
        image_start = positions.get('image_start')
        image_end = positions.get('image_end')
        
        if self.verbose:
            print(f"Computing similarity matrix for layer {layer_idx}")
            print(f"Text tokens: {text_start}:{text_end} ({text_end - text_start} tokens)")
            print(f"Image tokens: {image_start}:{image_end} ({image_end - image_start} tokens)")
        
        text_to_image_attention = attention_matrix[text_start:text_end, image_start:image_end]
        
        scaled_attention = text_to_image_attention / temperature
        
        similarity_matrix = self.apply_softmax(scaled_attention, axis=1)
        
        if self.verbose:
            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            print(f"Similarity range: [{similarity_matrix.min()}, {similarity_matrix.max()}]")
            print(f"Row sums (should be ~1.0): min={similarity_matrix.sum(axis=1).min()}, "
                  f"max={similarity_matrix.sum(axis=1).max()}")
        
        return similarity_matrix
    
    def apply_softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def prepare_inputs(self, image: Image.Image, text: str) -> dict:
        """Prepare inputs for model inference
        
        Args:
            image: Image URL or base64 string
            text: Input text/question
        
        Returns:
            Dictionary of model inputs
        """
        # Load and convert image
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        # Process inputs with processor using the new method
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                if key == "pixel_values" and self.device.type == "cuda":
                    inputs[key] = inputs[key].to(device=self.device, dtype=self.dtype)
                else:
                    inputs[key] = inputs[key].to(device=self.device)
        
        return inputs

    def extract_attention(self, inputs, max_new_tokens: int = 50, store_both: bool = False) -> dict:
        # Ensure model is loaded
        self.ensure_model_loaded()

        self.model.config.output_attentions = True
        
        with torch.no_grad():
            # Generate response with attention extraction
            generation_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=self.temperature
            )
        
        response_text = self.processor.decode(
            generation_outputs.sequences[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Get generated tokens for individual visualization
        generated_tokens = self.processor.tokenizer.convert_ids_to_tokens(
            generation_outputs.sequences[0][inputs['input_ids'].shape[1]:]
        )
        
        # Extract attentions from generation
        raw_attentions = generation_outputs.attentions
        
        self.individual_step_attentions = raw_attentions
        
        if raw_attentions and len(raw_attentions) > 0:
            print(f"Layers per step: {len(raw_attentions[0])}")
            
            input_stage_attentions = raw_attentions[0]
            
            if isinstance(input_stage_attentions, tuple):
                processed_attentions = list(input_stage_attentions)
            else:
                processed_attentions = [input_stage_attentions] if not isinstance(input_stage_attentions, list) else input_stage_attentions
            
            if store_both:
                self.aggregated_attentions = processed_attentions
        else:
            processed_attentions = []
            self.individual_step_attentions = []
            if store_both:
                self.aggregated_attentions = []
                
        self.generated_tokens = generated_tokens
        
        # Reset cached positions and grid dimensions when new input is processed
        self.vision_token_positions = None
        self.grid_dimensions = None
        
        print(f"Response: {response_text}")
        print(f"Raw attention steps: {len(raw_attentions)}")
        print(f"Processed attention layers: {len(processed_attentions)}")
        if processed_attentions:
            print(f"Attention tensor shape: {processed_attentions[0].shape}")
        
        return processed_attentions

    def get_input_text_tokens(self, input_tokens) -> Optional[Tuple[List[str], List[int]]]:
        # Use cached vision token positions
        try:
            positions = self.compute_vision_token_positions(input_tokens)
            text_start = positions['text_start']
            text_end = positions['text_end']
        except ValueError:
            return None
        
        # Get input text tokens (after vision_end)
        input_text_tokens = input_tokens[text_start:text_end]
        
        if not input_text_tokens:
            return None
        
        # Create position list
        token_positions = list(range(text_start, text_start + len(input_text_tokens)))
        
        return input_text_tokens, token_positions

    @abstractmethod
    def load_model(self):
        """Load the specific model - must set self.model_loaded = True when successful"""
        pass
    
    @abstractmethod
    def __call__(self, messages):
        """Handle model specific inference
        
        Args:
            messages: List of message dictionaries containing role and content
            
        Returns:
            Model response text
        """
        pass

    @abstractmethod
    def find_best_grid_dimensions(self, n_patches: int, target_aspect_ratio: float) -> Tuple[int, int]:
        """Find best grid dimensions for the model"""
        pass

    @abstractmethod
    def compute_vision_token_positions(self, input_tokens: List[str], seq_len: int) -> dict:
        """Compute vision token positions in the input sequence"""
        pass