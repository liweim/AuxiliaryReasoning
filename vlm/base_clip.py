from abc import ABC, abstractmethod
import torch

class LocalCLIP(ABC):
    """Abstract base class for attention visualizers"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = True
        self.model_loaded = False
        self.model = None
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model_loaded and self.model is not None
    
    def ensure_model_loaded(self):
        """Ensure model is loaded, load if not already loaded"""
        if not self.is_model_loaded():
            print(f"Loading model from {self.model_path}...")
            self.load_model()
            self.model_loaded = True
            print("✓ Model loaded successfully")
        else:
            if self.verbose:
                print("✓ Model already loaded")
    
    @abstractmethod
    def load_model(self):
        """Load the specific model - must set self.model_loaded = True when successful"""
        pass
    
    @abstractmethod
    def extract_attention(self, image, question, layer_idx):
        pass