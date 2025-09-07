from .base_clip import LocalCLIP
from transformers import AutoModel, AutoProcessor
import numpy as np


class SigLIP(LocalCLIP):
    def __init__(self, model_path):
        super().__init__(model_path)

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def extract_attention(self, image, question, layer_idx) -> dict:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=True,
            return_dict=True
        )
        attentions = outputs.attentions  # tuple, attention shape = (batch, heads, tokens, tokens)
        attn = attentions[layer_idx].detach()[0]  # (heads, N, N)
        patch_map = attn.mean(0)  # (tokens, tokens)
        side = int(np.sqrt(patch_map.shape[0]))
        # print(patch_map.shape, side, side*side)
        mask_arr = patch_map.reshape(side, side, side, side).mean(axis=(2,3)).cpu().numpy()
        mask_arr = (mask_arr-np.min(mask_arr))/(np.max(mask_arr)-np.min(mask_arr))
        mask_arr = (mask_arr * 255).astype(np.uint8)
        return mask_arr

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image

    image = Image.open("/media/data/lwm/GridGround/data/000000005862.jpg")
    size = 384
    image = image.resize((size, size))
    query = "brown dog"

    model_path = "google/siglip-so400m-patch14-384"
    # model_path = "google/siglip-base-patch16-224"
    model = SigLIP(model_path)
    model._load_model()
    mask_arr = model.extract_attention(image, query, layer_idx=11)
    mask_arr = cv2.resize(mask_arr, (size, size))
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask_arr)
    plt.savefig("attention_map.png")