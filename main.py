import os
import json
from PIL import Image
from metric import click_acc_metric
from base_runner import BaseRunner, BaseGridMarkRunner
from prepare_dataset import ROOT
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Default parameters configuration to avoid repetition
DEFAULT_PARAMS = {
    "resume": True,                    # Resume from previous checkpoint if available
    "add_grid": False,                 # Overlay grid lines on the image for visual guidance
    "grid_size": 100,                  # Size of grid lines in pixels
    "with_origin_chart": False,        # Include original chart/image alongside processed version
    "add_axis": False,                 # Add coordinate axis to the image
    "axis_position": "bottom_left",    # Position of the coordinate axis (bottom_left, bottom_right, top_left, top_right and all_sides)
    "color": "red",                    # Color for grid lines
    "use_scaffold": False,             # Enable scaffold prompting technique
    "scaffold_mode": "default",        # Scaffold mode: default, only_dots and coordinate
    "run_attention": False,            # Run attention mechanism analysis
    "use_all_layers": False,           # Use all model layers for analysis
    "run_all": False,                  # Process entire dataset instead of subset
    "verbose": True,                   # Enable verbose output logging
    "debug": False                     # Enable debug mode with additional information
}

# Default parameters for grid mark runner
DEFAULT_GRID_MARK_PARAMS = {
    "resume": True,                    # Resume from previous checkpoint if available
    "num_grid": 5,                     # Number of grid divisions per dimension (5x5 grid)
    "num_zone_in": 1,                  # Number of zone-in iterations for refinement
    "color": "red",                    # Color for grid marks and labels
    "with_origin_chart": False,        # Include original chart/image alongside grid-marked version
    "enlarge": False,                  # Enlarge the zone-in area for better visibility
    "run_all": False,                  # Process entire dataset instead of subset
    "verbose": True,                   # Enable verbose output logging
    "debug": False                     # Enable debug mode with additional information
}

def merge_params(default_params, **kwargs):
    """Helper function to merge default parameters with provided kwargs"""
    params = default_params.copy()
    params.update(kwargs)
    return params

class ScreenSpotRunner(BaseRunner):
    """Runner for ScreenSpot dataset"""

    def __init__(
        self,
        name,
        folder,
        save_path,
        settings,
        **kwargs,
    ):
        super().__init__(
            name,
            folder,
            save_path,
            settings,
            **merge_params(DEFAULT_PARAMS, **kwargs),
        )

    def load_data(self):
        """Load ScreenSpot dataset"""
        data_path = f"{self.folder}/meta.json"
        with open(data_path, "r", encoding="utf-8") as f:
            if self.run_all:
                data = json.load(f)  # Load entire dataset
                # data = data[:len(data)//10]
            else:
                data = json.load(f)[:100]  # Load only first 100 samples for testing
        return data

    def get_prompt_template(self, **kwargs):
        """Get ScreenSpot specific prompt template"""
        item = kwargs.get("item")
        instruction = item["instruction"]
        prompt = f"""In the image, where should I click if I want to {instruction}?\nThe output coordinate should be in the format: ```json\n(x, y)\n```, for example: ```json\n(100, 100)\n```."""
        
        visualize_text = f" {instruction}"
        return prompt, visualize_text

    def extract_item_info(self, item):
        """Extract ScreenSpot item information"""
        idx = str(item["id"])
        bbox = item["bbox"]
        filename = item["filename"]
        return idx, bbox, filename

    def load_image(self, item):
        """Load ScreenSpot image"""
        filename = item["filename"]
        img_path = f"{self.folder}/images/{filename}"
        # Fallback to root folder if image not found in images subfolder
        if not os.path.exists(img_path):
            img_path = f"{self.folder}/{filename}"
        return Image.open(img_path).convert("RGB")

    def process_prediction(self, pred, item):
        """Process ScreenSpot prediction"""
        bbox = item["bbox"]
        score = click_acc_metric(bbox, pred)
        return score

    def get_class_score(self):
        qa_data = self.load_data()
        get_class_score(qa_data, self.save_path)


class ScreenSpotGridMarkRunner(BaseGridMarkRunner):
    """Grid mark runner for ScreenSpot dataset"""

    def __init__(
        self,
        name,
        folder,
        save_path,
        settings,
        **kwargs,
    ):
        super().__init__(
            name,
            folder,
            save_path,
            settings,
            **merge_params(DEFAULT_GRID_MARK_PARAMS, **kwargs),
        )

    def load_data(self):
        """Load ScreenSpot dataset"""
        with open(f"{self.folder}/meta.json", "r", encoding="utf-8") as f:
            if self.run_all:
                data = json.load(f)
                # data = data[:len(data)//10]
            else:
                data = json.load(f)[:100]
        return data

    def extract_item_info(self, item):
        """Extract ScreenSpot item information for grid mark"""
        idx = str(item["id"])
        bbox = item["bbox"]
        instruction = item["instruction"]
        return idx, bbox, instruction

    def load_image(self, item):
        """Load ScreenSpot image"""
        filename = item["filename"]
        img_path = f"{self.folder}/images/{filename}"
        if not os.path.exists(img_path):
            img_path = f"{self.folder}/{filename}"
        return Image.open(img_path).convert("RGB")

    def get_grid_mark_prompt_template(self, instruction):
        """Get ScreenSpot specific grid mark prompt template"""
        return f"A UI img is overlaid with a labeled grid in red (each cell marked with its ID at the center in red), determine which grid cell(s) contain the UI element needed to complete the following instruction: {instruction}."

    def calculate_score(self, pred, item):
        """Calculate score for ScreenSpot"""
        bbox = item["bbox"]
        return click_acc_metric(bbox, pred)

    def get_class_score(self):
        qa_data = self.load_data()
        get_class_score(qa_data, self.save_path)


def get_class_score(qa_data, save_path):
    """Calculate and display accuracy scores by device class (mobile, pc, web)"""
    all_cls = ["mobile", "pc", "web"]
    answer_data = json.load(open(save_path, "r", encoding="utf-8"))["data"]
    
    # Map each item ID to its device class based on filename prefix
    id_cls = {}
    for qa in qa_data:
        filename = qa["filename"]
        cls = filename.split("_")[0]  # Extract device class from filename
        id_cls[qa["id"]] = cls

    # Group scores by device class
    cls_score = {cls: [] for cls in all_cls}
    for id, item in answer_data.items():
        if id not in id_cls:
            continue
        cls = id_cls[id]
        score = item["score"] if item["score"] is not None else 0
        cls_score[cls].append(score)
    
    # Calculate and print mean accuracy for each class
    for cls in cls_score:
        num = len(cls_score[cls])
        mean_score = sum(cls_score[cls]) / num
        print(f"{cls}/{num}: {mean_score:.2%}")


def run(
    name,
    folder,
    save_path,
    settings,
    **kwargs,
):
    """Wrapper function to maintain compatibility"""
    runner = ScreenSpotRunner(
        name=name,
        folder=folder,
        save_path=save_path,
        settings=settings,
        **kwargs,
    )
    return runner.run()
    # runner.get_class_score()


def run_grid_mark(
    name,
    folder,
    save_path,
    settings,
    **kwargs,
):
    """Wrapper function using BaseGridMarkRunner"""
    runner = ScreenSpotGridMarkRunner(
        name,
        folder,
        save_path,
        settings,
        **kwargs,
    )
    return runner.run_grid_mark()
    # runner.get_class_score()


def run_all(dataset, models, methods, debug=False):
    """Run evaluation on all specified models and methods"""
    run_all = True
    folder = f"{ROOT}/{dataset}"
    save_folder = f"results/{dataset}"
    print("running on dataset:", folder)

    all_results = []
    for name in models:
        if name in ["gemma-3-12b", "gemma-3-4b", "siglip-400m"]:
            resize = True
        else:
            resize = False
        for method in methods:
            if method == "Direct Prediction":
                save_path = rf"{save_folder}/{name}_direct_prediction.json"
                settings = f"{name}, resize: {resize}"
                result = run(name, folder, save_path, settings, resume=True, resize=resize, run_all=run_all, debug=debug)
                all_results.append([save_path, result])

            elif method == "Grid-Augmented Vision":
                save_path = rf"{save_folder}/{name}_grid_augmented_vision.json"
                settings = f"{name}, baseline: 9x9 gray, 0.3 alpha, resize: {resize}"
                result = run(name, folder, save_path, settings, resume=True, add_grid=True, grid_size=1/9, alpha=0.3, color="gray", resize=resize, run_all=run_all, debug=debug)
                all_results.append([save_path, result])

            elif method == "Scaffold Prompting":
                save_path = rf"{save_folder}/{name}_scaffold_prompting.json"
                settings = f"{name}, use scaffold, resize: {resize}"
                result = run(name, folder, save_path, settings, resume=True, use_scaffold=True, resize=resize, run_all=run_all, debug=debug)
                all_results.append([save_path, result])

            elif method == "Coordinate Scaffold":
                save_path = rf"{save_folder}/{name}_coordinate_scaffold.json"
                settings = f"{name}, use scaffold, coordinate, resize: {resize}"
                result = run(name, folder, save_path, settings, resume=True, use_scaffold=True, scaffold_mode='coordinate', resize=resize, run_all=run_all, debug=debug)
                all_results.append([save_path, result])

            elif method == "Axis-Grid Scaffold":
                save_path = rf"{save_folder}/{name}_axis_grid_scaffold.json"
                settings = f"{name}, add axis on all sides, with grid in red, resize: {resize}"
                result = run(
                    name,
                    folder,
                    save_path,
                    settings,
                    resume=True,
                    add_axis=True,
                    axis_position="all_sides",
                    add_grid=True,
                    color="red",
                    resize=resize,
                    run_all=run_all,
                    debug=debug
                )
                all_results.append([save_path, result])

            elif method == "Mark-Grid Scaffold":
                save_path = rf"{save_folder}/{name}_mark_grid_scaffold.json"
                settings = f"{name} with 8 grid with mark, zone in 1 time, with origin chart, enlarge zone-in image, resize: {resize}"
                result = run_grid_mark(
                    name,
                    folder,
                    save_path,
                    settings,
                    resume=True,
                    num_grid=8,
                    num_zone_in=1,
                    color="red",
                    with_origin_chart=True,
                    enlarge=True,
                    resize=resize,
                    run_all=run_all,
                    debug=debug
                )
                all_results.append([save_path, result])

    print("*" * 100)
    for path, score in all_results:
        print(f"{path}: {score:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ScreenSpot-v2", choices=["ScreenSpot", "ScreenSpot-v2", "ScreenSpot-Pro", "UI-I2E-Bench"], help="Dataset to use for evaluation")
    parser.add_argument("--models", nargs='+', default=["gemini-2.5-flash"], choices=["gemini-2.5-flash", "gpt-4o", "claude-3.5", "qwen2-vl-7b", "gemma-3-12b", "gemma-3-4b", "siglip-400m"], help="List of models to evaluate")
    parser.add_argument("--methods", nargs='+', default=["Direct Prediction", "Grid-Augmented Vision", "Scaffold Prompting", "Coordinate Scaffold", "Axis-Grid Scaffold", "Mark-Grid Scaffold"], choices=["Direct Prediction", "Grid-Augmented Vision", "Scaffold Prompting", "Coordinate Scaffold", "Axis-Grid Scaffold", "Mark-Grid Scaffold"], help="List of reasoning methods to test")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    run_all(dataset=args.dataset, models=args.models, methods=args.methods, debug=args.debug)