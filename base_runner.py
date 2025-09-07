import json
import os
import time
from matplotlib import colorbar
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils import (
    encode_image,
    create_grid_overlay,
    create_axis_overlay,
    create_grid_mark,
)
from PIL import ImageDraw
import re
import traceback
from metric import extract_answer
from llm import AbstractLLM
from scaffold import dot_matrix_two_dimensional
import base64
import io
from PIL import Image
import uuid


class BaseRunner(ABC):
    """Base class for running different datasets with LLM evaluation"""

    def __init__(
        self,
        name,
        folder,
        save_path,
        settings,
        resume=True,
        rephrase=False,
        add_grid=False,
        grid_size=100,
        with_origin_chart=False,
        add_axis=False,
        axis_position="bottom_left",
        color="black",
        run_som=False,
        run_attention=False,
        rm_spikes=False,
        use_all_layers=False,
        alpha=1,
        extra_axis=False,
        extra_info=True,
        with_label=False,
        save_internal=False,
        resize=False,
        is_chart=False,
        use_scaffold=False,
        scaffold_mode='default',
        run_all=False,
        verbose=True,
        debug=False,
        **kwargs,
    ):
        self.name = name
        self.folder = folder
        self.save_path = save_path
        self.settings = settings
        self.resume = resume

        # Image processing parameters
        self.rephrase = rephrase
        self.add_grid = add_grid
        self.grid_size = grid_size
        self.with_origin_chart = with_origin_chart
        self.add_axis = add_axis
        self.axis_position = axis_position
        self.color = color
        self.run_som = run_som
        self.run_attention = run_attention
        self.rm_spikes = rm_spikes
        self.use_all_layers = use_all_layers
        self.alpha = alpha
        self.extra_axis = extra_axis
        self.extra_info = extra_info
        self.with_label = with_label
        self.save_internal = save_internal
        self.resize = resize
        self.is_chart = is_chart
        self.run_all = run_all
        self.use_scaffold = use_scaffold
        self.scaffold_mode = scaffold_mode
        self.verbose = verbose
        self.debug = debug

        # Store additional kwargs
        self.kwargs = kwargs

        self.llm = AbstractLLM(model_name=name)

        # Create save folder if not exists
        self.save_folder = os.path.dirname(save_path)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Load or initialize results
        if os.path.exists(save_path) and resume:
            self.res = json.load(open(save_path, "r", encoding="utf-8"))
        else:
            self.res = {"score": 0, "settings": settings, "data": {}}

        if self.grid_size is None:
            self.grid_size = self.llm.patch_size
         
        if self.debug:
            self.tmp_folder = "results/tmp"
            os.makedirs(self.tmp_folder, exist_ok=True)

    @abstractmethod
    def load_data(self):
        """Load dataset specific data"""
        pass

    @abstractmethod
    def get_prompt_template(self, **kwargs):
        """Get dataset specific prompt template"""
        pass

    @abstractmethod
    def extract_item_info(self, item):
        """Extract item specific information (id, gt, instruction, etc.)"""
        pass

    @abstractmethod
    def load_image(self, item):
        """Load image for the item"""
        pass

    @abstractmethod
    def process_prediction(self, pred, item):
        """Process LLM response to get prediction and score"""
        pass

    def prepare_image(self, img, item_id, item=None):
        """Prepare image with optional grid or axis overlay"""
        img_grid = None

        if self.add_grid:
            img_grid = create_grid_overlay(
                img, self.grid_size, self.with_label, self.color, self.alpha
            )
            if self.save_internal:
                tmp_path = f"{self.save_folder}/grid/{item_id}.png"
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                img_grid.save(tmp_path)
            if not self.with_origin_chart:
                img = img_grid

        if self.add_axis:
            img = create_axis_overlay(
                img, self.grid_size, self.axis_position, self.extra_axis
            )
            if self.save_internal:
                tmp_path = f"{self.save_folder}/axis/{item_id}.png"
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                img.save(tmp_path)
        
        if self.use_scaffold:
            img = dot_matrix_two_dimensional(img, scaffold_mode=self.scaffold_mode)

        return img, img_grid

    def build_messages(self, prompt, img, img_grid=None):
        """Build messages for LLM API call"""
        if self.use_scaffold:
            if self.scaffold_mode:
                text = "The image is overlaid with a dot matrix of a shape of 6 * 6 to help you with your task.\n 1. When you mention any key objects in the image, first output their nearest dots then identify them.\n 2. You use the dots to determine the spatial relationships of the objects.\n 3. You can search and reason region by region with the help of the dots."
            else:
                text = "The image is overlaid with a dot matrix of a shape of 6 * 6 to help you with your task, and each dot is labeled with two-dimensional coordinates (x,y).\n 1. When you mention any key objects in the image, first output their nearest coordinates then identify them.\n 2. You use the coordinates to determine the spatial relationships of the objects. Within each column, the x-coordinate increases from top to bottom, and within each row, the y-coordinate increases from left to right.\n 3. You can search and reason region by region with the help of the dots."
            messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": text}]}]
        else:
            messages = []

        if self.add_grid and self.with_origin_chart:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(img)
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(img_grid)
                            },
                        },
                    ],
                }
            ]
            if self.debug:
                img.show()
                img.save(f"{self.tmp_folder}/{uuid.uuid4()}.png")
                img_grid.show()
                img_grid.save(f"{self.tmp_folder}/{uuid.uuid4()}.png")
        else:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(img)
                            },
                        },
                    ],
                }
            ]
            if self.debug:
                img.show()
                img.save(f"{self.tmp_folder}/{uuid.uuid4()}.png")

        if self.debug:
            for msg in messages:
                print(msg["content"][0]["text"])
        return messages

    def update_prompt_for_overlays(self, prompt):
        """Update prompt based on overlay options"""
        if self.add_axis:
            prompt += "\nThe axis is shown in the image to assist you in determining precise positions."

        if self.add_grid:
            if self.with_origin_chart:
                prompt += f"\nYou'll see two versions: the original image, and a second version that has a {self.color} coordinate grid overlaid on top to assist you in determining precise positions."
            else:
                prompt += f"\nNote: a {self.color} coordinate grid overlay has been added to assist you in determining precise positions."

        return prompt

    def save_results(self):
        """Save results to file"""
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.res, f, ensure_ascii=False)
        print("save to: ", self.save_path)

    def postprocess_pred(self, response, item, meta=None):
        try:
            if self.is_chart:
                pred, score = self.process_prediction(response, item)
            else:
                pred = extract_answer(response, self.verbose)
                if meta is not None:
                    width, height, resize_ratio_width, resize_ratio_height = meta
                else:
                    img = self.load_image(item)
                    if self.resize:
                        img, resize_ratio_width, resize_ratio_height = (
                            self.llm.resize_image(img)
                        )
                    width, height = img.size

                if self.name.startswith("qwen2-vl") or self.name.startswith("minicpm"):
                    if len(pred) == 2:
                        pred = [pred[0] / 1000 * width,
                            pred[1] / 1000 * height]
                    else:
                        pred = [
                            pred[0] / 1000 * width,
                            pred[1] / 1000 * height,
                            pred[2] / 1000 * width,
                            pred[3] / 1000 * height,
                        ]
                elif (
                    self.name.startswith("qwen2.5-vl")
                    or self.name in ["qwen-vl-max", "uitars-1.5-7b", "uitars-7b"]
                ) and self.add_axis:
                    pred = [p - 100 for p in pred]  # rm shift

                if max(pred) <= 1 and max(pred) > 0:
                    if len(pred) == 2:
                        pred = [pred[0] * width, pred[1] * height]
                    elif len(pred) == 4:
                        pred = [
                            pred[0] * width,
                            pred[1] * height,
                            pred[2] * width,
                            pred[3] * height,
                        ]

                if self.resize:
                    if len(pred) == 2:
                        pred = [
                            int(pred[0] * resize_ratio_width),
                            int(pred[1] * resize_ratio_height),
                        ]
                    elif len(pred) == 4:
                        pred = [
                            int(pred[0] * resize_ratio_width),
                            int(pred[1] * resize_ratio_height),
                            int(pred[2] * resize_ratio_width),
                            int(pred[3] * resize_ratio_height),
                        ]
                score = self.process_prediction(pred, item)

        except Exception as e:
            print(f"Exception in postprocess_pred: {e}")
            pred = None
            score = None
        return pred, score
    
    def get_pred_from_marks(self, response, item, meta=None):
        img = self.load_image(item)
        if meta is not None:
            width, height, resize_ratio_width, resize_ratio_height = meta
        else:
            resize_ratio_width, resize_ratio_height = 1, 1
            if self.resize:
                img, resize_ratio_width, resize_ratio_height = (
                    self.llm.resize_image(img)
                )
        
        if self.use_all_layers: 
            for layer_index, layer_response in response.items():
                pred = layer_response['global_max_point']
                pred = [
                    int(pred[0] * resize_ratio_width),
                    int(pred[1] * resize_ratio_height),
                ]
                score = self.process_prediction(pred, item)
                if score == 1:
                    return pred, score
        else:
            # marks = response['marks']
            # if len(marks) > 0:
            #     max_attention = max(marks, key=lambda x: x["mean_attention"])
            #     pred = max_attention["center"]
            # else:
            #     pred = response['global_max_point']
            pred = response['global_max_point']
            pred = [
                int(pred[0] * resize_ratio_width),
                int(pred[1] * resize_ratio_height),
            ]
            score = self.process_prediction(pred, item)
        return pred, score

    def run(self):
        """Main run method"""
        data = self.load_data()
        count = 1
        fail = 0
        scores = []

        for item in tqdm(data):
            try:
                item_id, gt, filename = self.extract_item_info(item)
                # if item_id != "681e4e4159fd36c24b0de293":
                #     continue

                # Check if already processed
                if (
                    item_id in self.res["data"]
                    and self.res["data"][item_id]["pred"] is not None
                    and not self.debug
                ):
                    # score = self.res["data"][item_id]["score"]
                    response = self.res["data"][item_id]["pred"]
                    if type(response) == dict:
                        pred, score = self.get_pred_from_marks(response, item)
                    else:
                        pred, score = self.postprocess_pred(response, item)
                    if score == 1:
                        print("correct:", item_id)
                else:
                    # raise
                    print('*'*100)
                    print(f"ID: {item_id}")
                    img = self.load_image(item)
                    origin_width, origin_height = img.size
                    resize_ratio_width, resize_ratio_height = 1, 1
                    if self.resize:
                        img, resize_ratio_width, resize_ratio_height = (
                            self.llm.resize_image(img)
                        )
                    width, height = img.size
                    meta = (width, height, resize_ratio_width, resize_ratio_height)
                    img, img_grid = self.prepare_image(img, item_id, item=item)

                    # Get prompt and update for overlays
                    prompt, instruction = self.get_prompt_template(item=item, **self.kwargs)

                    if self.run_attention:
                        response = self.llm.get_token_attention_map(encode_image(img), prompt=prompt, visualize_text=instruction, rm_spikes=self.rm_spikes, use_all_layers=self.use_all_layers)
                        if self.debug:
                            print(response)

                        if response is None:
                            pred = None
                            score = None
                        else:
                            pred, score = self.get_pred_from_marks(response, item, meta)
                            if 'marked_image' in response:
                                if self.save_internal:
                                    marked_image = response['marked_image']
                                    img = Image.open(io.BytesIO(base64.b64decode(marked_image)))
                                    img = img.resize((origin_width, origin_height))
                                    tmp_img_path = f"{self.save_folder}/attention/{item_id}.png"
                                    os.makedirs(os.path.dirname(tmp_img_path), exist_ok=True)
                                    img.save(tmp_img_path)
                                del response['marked_image']
                    else:
                        if not self.add_axis and not self.run_som and self.extra_info:
                            prompt += f"\nThe width and height of the image are {width} and {height}. The origin coordinate of the image is top-left corner."
                        prompt = self.update_prompt_for_overlays(prompt)

                        # Build messages and call LLM
                        if self.run_som:
                            prompt = f"""A bright numeric ID at the center for each object is labeled in the image, please tell me the corresponding ID for the expression.\nThe output should be in the format: ```json\nID\n```, for example: ```json\n1\n```\n\nExpression: {instruction}"""
                        messages = self.build_messages(prompt, img, img_grid)

                        # Call LLM with built-in retry mechanism
                        # response = self.llm(messages)
                        # if response is None:
                        #     score = None
                        # else:
                        #     # Process prediction and calculate score
                        #     pred, score = self.postprocess_pred(response, item, meta)
                        try:
                            response = self.llm(messages)
                            if response is None:
                                score = None
                                pred = None
                            else:
                                pred, score = self.postprocess_pred(response, item, meta)
                        except Exception as e:
                            print(f"Error during LLM call or postprocessing: {e}")
                            pred = None
                            score = None
                    
                    tmp_mean_score = np.mean(scores+[score] if score is not None else scores+[0])
                    print(
                        f"id: {item_id}, gt: {gt}, pred: {pred}, score: {score}, mean_score: {tmp_mean_score:.2%}"
                    )
                    count += 1

            except Exception as e:
                traceback.print_exc()
                score = None
                # response = None
            
            if self.debug:
                return 0

            # Store results
            self.res["data"][item_id] = {"score": score, "gt": gt, "pred": response}

            if score is None:
                fail += 1
                score = 0
            scores.append(score)

            # Save periodically
            if count % 10 == 0:
                count += 1
                self.save_results()

        # Calculate final score and save
        mean_score = np.mean(scores)
        self.res["score"] = mean_score
        print(f"score: {mean_score:.2%}, fail: {fail}/{len(scores)}")
        self.save_results()
        return mean_score


class BaseGridMarkRunner(ABC):
    """Base class for grid mark functionality"""

    def __init__(
        self,
        name,
        folder,
        save_path,
        settings,
        resume=True,
        coord_type="center",
        num_grid=5,
        num_zone_in=1,
        color="black",
        with_origin_chart=False,
        enlarge=False,
        save_internal=False,
        resize=False,
        run_all=False,
        verbose=True,
        debug=False,
        **kwargs,
    ):
        self.name = name
        self.folder = folder
        self.save_path = save_path
        self.settings = settings
        self.resume = resume

        # Grid mark parameters
        self.coord_type = coord_type
        self.num_grid = num_grid
        self.num_zone_in = num_zone_in
        self.color = color
        self.with_origin_chart = with_origin_chart
        self.enlarge = enlarge
        self.save_internal = save_internal
        self.resize = resize
        self.run_all = run_all
        self.verbose = verbose
        self.debug = debug
        
        # Store additional kwargs
        self.kwargs = kwargs

        self.llm = AbstractLLM(model_name=name)

        # Create save folder if not exists
        self.save_folder = os.path.dirname(save_path)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Load or initialize results
        if os.path.exists(save_path) and resume:
            self.res = json.load(open(save_path, "r", encoding="utf-8"))
        else:
            self.res = {"score": 0, "settings": settings, "data": {}}

        if self.debug:
            self.tmp_folder = "results/tmp"
            os.makedirs(self.tmp_folder, exist_ok=True)

    @abstractmethod
    def load_data(self):
        """Load dataset specific data"""
        pass

    @abstractmethod
    def extract_item_info(self, item):
        """Extract item specific information (id, gt, instruction, etc.)"""
        pass

    @abstractmethod
    def load_image(self, item):
        """Load image for the item"""
        pass

    @abstractmethod
    def get_grid_mark_prompt_template(self, instruction):
        """Get dataset specific grid mark prompt template"""
        pass

    @abstractmethod
    def calculate_score(self, pred, item):
        """Calculate score for the prediction"""
        pass

    def save_results(self):
        """Save results to file"""
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.res, f, ensure_ascii=False)
        print("save to: ", self.save_path)

    def grid_mark_function(
        self, instruction, img_origin, img, save_path, gt, use_origin_chart, item
    ):
        """Unified grid mark function implementation"""
        img, id_coord = create_grid_mark(img, self.num_grid, self.enlarge, color=self.color)
        prompt_template = self.get_grid_mark_prompt_template(instruction)
        max_id = self.num_grid * self.num_grid - 1
        format_prompt = f"\nList exactly 4 grid IDs corresponding to the leftmost, topmost, rightmost and bottommost cells that contain the object (these IDs can be identical if the object fits in one cell).  The grid IDs should be between 0 and {max_id}.\nThe output should be in the format: ```json\n[id1, id2, ...]\n```, for example: ```json\n[3, 3, 4, 9]\n```."
        prompt_template += format_prompt
        if self.save_internal:
            img.save(save_path.replace(".png", "_grid.png"))

        if use_origin_chart:
            x1, y1, x2, y2 = gt
            print(gt)
            img_origin_mark = img_origin.copy()
            draw = ImageDraw.Draw(img_origin_mark)
            draw.rectangle([x1, y1, x2, y2], outline=self.color, width=3)
            if self.save_internal:
                img_origin_mark.save(save_path.replace(".png", "_origin.png"))

            prompt = (
                prompt_template
                + f"\nYou'll see two images: an overview image with a {self.color} box highlighting the relevant area, followed by a zoomed-in view of that area overlaid with a grid."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(img_origin_mark)
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(img)
                            },
                        },
                    ],
                }
            ]
            if self.debug:
                img_origin_mark.show()
                img_origin_mark.save(f"{self.tmp_folder}/{uuid.uuid4()}.png")
                img.show()
                img.save(f"{self.tmp_folder}/{uuid.uuid4()}.png")
        else:
            prompt = prompt_template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(img)
                            },
                        },
                    ],
                }
            ]
            if self.debug:
                img.show()
                img.save(f"{self.tmp_folder}/{uuid.uuid4()}.png")

        if self.debug:
            for msg in messages:
                print(msg["content"][0]["text"])

        # Call LLM with built-in retry mechanism
        response = self.llm(messages)
        if response is None:
            return None

        try:
            pred = extract_answer(response, self.verbose)
            if pred[0] == 46:
                pred = [46,46,47,47]
            pred = set(pred)
            cells = np.array([id_coord[p] for p in pred])
            # Find bounding box of all cells
            x1 = min(cells[:, 0])  # Leftmost x
            y1 = min(cells[:, 1])  # Topmost y
            x2 = max(cells[:, 2])  # Rightmost x
            y2 = max(cells[:, 3])  # Bottommost y
            return (x1, y1, x2, y2)
        except Exception as e:
            traceback.print_exc()
            return None

    def run_grid_mark(self):
        """Main grid mark run method"""
        data = self.load_data()
        count = 1
        fail = 0
        scores = []
        save_folder = f"{self.save_folder}/grid_mask"
        if self.save_internal and not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for item in tqdm(data):
            try:
                item_id, gt, instruction = self.extract_item_info(item)
                # if item_id != "681e4e4159fd36c24b0de293":
                #     continue

                if (
                    item_id in self.res["data"]
                    and self.res["data"][item_id]["pred"] is not None
                    and not self.debug
                ):
                    # score = self.res["data"][item_id]["score"]
                    pred = self.res["data"][item_id]["pred"]
                    score = self.calculate_score(pred, item)
                else:
                    for k in range(3):
                        img = self.load_image(item)
                        if self.resize:
                            img, resize_ratio_width, resize_ratio_height = (
                                self.llm.resize_image(img)
                            )
                        img_origin = img.copy()
                        width, height = img.size

                        save_roi_path = f"{save_folder}/{item_id}.png"
                        x1, y1, x2, y2 = 0, 0, width, height
                        success = True
                        for i in range(self.num_zone_in + 1):
                            if self.with_origin_chart and i > 0:
                                use_origin_chart = True
                            else:
                                use_origin_chart = False

                            roi = self.grid_mark_function(
                                instruction,
                                img_origin,
                                img,
                                save_roi_path,
                                (x1, y1, x2, y2),
                                use_origin_chart,
                                item,
                            )

                            if roi is None:
                                success = False
                                break

                            img = img.crop(roi)
                            x1, y1, x2, y2 = (
                                roi[0] + x1,
                                roi[1] + y1,
                                roi[2] + x1,
                                roi[3] + y1,
                            )

                        if success:
                            if self.resize:
                                x1, y1, x2, y2 = (
                                    int(x1 * resize_ratio_width),
                                    int(y1 * resize_ratio_height),
                                    int(x2 * resize_ratio_width),
                                    int(y2 * resize_ratio_height),
                                )
                            if self.coord_type == "center":
                                pred = ((x1 + x2) / 2, (y1 + y2) / 2)
                            elif self.coord_type == "bbox":
                                pred = (x1, y1, x2, y2)
                            score = self.calculate_score(pred, item)
                            break
                        else:
                            print(f'retry {k+2}')
                    if not success:
                        print(f"Failed!")
                        score = None
                        pred = None

                    tmp_mean_score = np.mean(scores+[score] if score is not None else scores+[0])
                    print(
                        f"idx: {item_id}, gt: {gt}, pred: {pred}, score: {score}, mean_score: {tmp_mean_score:.2%}"
                    )
                    count += 1

            except Exception as e:
                traceback.print_exc()
                score = None
            
            if self.debug:
                return 0

            self.res["data"][item_id] = {"score": score, "gt": gt, "pred": pred}

            if score is None:
                score = 0
                fail += 1
            scores.append(score)

            # Save periodically
            if count % 10 == 0:
                count += 1
                self.save_results()

        # Calculate final score and save
        mean_score = np.mean(scores)
        self.res["score"] = mean_score
        print(f"score: {mean_score:.2%}, fail: {fail}/{len(scores)}")
        self.save_results()
        return mean_score
