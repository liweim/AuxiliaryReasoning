import os
import json
from PIL import Image
import glob
import random
import jsonlines
import pandas as pd
import io
import uuid
import argparse

ROOT = 'dataset'

def prepare_dataset(name):
    res = []
    if name == "ScreenSpot":
        for path in glob.glob(f"{ROOT}/{name}/data/*.parquet"):
            df = pd.read_parquet(path)
            for i, row in df.iterrows():
                filename = row["file_name"]
                image_path = f"{ROOT}/{name}/images/{filename}"
                if not os.path.exists(image_path):
                    image = row["image"]["bytes"]
                    image = Image.open(io.BytesIO(image)).convert("RGB")
                    image.save(image_path)
                width, height = image.size
                bbox = row["bbox"]
                bbox = [
                    bbox[0] * width,
                    bbox[1] * height,
                    bbox[2] * width,
                    bbox[3] * height,
                ]

                res.append(
                    {
                        "id": str(uuid.uuid4()),
                        "filename": row["file_name"],
                        "instruction": row["instruction"],
                        "bbox": bbox,
                        "data_type": row["data_type"],
                        "data_source": row["data_source"],
                    }
                )
    elif name == "ScreenSpot-v2":
        path = f"{ROOT}/ScreenSpot-v2/samples.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data["samples"]:
            id = item["_id"]["$oid"]
            x1, y1, w, h = item["action_detection"]["bounding_box"]
            filename = item["filepath"]
            instruction = item["instruction"]
            width = item["metadata"]["width"]
            height = item["metadata"]["height"]
            bbox = (
                int(x1 * width),
                int(y1 * height),
                int((x1 + w) * width),
                int((y1 + h) * height),
            )
            res.append(
                {
                    "id": id,
                    "filename": filename,
                    "instruction": instruction,
                    "bbox": bbox,
                }
            )
    elif name == "ScreenSpot-Pro":
        for path in glob.glob(f"{ROOT}/ScreenSpot-Pro/annotations/*.json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                res.append(
                    {
                        "id": item["id"],
                        "filename": item["img_filename"],
                        "instruction": item["instruction"],
                        "instruction_cn": item["instruction_cn"],
                        "bbox": item["bbox"],
                        "application": item["application"],
                        "platform": item["platform"],
                        "ui_type": item["ui_type"],
                        "group": item["group"],
                    }
                )
    elif name == "I2E-Bench":
        path = f"{ROOT}/{name}/I2E-bench-annotation.jsonl"
        with jsonlines.open(path) as reader:
            for i, item in enumerate(reader):
                res.append(
                    {
                        "id": i,
                        "filename": item["image"],
                        "instruction": item["instruction"],
                        "bbox": item["bounding_box"],
                        "instr_type": item["annotations"]["instr_type"],
                        "el_type": item["el_type"],
                        "source": item["source"],
                    }
                )
    random.shuffle(res)
    with open(f"{ROOT}/{name}/meta.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ScreenSpot", choices=["ScreenSpot", "ScreenSpot-v2", "ScreenSpot-Pro", "I2E-Bench"])
    args = parser.parse_args()
    prepare_dataset(args.dataset)