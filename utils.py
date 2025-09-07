import math
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io
import base64
import json


def json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON serializable format"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [json_serializable(item) for item in obj]
    else:
        return obj


def safe_json_dump(data, file_path, **kwargs):
    """Safely dump data to JSON file, handling numpy arrays and other non-serializable types"""
    serializable_data = json_serializable(data)
    with open(file_path, 'w', **kwargs) as f:
        json.dump(serializable_data, f)


def encode_image(image: Image.Image):
    # Save image to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # Encode to base64
    return f"data:image/jpeg;base64,{img_base64}"


def decode_image(base64_string: str) -> Image.Image:
    """Convert base64 encoded image string back to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string (with or without data URL prefix)
    
    Returns:
        PIL Image object
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        # Extract base64 part after comma
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    return image


def draw_idx(draw, idx, x1, y1, x2, y2, font):
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    text_position = (x1, y2)  # Adjust Y to be above the bottom right
    text_bbox = draw.textbbox(text_position, str(idx), font=font, anchor="lb")
    draw.rectangle(text_bbox, fill="black")
    draw.text(text_position, str(idx), font=font, anchor="lb", fill="white")


def convert_to_text(num, high_precision=False):
    if high_precision:
        text1 = str(eval(f"{num}"))
        text2 = f"{num:.6e}"
    else:
        text1 = str(eval(f"{num:.2f}"))
        text2 = f"{num:.3e}"
    if len(text1) < len(text2):
        return text1
    else:
        return text2


def draw_grid(image, x1, y1, x2, y2, text, is_normal_chart, grid_size, color="black"):
    # Draw a red line
    h, w = image.shape[:2]
    # Convert cv2 image to PIL Image for text operations
    # if color == "gray":
    #     if is_normal_chart:
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         max_value = np.max(gray)
    #         mask = np.ones(gray.shape)
    #         cv2.line(mask, (x1, y1), (x2, y2), 0.7, 2)
    #         mask[np.where(gray < max_value - 10)] = 1
    #         mask = np.stack([mask, mask, mask], axis=2)
    #         image = (image * mask).astype(np.uint8)
    #     else:
    #         cv2.line(image, (x1, y1), (x2, y2), (180, 180, 180), 2)
    # else:
    #     cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.line([x1, y1, x2, y2], fill=color, width=2)

    # Add x coordinate text with black background
    text_x = x1
    text_y = y2
    if y1 == y2:
        font_size = int(grid_size * 0.5)
    else:
        font_size = int(grid_size * 0.3)

    font = ImageFont.truetype("arial.ttf", font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Adjust font size if text too wide
    if y1 != y2 and text_width > grid_size * 0.7:
        scale = grid_size * 0.7 / text_width
        font_size = int(font_size * scale)
        font = ImageFont.truetype("arial.ttf", font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

    # Adjust text position
    if text_x + text_width > w:
        text_x = w - text_width
    else:
        text_y = text_y - text_height

    # Get actual text bbox at the target position
    # Get text bbox and adjust position if outside image bounds
    # Get initial text bbox
    margin = 5
    bbox = draw.textbbox((text_x, text_y), text, font=font)
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    if text_x < 0:
        text_x = 0
    elif text_x + bbox_w > w - margin:
        text_x = w - margin - bbox_w

    if text_y < 0:
        text_y = 0
    elif text_y + bbox_h > h - margin:
        text_y = h - margin - bbox_h
    actual_bbox = draw.textbbox((text_x, text_y), text, font=font)

    # Draw black background rectangle aligned with actual text position
    margin = 2
    draw.rectangle(
        [
            actual_bbox[0] - margin,
            actual_bbox[1] - margin,
            actual_bbox[2] + margin,
            actual_bbox[3] + margin,
        ],
        fill="black",
    )

    # Draw text
    if color == "gray":
        text_color = "white"
    else:
        text_color = color
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    # Convert back to cv2 image
    image = np.array(pil_image)
    return image


def enlarge_grid(
    img,
    x1,
    y1,
    x2,
    y2,
    axis_min,
    axis_max,
    new_axis_min,
    new_axis_max,
    is_x_axis,
    num_grid,
    is_normal_chart,
):
    fix_size = 512
    axis_range = axis_max - axis_min
    w = x2 - x1
    h = y2 - y1
    if is_x_axis:
        new_x1 = int(x1 + (new_axis_min - axis_min) * w / axis_range)
        new_x2 = int(x1 + (new_axis_max - axis_min) * w / axis_range)
        new_y1 = y1
        new_y2 = y2
    else:
        new_x1 = x1
        new_x2 = x2
        new_y1 = int(y1 + (new_axis_min - axis_min) * h / axis_range)
        new_y2 = int(y1 + (new_axis_max - axis_min) * h / axis_range)
    assert (
        new_x1 >= 0
        and new_x2 <= img.shape[1]
        and new_y1 >= 0
        and new_y2 <= img.shape[0]
        and new_x1 < new_x2
        and new_y1 < new_y2
    )
    roi = img[new_y1:new_y2, new_x1:new_x2]
    new_img = cv2.resize(roi, (fix_size, fix_size))

    # Calculate step size for data values
    data_step = (new_axis_max - new_axis_min) / num_grid
    grid_size = fix_size / num_grid
    if is_x_axis:
        for i in range(num_grid):
            # Calculate data value for this grid line
            data_value = new_axis_min + i * data_step

            # Convert data value to pixel coordinate
            x = min(
                int(
                    (data_value - new_axis_min)
                    * fix_size
                    / (new_axis_max - new_axis_min)
                ),
                fix_size - 1,
            )
            text = convert_to_text(data_value, high_precision=True)
            new_img = draw_grid(
                new_img, x, 0, x, fix_size, text, is_normal_chart, grid_size
            )
    else:
        for i in range(1, num_grid + 1):
            # Calculate data value for this grid line
            data_value = new_axis_min + i * data_step
            # Convert data value to pixel coordinate
            y = fix_size - min(
                int(
                    (data_value - new_axis_min)
                    * fix_size
                    / (new_axis_max - new_axis_min)
                ),
                fix_size - 1,
            )
            text = convert_to_text(data_value, high_precision=True)
            new_img = draw_grid(
                new_img, 0, y, fix_size, y, text, is_normal_chart, grid_size
            )
    return new_img


def add_axis_text(img, num_grid, is_x_axis, bbox, chart_bbox, minmax):
    border = 100
    width, height = img.size
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = chart_bbox
    axis_min, axis_max = minmax
    axis_img = Image.new(
        "RGB", (width + border * 2, height + border * 2), (255, 255, 255)
    )
    axis_img.paste(img, (border, border))
    draw = ImageDraw.Draw(axis_img)

    font = ImageFont.truetype("arial.ttf", 30)
    if is_x_axis:
        draw.line(
            [0, height + border, width + border * 2, height + border],
            fill="black",
            width=3,
        )
        axis_x1 = (x1 - cx1) / (cx2 - cx1) * (axis_max - axis_min) + axis_min
        axis_x2 = (x2 - cx1) / (cx2 - cx1) * (axis_max - axis_min) + axis_min
        for i in range(num_grid + 1):
            axis_x = axis_x1 + i * (axis_x2 - axis_x1) / num_grid
            x = width / num_grid * i + border
            text = convert_to_text(axis_x)
            draw.text((x, height + border), text, fill="black", font=font)
            draw.line(
                [x, height + border - 10, x, height + border], fill="black", width=3
            )
    else:
        draw.line([border, 0, border, height + border * 2], fill="black", width=3)
        axis_y1 = (cy2 - y1) / (cy2 - cy1) * (axis_max - axis_min) + axis_min
        axis_y2 = (cy2 - y2) / (cy2 - cy1) * (axis_max - axis_min) + axis_min
        for i in range(num_grid + 1):
            axis_y = axis_y1 + i * (axis_y2 - axis_y1) / num_grid
            y = height / num_grid * i + border
            text = convert_to_text(axis_y)
            text_bbox = draw.textbbox((border, y), text, font=font, anchor="lb")
            text_height = text_bbox[2] - text_bbox[0]
            draw.text(
                (max(0, border - text_height - 10), y), text, fill="black", font=font
            )
            draw.line([border, y, border + 10, y], fill="black", width=3)
    return axis_img


def create_grid_overlay(img, grid_size=100, with_label=True, color="black", alpha=1):
    """
    Create a grid overlay on the image with coordinates

    Args:
        image_path: Path to the original img
        grid_size: Size of grid cells in pixels

    Returns:
        PIL.Image: Image with grid overlay
    """
    # Load the original image
    width, height = img.size

    # Create a copy for drawing
    img_add_grid = img.copy()
    draw = ImageDraw.Draw(img_add_grid)

    if grid_size < 1:
        num = int(1 / grid_size)
        # Draw grid lines
        for i in range(num):
            x = int(width / num * i)
            draw.line([(x, 0), (x, height)], fill=color, width=2)
        for i in range(num):
            y = int(height / num * i)
            draw.line([(0, y), (width, y)], fill=color, width=2)

        # Blend original image with grid
        img_add_grid = Image.blend(img, img_add_grid, alpha)
    else:
        font = ImageFont.truetype("arial.ttf", 20)

        # Draw vertical lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=color, width=2)
            if with_label:
                # Add x coordinate text with black background
                text = str(x)
                text_bbox = draw.textbbox((x + 5, 5), text, font=font)
                draw.rectangle(
                    [
                        text_bbox[0] - 2,
                        text_bbox[1] - 2,
                        text_bbox[2] + 2,
                        text_bbox[3] + 2,
                    ],
                    fill="black",
                )
                draw.text((x + 5, 5), text, fill=color, font=font)

        # Draw horizontal lines
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=color, width=2)
            if with_label:
                # Add y coordinate text with black background
                text = str(y)
                text_bbox = draw.textbbox((5, y + 5), text, font=font)
                draw.rectangle(
                    [
                        text_bbox[0] - 2,
                        text_bbox[1] - 2,
                        text_bbox[2] + 2,
                        text_bbox[3] + 2,
                    ],
                    fill="black",
                )
                draw.text((5, y + 5), text, fill=color, font=font)

    return img_add_grid


def create_axis_overlay(
    img, grid_size=100, axis_position="bottom_left", extra_axis=False
):
    border = 100
    if grid_size == 50:
        fontsize = 15
    else:
        fontsize = 30
    font = ImageFont.truetype("arial.ttf", fontsize)

    # Load the original image
    width, height = img.size

    if grid_size < 1:
        xgrid_size = int(width * grid_size)
        ygrid_size = int(height * grid_size)
        xrange = list(range(0, width, xgrid_size))
        yrange = list(range(0, height, ygrid_size))
    else:
        # grid_size = (border//grid_size)*grid_size
        xrange = list(range(0, width, grid_size)) + [width]
        yrange = list(range(0, height, grid_size)) + [height]
        xgrid_size = grid_size
        ygrid_size = grid_size

    if axis_position == "bottom_right":
        # Create a copy for drawing
        axis_img = Image.new("RGB", (width + border*2, height + border*2), (255, 255, 255))
        axis_img.paste(img, (border, border))
        draw = ImageDraw.Draw(axis_img)

        # Draw vertical lines
        for x in xrange:
            # Add x coordinate text with black background
            draw.line(
                [x + border, height + border, x + border, height + border + 20],
                fill="black",
                width=3,
            )
            if extra_axis and x < width - xgrid_size//2:
                for i in range(1, 5):
                    nx = x + (xrange[1] - xrange[0]) / 5 * i
                    draw.line(
                        [nx + border, height + border, nx + border, height + border + 10],
                        fill="black",
                        width=3,
                    )
            text = str(x)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if x == width:
                x += text_width // 2
            draw.text(
                (x + border - text_width // 2, height + border + 20),
                text,
                fill="black",
                font=font,
            )

        # Draw horizontal lines
        for y in yrange:
            # Add y coordinate text with black background
            draw.line(
                [width + border, y + border, width + border + 20, y + border],
                fill="black",
                width=3,
            )
            if extra_axis and y < height - ygrid_size//2:
                for i in range(1, 5):
                    ny = y + (yrange[1] - yrange[0]) / 5 * i
                    draw.line(
                        [width + border, ny + border, width + border + 10, ny + border],
                        fill="black",
                        width=3,
                    )
            text = str(y)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if y == height:
                y += text_height // 2
            draw.text(
                (width + border + 20, y + border - text_height // 2),
                text,
                fill="black",
                font=font,
            )
    elif axis_position == "top_left":
        # Create a copy for drawing
        axis_img = Image.new(
            "RGB", (width + border * 2, height + border * 2), (255, 255, 255)
        )
        axis_img.paste(img, (border, border))
        draw = ImageDraw.Draw(axis_img)

        # Draw vertical lines
        for x in xrange:
            # Add x coordinate text with black background
            draw.line(
                [x + border, border - 20, x + border, border],
                fill="black",
                width=3,
            )
            if extra_axis and x < width - xgrid_size//2:
                for i in range(1, 5):
                    nx = x + (xrange[1] - xrange[0]) / 5 * i
                    draw.line(
                        [nx + border, border - 10, nx + border, border],
                        fill="black",
                        width=3,
                    )
            text = str(x)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if x == width:
                x += text_width // 2
            draw.text(
                (x + border - text_width // 2, border - text_height - 30),
                text,
                fill="black",
                font=font,
            )

        # Draw horizontal lines
        for y in yrange:
            # Add y coordinate text with black background
            draw.line(
                [border - 20, y + border, border, y + border],
                fill="black",
                width=3,
            )
            if extra_axis and y < height - ygrid_size//2:
                for i in range(1, 5):
                    ny = y + (yrange[1] - yrange[0]) / 5 * i
                    draw.line(
                        [border - 10, ny + border, border, ny + border],
                        fill="black",
                        width=3,
                    )
            text = str(y)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if y == height:
                y += text_height // 2
            draw.text(
                (border - text_width - 20, y + border - text_height // 2),
                text,
                fill="black",
                font=font,
            )
    elif axis_position == "top_right":
        # Create a copy for drawing
        axis_img = Image.new(
            "RGB", (width + border * 2, height + border * 2), (255, 255, 255)
        )
        axis_img.paste(img, (border, border))
        draw = ImageDraw.Draw(axis_img)

        # Draw vertical lines
        for x in xrange:
            # Add x coordinate text with black background
            draw.line(
                [x + border, border - 20, x + border, border],
                fill="black",
                width=3,
            )
            if extra_axis and x < width - xgrid_size//2:
                for i in range(1, 5):
                    nx = x + (xrange[1] - xrange[0]) / 5 * i
                    draw.line(
                        [nx + border, border - 10, nx + border, border],
                        fill="black",
                        width=3,
                    )
            text = str(x)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if x == width:
                x += text_width // 2
            draw.text(
                (x + border - text_width // 2, border - text_height - 30),
                text,
                fill="black",
                font=font,
            )

        # Draw horizontal lines
        for y in yrange:
            # Add y coordinate text with black background
            draw.line(
                [width+border, y+border, width + border + 20, y+border],
                fill="black",
                width=3,
            )
            if extra_axis and y < height - ygrid_size//2:
                for i in range(1, 5):
                    ny = y + (yrange[1] - yrange[0]) / 5 * i
                    draw.line(
                        [width+border, ny+border, width + border + 10, ny+border],
                        fill="black",
                        width=3,
                    )
            text = str(y)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if y == height:
                y += text_height // 2
            draw.text(
                (width + border + 20, y + border - text_height // 2),
                text,
                fill="black",
                font=font,
            )
    elif axis_position == "bottom_left":
        # Create a copy for drawing
        axis_img = Image.new(
            "RGB", (width + border * 2, height + border * 2), (255, 255, 255)
        )
        axis_img.paste(img, (border, border))
        draw = ImageDraw.Draw(axis_img)

        # Draw vertical lines
        for x in xrange:
            # Add x coordinate text with black background
            draw.line(
                [x + border, height + border, x + border, height + border + 20],
                fill="black",
                width=3,
            )
            if extra_axis and x < width - xgrid_size//2:
                for i in range(1, 5):
                    nx = x + (xrange[1] - xrange[0]) / 5 * i
                    draw.line(
                        [
                            nx + border,
                            height + border,
                            nx + border,
                            height + border + 10,
                        ],
                        fill="black",
                        width=3,
                    )
            text = str(x)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if x == width:
                x += text_width // 2
            draw.text(
                (x + border - text_width // 2, height + border + 20),
                text,
                fill="black",
                font=font,
            )

        # Draw horizontal lines
        for y in yrange:
            # Add y coordinate text with black background
            draw.line(
                [border - 20, y + border, border, y + border],
                fill="black",
                width=3,
            )
            if extra_axis and y < height - ygrid_size//2:
                for i in range(1, 5):
                    ny = y + (yrange[1] - yrange[0]) / 5 * i
                    draw.line(
                        [border - 10, ny + border, border, ny + border],
                        fill="black",
                        width=3,
                    )
            text = str(y)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if y == height:
                y += text_height // 2
            draw.text(
                (border - text_width - 20, y + border - text_height // 2),
                text,
                fill="black",
                font=font,
            )
    elif axis_position == "all_sides":
        # Create a copy for drawing
        axis_img = Image.new(
            "RGB", (width + border * 2, height + border * 2), (255, 255, 255)
        )
        axis_img.paste(img, (border, border))
        draw = ImageDraw.Draw(axis_img)

        # Draw vertical lines
        for x in xrange:
            # Add x coordinate text with black background
            draw.line(
                [x + border, height + border, x + border, height + border + 20],
                fill="black",
                width=3,
            )
            draw.line(
                [x + border, border - 20, x + border, border],
                fill="black",
                width=3,
            )
            if extra_axis and x < width - xgrid_size//2:
                for i in range(1, 5):
                    nx = x + (xrange[1] - xrange[0]) / 5 * i
                    draw.line(
                        [
                            nx + border,
                            height + border,
                            nx + border,
                            height + border + 10,
                        ],
                        fill="black",
                        width=3,
                    )
                    draw.line(
                        [nx + border, border - 10, nx + border, border],
                        fill="black",
                        width=3,
                    )
            text = str(x)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if x == width:
                x += text_width // 2
            draw.text(
                (x + border - text_width // 2, height + border + 20),
                text,
                fill="black",
                font=font,
            )
            draw.text(
                (x + border - text_width // 2, border - text_height - 30),
                text,
                fill="black",
                font=font,
            )

        # Draw horizontal lines
        for y in yrange:
            # Add y coordinate text with black background
            draw.line(
                [border - 20, y + border, border, y + border],
                fill="black",
                width=3,
            )
            draw.line(
                [border + width, border + y, border + width + 20, border + y],
                fill="black",
                width=3,
            )
            if extra_axis and y < height - ygrid_size//2:
                for i in range(1, 5):
                    ny = y + (yrange[1] - yrange[0]) / 5 * i
                    draw.line(
                        [border - 10, ny + border, border, ny + border],
                        fill="black",
                        width=3,
                    )
                    draw.line(
                        [border + width, ny + border, border + width + 10, ny + border],
                        fill="black",
                        width=3,
                    )
            text = str(y)
            text_bbox = draw.textbbox((0, 0), text, font=font, anchor="lb")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if y == height:
                y += text_height // 2
            draw.text(
                (border - text_width - 20, y + border - text_height // 2),
                text,
                fill="black",
                font=font,
            )
            draw.text(
                (border + width + 20, border + y - text_height // 2),
                text,
                fill="black",
                font=font,
            )
    else:
        raise ValueError(f"Invalid axis position: {axis_position}")
    return axis_img


def create_grid_mark(img, num_grid, enlarge=False, chart_bbox=None, min_size=512, color="black"):
    """
    Create a grid overlay on the image with coordinates

    Args:
        image_path: Path to the original img
        grid_size: Size of grid cells in pixels

    Returns:
        PIL.Image: Image with grid overlay
    """

    resize_ratio = 1
    if enlarge:
        width, height = img.size
        if min(width, height) < min_size:
            if width < height:
                new_width = min_size
                resize_ratio = new_width / width
                new_height = int(height * resize_ratio)
            else:
                new_height = min_size
                resize_ratio = new_height / height
                new_width = int(width * resize_ratio)
            img = img.resize((new_width, new_height))

    if chart_bbox:
        x1, y1, x2, y2 = (
            int(chart_bbox[0] / resize_ratio),
            int(chart_bbox[1] / resize_ratio),
            int(chart_bbox[2] / resize_ratio),
            int(chart_bbox[3] / resize_ratio),
        )
        width, height = x2 - x1, y2 - y1
    else:
        width, height = img.size
        x1, y1, x2, y2 = 0, 0, width, height

    # Create a copy for drawing
    img_add_grid = img.copy()
    draw = ImageDraw.Draw(img_add_grid)

    fontsize = min(width, height) // num_grid // 3
    font = ImageFont.truetype("arial.ttf", fontsize)

    # Draw vertical lines
    id_coord = {}
    count = 0
    grid_width = math.ceil(width / num_grid)
    grid_height = math.ceil(height / num_grid)
    for y in range(y1, y2, grid_height):
        for x in range(x1, x2, grid_width):
            id_coord[count] = (
                x / resize_ratio,
                y / resize_ratio,
                (x + grid_width) / resize_ratio,
                (y + grid_height) / resize_ratio,
            )
            draw.rectangle(
                [
                    (x, y),
                    (x + grid_width, y + grid_height),
                ],
                outline=color,
                width=2,
            )
            # Add x coordinate text with black background
            text = str(count)
            # Calculate text size to center it in the grid cell
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Calculate center position of grid cell
            center_x = x + (grid_width - text_width) // 2
            center_y = y + (grid_height - text_height) // 2

            # Draw centered text
            draw.text((center_x, center_y), text, fill=color, font=font)
            count += 1
    return img_add_grid, id_coord


def resize_image(img, max_size=1024):
    # Apply size constraint: max edge = 1024 pixels
    width, height = img.size
    max_edge = max(width, height)
        
    if max_edge > max_size:
        # Calculate resize ratio to keep aspect ratio
        ratio = max_size / max_edge
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height))
    return img