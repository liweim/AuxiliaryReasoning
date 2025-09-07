from PIL import Image, ImageDraw, ImageFont

def open_image(image_or_image_path):
    if isinstance(image_or_image_path, Image.Image):
        return image_or_image_path
    elif isinstance(image_or_image_path, str):
        return Image.open(image_or_image_path)
    else:
        raise ValueError("Unsupported input type!")

def dot_matrix_two_dimensional(image_or_image_path, save_path = None, dots_size_w = 6, dots_size_h = 6, save_img = False, font_path = 'arial.ttf', scaffold_mode='default'):
    """
    takes an original image as input, save the processed image to save_path. Each dot is labeled with two-dimensional Cartesian coordinates (x,y). Suitable for single-image tasks.
    control args:
    1. dots_size_w: the number of columns of the dots matrix
    2. dots_size_h: the number of rows of the dots matrix
    """
    with open_image(image_or_image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img, 'RGB')

        width, height = img.size
        grid_size_w = dots_size_w + 1
        grid_size_h = dots_size_h + 1
        cell_width = width / grid_size_w
        cell_height = height / grid_size_h

        font_size = width // 40
        if scaffold_mode == 'coordinate':
            font_size = width // 60
        font = ImageFont.truetype(font_path, font_size)  # Adjust font size if needed; default == width // 40

        count = 0
        for j in range(1, grid_size_h):
            for i in range(1, grid_size_w):
                x = int(i * cell_width)
                y = int(j * cell_height)

                # Calculate text background region for color analysis
                text_x, text_y = x + 3, y
                count_w = count // dots_size_w
                count_h = count % dots_size_w
                if scaffold_mode == 'default':
                    label_str = f"({count_w+1},{count_h+1})"
                elif scaffold_mode == 'coordinate':
                    label_str = f"({x},{y})"
                elif scaffold_mode == 'only_dots':
                    label_str = ""
                
                # Get text bounding box to determine background region
                bbox = draw.textbbox((text_x, text_y), label_str, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Sample pixels in the text background region
                sample_points = []
                for sample_x in range(int(text_x), int(text_x + text_width), max(1, int(text_width // 5))):
                    for sample_y in range(int(text_y), int(text_y + text_height), max(1, int(text_height // 3))):
                        if 0 <= sample_x < width and 0 <= sample_y < height:
                            sample_points.append(img.getpixel((sample_x, sample_y)))
                
                # Calculate average color of background region
                if sample_points:
                    avg_r = sum(p[0] for p in sample_points) // len(sample_points)
                    avg_g = sum(p[1] for p in sample_points) // len(sample_points)
                    avg_b = sum(p[2] for p in sample_points) // len(sample_points)
                    avg_color = (avg_r, avg_g, avg_b)
                else:
                    # Fallback to original pixel color if no samples
                    avg_color = img.getpixel((x, y))
                
                # choose a more contrasting color from black and white based on background average
                if avg_color[0] + avg_color[1] + avg_color[2] >= 255 * 3 / 2:
                    opposite_color = (0,0,0)
                else:
                    opposite_color = (255,255,255)
                    
                # opposite_color = (0,0,255)

                circle_radius = width // 240  # Adjust dot size if needed; default == width // 240
                draw.ellipse([(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)], fill=opposite_color)

                # Draw text with the calculated opposite color
                if scaffold_mode == 'default':
                    draw.text((text_x, text_y), label_str, fill=opposite_color, font=font)
                elif scaffold_mode == 'coordinate':
                    draw.text((text_x, text_y), f"({x},{y})", fill=opposite_color, font=font)
                count += 1
        if save_img:
            print(">>> dots overlaid image processed, stored in", save_path)
            img.save(save_path)
        return img