from pathlib import Path

import torch

from PIL import Image

from torchvision import transforms as TF

def list_image_files(image_dir: str) -> list[str]:

    root = Path(image_dir)

    if not root.is_dir():

        raise FileNotFoundError(f"Input directory not found: {image_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = [str(path) for path in sorted(root.iterdir()) if path.suffix.lower() in exts]

    if not image_paths:

        raise ValueError(f"No supported images found in {image_dir}")

    return image_paths

def load_and_preprocess_images(image_path_list: list[str], mode: str = "crop") -> torch.Tensor:

    if len(image_path_list) == 0:

        raise ValueError("At least 1 image is required")

    if mode not in {"crop", "pad"}:

        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []

    shapes = set()

    to_tensor = TF.ToTensor()

    target_size = 518

    for image_path in image_path_list:

        img = Image.open(image_path)

        if img.mode == "RGBA":

            background = Image.new("RGBA", img.size, (255, 255, 255, 255))

            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":

            if width >= height:

                new_width = target_size

                new_height = round(height * (new_width / width) / 14) * 14

            else:

                new_height = target_size

                new_width = round(width * (new_height / height) / 14) * 14

        else:

            new_width = target_size

            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        img = to_tensor(img)

        if mode == "crop" and new_height > target_size:

            start_y = (new_height - target_size) // 2

            img = img[:, start_y : start_y + target_size, :]

        if mode == "pad":

            h_padding = target_size - img.shape[1]

            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:

                pad_top = h_padding // 2

                pad_bottom = h_padding - pad_top

                pad_left = w_padding // 2

                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(

                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0

                )

        shapes.add((img.shape[1], img.shape[2]))

        images.append(img)

    if len(shapes) > 1:

        max_height = max(shape[0] for shape in shapes)

        max_width = max(shape[1] for shape in shapes)

        padded_images = []

        for img in images:

            h_padding = max_height - img.shape[1]

            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:

                pad_top = h_padding // 2

                pad_bottom = h_padding - pad_top

                pad_left = w_padding // 2

                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(

                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0

                )

            padded_images.append(img)

        images = padded_images

    return torch.stack(images)
