import modal

app = modal.App("font-augment-samples")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "Pillow", "numpy")
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)

@app.function(image=image, timeout=120, volumes={"/data": volume})
def save_samples():
    import glob, os, random, torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from torchvision import transforms

    random.seed(42)
    OUT_DIR = "/data/recall_samples_v6"
    os.makedirs(OUT_DIR, exist_ok=True)

    class RandomBackground:
        def __call__(self, img):
            w, h = img.size
            r, g, b = random.randint(30, 240), random.randint(30, 240), random.randint(30, 240)
            bg = Image.new("RGB", (w, h), (r, g, b))
            draw = ImageDraw.Draw(bg)

            if random.random() < 0.4:
                bg_arr = np.array(bg).astype(np.int16)
                shift = random.randint(-40, 40)
                grad = np.linspace(0, shift, h).reshape(h, 1, 1).astype(np.int16)
                bg_arr = np.clip(bg_arr + grad, 0, 255).astype(np.uint8)
                bg = Image.fromarray(bg_arr)

            if random.random() < 0.3:
                lr, lg, lb = (
                    max(0, r - random.randint(20, 60)),
                    max(0, g - random.randint(20, 60)),
                    max(0, b - random.randint(20, 60)),
                )
                draw = ImageDraw.Draw(bg)
                spacing = random.randint(15, 50)
                thickness = random.randint(1, 3)
                for y in range(0, h, spacing):
                    draw.line([(0, y), (w, y)], fill=(lr, lg, lb), width=thickness)

            if random.random() < 0.2:
                noise = np.random.randint(-25, 25, (h, w, 3), dtype=np.int16)
                bg_arr = np.clip(np.array(bg).astype(np.int16) + noise, 0, 255).astype(np.uint8)
                bg = Image.fromarray(bg_arr)

            arr = np.array(img)
            gray = np.mean(arr, axis=2)
            mask = (gray < 240).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask, mode="L")
            bg.paste(img, mask=mask_img)
            return bg

    class RandomLineOcclusion:
        def __init__(self, p=0.25, max_lines=5):
            self.p = p
            self.max_lines = max_lines
        def __call__(self, img):
            if random.random() > self.p:
                return img
            img = img.copy()
            draw = ImageDraw.Draw(img)
            w, h = img.size
            n = random.randint(1, self.max_lines)
            for _ in range(n):
                y = random.randint(0, h)
                color = random.randint(0, 180)
                thickness = random.randint(1, 4)
                draw.line([(0, y), (w, y)], fill=(color, color, color), width=thickness)
            return img

    class RandomShadow:
        def __init__(self, p=0.3):
            self.p = p
        def __call__(self, img):
            if random.random() > self.p:
                return img
            arr = np.array(img).astype(np.float32)
            h, w = arr.shape[:2]
            end_val = random.uniform(0.5, 0.85)
            if random.random() < 0.5:
                grad = np.linspace(1.0, end_val, h).reshape(h, 1, 1)
            else:
                grad = np.linspace(1.0, end_val, w).reshape(1, w, 1)
            arr = np.clip(arr * grad, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

    class GaussianNoise:
        def __init__(self, max_sigma=0.05):
            self.max_sigma = max_sigma
        def __call__(self, tensor):
            sigma = random.uniform(0, self.max_sigma)
            return tensor + torch.randn_like(tensor) * sigma

    train_aug = transforms.Compose([
        RandomBackground(),
        RandomLineOcclusion(p=0.25, max_lines=5),
        RandomShadow(p=0.3),
        transforms.Resize((224, 224)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.4, hue=0.06),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomInvert(p=0.15),
        transforms.ToTensor(),
        GaussianNoise(max_sigma=0.05),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])

    camera_aug = transforms.Compose([
        RandomBackground(),
        RandomLineOcclusion(p=0.2, max_lines=3),
        RandomShadow(p=0.25),
        transforms.Resize((224, 224)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.RandomRotation(6),
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.3, hue=0.06),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomInvert(p=0.1),
        transforms.ToTensor(),
        GaussianNoise(max_sigma=0.06),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    def render_text(font_path, text="Hamburgevons", size=224):
        for font_size in range(120, 20, -4):
            try:
                font = ImageFont.truetype(font_path, font_size)
                img = Image.new("RGB", (size, size), "white")
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if tw <= size - 10 and th <= size - 10:
                    x = (size - tw) // 2 - bbox[0]
                    y = (size - th) // 2 - bbox[1]
                    draw.text((x, y), text, font=font, fill="black")
                    return img
            except Exception:
                continue
        return None

    def tensor_to_pil(t):
        t = t.clamp(0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    font_files = glob.glob("/data/google-fonts/**/*.ttf", recursive=True)
    random.shuffle(font_files)

    saved = 0
    for font_path in font_files:
        if saved >= 10:
            break
        font_name = os.path.basename(font_path).replace(".ttf", "")
        if "Guides" in font_name:
            continue

        base_img = render_text(font_path)
        if base_img is None:
            continue

        W, H = 224, 224
        grid = Image.new("RGB", (W * 7, H + 20), "white")

        labels = ["Clean", "Train 1", "Train 2", "Train 3", "Camera 1", "Camera 2", "Camera 3"]
        label_draw = ImageDraw.Draw(grid)
        for i, label in enumerate(labels):
            label_draw.text((i * W + 5, 2), label, fill="black")

        grid.paste(base_img.resize((W, H)), (0, 20))
        for i in range(3):
            aug_tensor = train_aug(base_img.copy())
            grid.paste(tensor_to_pil(aug_tensor).resize((W, H)), ((i + 1) * W, 20))
        for i in range(3):
            aug_tensor = camera_aug(base_img.copy())
            grid.paste(tensor_to_pil(aug_tensor).resize((W, H)), ((i + 4) * W, 20))

        grid.save(f"{OUT_DIR}/{font_name}.png")
        print(f"Saved {font_name}")
        saved += 1

    volume.commit()
    print(f"Done. {saved} sample grids saved to {OUT_DIR}/")

@app.local_entrypoint()
def main():
    save_samples.remote()
