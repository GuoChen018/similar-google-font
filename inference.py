import modal
import io

app = modal.App("font-inference")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "transformers",
        "Pillow", "numpy<2", "fastapi", "opencv-python-headless",
        "rembg[cpu]", "easyocr", "gdown",
    )
    .run_commands(
        "python3 -c \"from rembg import new_session; new_session('isnet-general-use')\"",
    )
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)


# ── Pre-compute font index (per-image embeddings with text labels) ─────────────

@app.function(image=image, gpu="A100-40GB", timeout=14400, volumes={"/data": volume})
def build_index():
    """Run once after training to embed every glyph/text image individually."""
    import glob, os, torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    GLYPHS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + ['&', '@', '$', '?', '!', '#']
    GLYPH_DIR = "/data/glyph_images"
    device = torch.device("cuda")

    class FontEmbedder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(1536, 512),
                nn.GELU(),
                nn.Linear(512, 256),
            )

        def forward(self, pixel_values):
            output = self.backbone(pixel_values=pixel_values).last_hidden_state
            cls = output[:, 0]
            patches = output[:, 1:].mean(dim=1)
            combined = torch.cat([cls, patches], dim=1)
            return F.normalize(self.head(combined), dim=1)

    backbone = AutoModel.from_pretrained("facebook/dinov2-base")
    model = FontEmbedder(backbone).to(device)
    model.load_state_dict(torch.load("/data/font_embedder_v8.pt", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    import re
    all_font_files = glob.glob("/data/google-fonts/**/*.ttf", recursive=True)

    SCRIPT_SUFFIXES = {
        'JP', 'KR', 'SC', 'TC', 'HK', 'Arabic', 'Hebrew', 'Thai', 'Bengali',
        'Devanagari', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Gujarati',
        'Gurmukhi', 'Oriya', 'Sinhala', 'Myanmar', 'Khmer', 'Lao', 'Georgian',
        'Armenian', 'Ethiopic', 'Cherokee', 'Tibetan', 'Mongolian', 'Symbols',
        'Math', 'Music', 'SignWriting',
    }
    NOTO_RE = re.compile(r'Noto(?:Sans|Serif)(' + '|'.join(SCRIPT_SUFFIXES) + r')')
    REGIONAL_PREFIXES = {}

    def should_skip_font(path):
        name = os.path.basename(path).replace(".ttf", "")
        if "Guides" in name:
            return True
        if NOTO_RE.search(name):
            return True
        for prefix in ["Playwrite"]:
            if name.startswith(prefix):
                variant = name.split("-")[0].split("[")[0]
                if prefix not in REGIONAL_PREFIXES:
                    REGIONAL_PREFIXES[prefix] = variant
                if variant != REGIONAL_PREFIXES[prefix]:
                    return True
        return False

    font_files = [f for f in all_font_files if not should_skip_font(f)]
    print(f"Scanning images for {len(font_files)} font files ({len(all_font_files) - len(font_files)} filtered out)...")

    TEXT_STRINGS = [
        "Hamburgevons", "Handgloves",
        "AB", "OF", "AC", "THE", "MR", "GO", "NYC",
        "Quick", "Gazing", "Sphinx", "Rhythm",
        "QUALITY", "DESIGN",
        "typography", "elegant",
        "0123456789", "No. 42",
        "Aa Bb Gg", "Quick fox",
        "WAVE", "Boxy", "fjifly", "Kpx",
        "abcdefg", "ABCDEFG",
    ]

    def safe_text_name(t):
        return t.replace(" ", "_").replace(".", "_")

    all_images = []
    all_font_names = []
    all_font_dirs = []
    all_labels = []

    for font_path in font_files:
        font_name = os.path.basename(font_path).replace(".ttf", "")
        font_dir = os.path.basename(os.path.dirname(font_path))

        for g in GLYPHS:
            p = f"{GLYPH_DIR}/{font_name}_{g}.png"
            if os.path.exists(p):
                all_images.append(p)
                all_font_names.append(font_name)
                all_font_dirs.append(font_dir)
                all_labels.append(g)

        for t in TEXT_STRINGS:
            p = f"{GLYPH_DIR}/{font_name}_text_{safe_text_name(t)}.png"
            if os.path.exists(p):
                all_images.append(p)
                all_font_names.append(font_name)
                all_font_dirs.append(font_dir)
                all_labels.append(t)

    unique_fonts = sorted(set(all_font_names))
    unique_families = sorted(set(all_font_dirs))
    print(f"Found {len(all_images)} images across {len(unique_fonts)} fonts ({len(unique_families)} families)")
    print(f"Label examples: {all_labels[:5]} ...")

    class GlyphImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    dataset = GlyphImageDataset(all_images, transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding images"):
            embs = model(pixel_values=batch.to(device))
            all_embeddings.append(embs.cpu())

    all_embeddings = torch.cat(all_embeddings)

    torch.save({
        "font_names": all_font_names,
        "font_dirs": all_font_dirs,
        "labels": all_labels,
        "embeddings": all_embeddings,
    }, "/data/font_index_v8.pt")
    volume.commit()
    print(f"Per-image index saved: {len(all_images)} embeddings across {len(unique_fonts)} fonts, "
          f"shape {all_embeddings.shape}")


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ── Text detector: EasyOCR (word-level boxes + OCR) ───────────────────────────

@app.cls(image=image, gpu="T4", timeout=60, volumes={"/data": volume})
class TextDetector:
    @modal.enter()
    def load_models(self):
        import easyocr
        self.reader = easyocr.Reader(
            ['en'], gpu=True,
            model_storage_directory='/data/easyocr_models'
        )
        print("TextDetector loaded: EasyOCR (en, CRAFT detection)")

    @modal.method()
    def detect(self, image_bytes: bytes, debug: bool = False):
        import numpy as np
        from PIL import Image
        import cv2
        import base64

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        raw_results = self.reader.readtext(img_np)
        regions = self._merge_boxes(raw_results, img_np.shape)

        response = {"regions": regions}

        if debug:
            annotated = img_np.copy()
            for region in regions:
                x, y = region["x"], region["y"]
                w, h = region["width"], region["height"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 3)
                cv2.putText(annotated, region["text"],
                            (x, max(y - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            response["debug_images"] = {
                "annotated": base64.b64encode(buf.getvalue()).decode()
            }

            import os
            os.makedirs("/data/debug_detect", exist_ok=True)
            Image.fromarray(annotated).save("/data/debug_detect/annotated.png")
            volume.commit()

        return response

    def _merge_boxes(self, ocr_results, img_shape):
        if not ocr_results:
            return []

        import math

        boxes = []
        for bbox, text, conf in ocr_results:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            tl, tr = bbox[0], bbox[1]
            angle = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
            angle = max(-45.0, min(45.0, angle))
            boxes.append({
                "x": int(min(xs)), "y": int(min(ys)),
                "width": int(max(xs) - min(xs)),
                "height": int(max(ys) - min(ys)),
                "text": text, "conf": conf,
                "angle": angle,
            })

        boxes.sort(key=lambda b: (b["y"], b["x"]))

        # Pass 1: merge words on the same line
        lines = []
        used = set()
        for i, box in enumerate(boxes):
            if i in used:
                continue
            group = [box]
            used.add(i)
            center_y = box["y"] + box["height"] / 2

            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                other = boxes[j]
                other_cy = other["y"] + other["height"] / 2
                height_ref = max(box["height"], other["height"])
                if abs(center_y - other_cy) < height_ref * 0.5:
                    group.append(other)
                    used.add(j)

            group.sort(key=lambda b: b["x"])
            min_x = min(b["x"] for b in group)
            min_y = min(b["y"] for b in group)
            max_x = max(b["x"] + b["width"] for b in group)
            max_y = max(b["y"] + b["height"] for b in group)

            avg_angle = sum(b["angle"] for b in group) / len(group)
            lines.append({
                "x": min_x, "y": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y,
                "text": " ".join(b["text"] for b in group),
                "angle": round(avg_angle, 2),
                "words": [
                    {"x": b["x"], "y": b["y"],
                     "width": b["width"], "height": b["height"],
                     "text": b["text"]}
                    for b in group
                ],
            })

        # Pass 2: cluster lines into blocks (same font = spatially close + similar height)
        lines.sort(key=lambda l: l["y"])
        blocks = []
        used2 = set()
        for i, line in enumerate(lines):
            if i in used2:
                continue
            cluster = [line]
            used2.add(i)

            for j in range(i + 1, len(lines)):
                if j in used2:
                    continue
                other = lines[j]

                # Height similarity: ratio of smaller to larger must be > 0.4
                h_min = min(line["height"], other["height"])
                h_max = max(line["height"], other["height"])
                if h_max > 0 and h_min / h_max < 0.4:
                    continue

                # Vertical proximity: gap between bottom of cluster and top of next line
                cluster_bottom = max(l["y"] + l["height"] for l in cluster)
                gap = other["y"] - cluster_bottom
                avg_height = sum(l["height"] for l in cluster) / len(cluster)
                if gap < avg_height * 1.5:
                    cluster.append(other)
                    used2.add(j)

            min_x = min(l["x"] for l in cluster)
            min_y = min(l["y"] for l in cluster)
            max_x = max(l["x"] + l["width"] for l in cluster)
            max_y = max(l["y"] + l["height"] for l in cluster)

            blocks.append({
                "id": str(len(blocks)),
                "x": min_x, "y": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y,
                "text": " ".join(l["text"] for l in cluster),
                "line_count": len(cluster),
                "lines": [
                    {"x": l["x"], "y": l["y"],
                     "width": l["width"], "height": l["height"],
                     "text": l["text"], "angle": l["angle"],
                     "words": l.get("words", [])}
                    for l in cluster
                ],
            })

        return blocks


# ── Font matcher: rembg preprocessing + OCR-filtered DINOv2 matching ───────────

@app.cls(image=image, gpu="T4", timeout=120, volumes={"/data": volume})
class FontMatcher:
    @modal.enter()
    def load_models(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModel
        from rembg import new_session

        self.device = torch.device("cuda")
        self.rembg_session = new_session("isnet-general-use")

        class FontEmbedder(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.Linear(1536, 512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                )
            def forward(self, pixel_values):
                output = self.backbone(pixel_values=pixel_values).last_hidden_state
                cls = output[:, 0]
                patches = output[:, 1:].mean(dim=1)
                combined = torch.cat([cls, patches], dim=1)
                return F.normalize(self.head(combined), dim=1)

        backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.font_model = FontEmbedder(backbone).to(self.device)
        self.font_model.load_state_dict(
            torch.load("/data/font_embedder_v8.pt", map_location=self.device)
        )
        self.font_model.eval()

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        index = torch.load("/data/font_index_v8.pt", map_location="cpu")
        self.index_font_names = index["font_names"]
        self.index_font_dirs = index["font_dirs"]
        self.index_labels = index["labels"]
        self.index_embeddings = index["embeddings"]

        self.font_name_to_dir = {}
        for name, d in zip(self.index_font_names, self.index_font_dirs):
            if name not in self.font_name_to_dir:
                self.font_name_to_dir[name] = d

        unique_families = len(set(self.index_font_dirs))
        print(f"Models loaded: DINOv2 + rembg (isnet) + per-image index "
              f"({len(self.index_font_names)} embeddings, {unique_families} families)")

    MAX_TILES_PER_BLOCK = 6
    MAX_CHARS_PER_GROUP = 3

    def _img_to_base64(self, pil_img):
        import base64
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _word_boxes_to_groups(self, words):
        """Split long words into groups of 1-3 chars via equal-width subdivision."""
        groups = []
        for w in words:
            text = w.get("text", "")
            n_chars = len(text.replace(" ", "")) or 1
            if n_chars <= self.MAX_CHARS_PER_GROUP:
                groups.append(w)
            else:
                n_groups = -(-n_chars // self.MAX_CHARS_PER_GROUP)  # ceil div
                slice_w = w["width"] / n_groups
                for i in range(n_groups):
                    groups.append({
                        "x": int(w["x"] + i * slice_w),
                        "y": w["y"],
                        "width": int(slice_w),
                        "height": w["height"],
                        "text": text[i * self.MAX_CHARS_PER_GROUP
                                     : (i + 1) * self.MAX_CHARS_PER_GROUP],
                    })
        return groups

    def _generous_crop(self, pil_img, box, pad_ratio=0.20):
        """Crop a box from the image with generous padding on all sides."""
        img_w, img_h = pil_img.size
        bx, by, bw, bh = box["x"], box["y"], box["width"], box["height"]
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        x0 = max(0, bx - pad_x)
        y0 = max(0, by - pad_y)
        x1 = min(img_w, bx + bw + pad_x)
        y1 = min(img_h, by + bh + pad_y)
        return pil_img.crop((x0, y0, x1, y1))

    def _tile_quality_ok(self, pil_sq):
        """Reject empty, nearly full, or extreme ink layouts."""
        import numpy as np

        g = np.array(pil_sq.convert("L"), dtype=np.float32)
        if g.size < 64:
            return False
        ink = (g < 200).astype(np.float32)
        frac = ink.mean()
        if frac < 0.004 or frac > 0.62:
            return False
        ys, xs = np.where(ink > 0.5)
        if len(xs) < 8:
            return False
        span_x = (xs.max() - xs.min() + 1) / g.shape[1]
        span_y = (ys.max() - ys.min() + 1) / g.shape[0]
        if span_x < 0.04 and span_y < 0.04:
            return False
        return True

    def _match_chars(self, char_images, ocr_text: str, top_k: int):
        """Batch embed character images and match against OCR-filtered index."""
        import torch
        from collections import defaultdict

        tensors = torch.stack([self.transform(img) for img in char_images]).to(self.device)
        with torch.inference_mode():
            char_embs = self.font_model(pixel_values=tensors)

        if ocr_text:
            ocr_chars = set(ocr_text)
            matching_indices = [
                i for i, label in enumerate(self.index_labels)
                if (len(label) == 1 and label in ocr_chars) or label == ocr_text
            ]
            if matching_indices:
                filtered_embs = self.index_embeddings[matching_indices]
                filtered_names = [self.index_font_names[i] for i in matching_indices]
                print(f"OCR filter: '{ocr_text}' → {len(matching_indices)} entries, "
                      f"{len(set(filtered_names))} fonts")
            else:
                filtered_embs = self.index_embeddings
                filtered_names = list(self.index_font_names)
                print(f"OCR filter: no matches for '{ocr_text}', using full index")
        else:
            filtered_embs = self.index_embeddings
            filtered_names = list(self.index_font_names)
            print(f"No OCR text — full index ({len(filtered_names)} entries)")

        sims = torch.mm(char_embs.cpu(), filtered_embs.T)

        font_entry_indices = defaultdict(list)
        for j, fn in enumerate(filtered_names):
            font_entry_indices[fn].append(j)

        font_scores = {}
        for fn, idxs in font_entry_indices.items():
            font_sims = sims[:, idxs]
            best_per_char = font_sims.max(dim=1).values
            font_scores[fn] = best_per_char.mean().item()

        print(f"Matched {len(char_images)} chars against {len(font_entry_indices)} fonts")

        family_best = {}
        for font_name, score in font_scores.items():
            font_dir = self.font_name_to_dir.get(font_name, font_name)
            if font_dir not in family_best or score > family_best[font_dir][1]:
                family_best[font_dir] = (font_name, score)

        ranked = sorted(family_best.values(), key=lambda x: x[1], reverse=True)

        SUPER_FAMILY_PREFIX_LEN = 6
        MAX_PER_SUPER_FAMILY = 2
        results = []
        for font_name, score in ranked:
            font_dir = self.font_name_to_dir.get(font_name, font_name)
            siblings = sum(
                1 for r in results
                if _common_prefix_len(font_dir,
                                      self.font_name_to_dir.get(r["font"], ""))
                   >= SUPER_FAMILY_PREFIX_LEN
            )
            if siblings >= MAX_PER_SUPER_FAMILY:
                continue
            results.append({
                "rank": len(results) + 1,
                "font": font_name,
                "family": self.font_name_to_dir.get(font_name, ""),
                "similarity": round(score, 4)
            })
            if len(results) >= top_k:
                break

        return results

    def _preprocess(self, image_bytes: bytes, angle: float = 0.0):
        """Single-image preprocess: rembg + sigmoid alpha inversion + crop + square."""
        import numpy as np
        from PIL import Image
        from rembg import remove

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        debug_data = {"crop": self._img_to_base64(img)}

        output = remove(img, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3].astype(np.float32) / 255.0
        alpha_sharp = 1.0 / (1.0 + np.exp(-50 * (alpha - 0.5)))
        bw = (255 * (1.0 - alpha_sharp)).astype(np.uint8)
        bw = np.where(bw > 220, 255, bw).astype(np.uint8)

        coords = np.argwhere(bw < 250)
        if len(coords) > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            margin = max((y1 - y0), (x1 - x0)) // 10
            h, w = bw.shape
            y0, x0 = max(0, y0 - margin), max(0, x0 - margin)
            y1, x1 = min(h - 1, y1 + margin), min(w - 1, x1 + margin)
            bw = bw[y0:y1 + 1, x0:x1 + 1]

        ch, cw = bw.shape
        size = max(ch, cw)
        square = np.full((size, size), 255, dtype=np.uint8)
        py, px = (size - ch) // 2, (size - cw) // 2
        square[py:py + ch, px:px + cw] = bw
        bw = square

        query_img = Image.fromarray(bw).convert("RGB")
        debug_data["preprocessed"] = self._img_to_base64(query_img)
        return query_img, debug_data

    def _match(self, query_img, ocr_text: str, top_k: int):
        """Embed single image and match against index (legacy)."""
        import torch
        from collections import defaultdict

        tensor = self.transform(query_img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            query_emb = self.font_model(pixel_values=tensor)

        if ocr_text:
            ocr_chars = set(ocr_text)
            matching_indices = [
                i for i, label in enumerate(self.index_labels)
                if label == ocr_text or (len(label) == 1 and label in ocr_chars)
            ]
            if matching_indices:
                filtered_embs = self.index_embeddings[matching_indices]
                filtered_names = [self.index_font_names[i] for i in matching_indices]
                sims = torch.mm(query_emb.cpu(), filtered_embs.T).squeeze(0)
                font_sims = defaultdict(list)
                for i, fn in enumerate(filtered_names):
                    font_sims[fn].append(sims[i].item())
                print(f"OCR filter: '{ocr_text}' → {len(matching_indices)} entries, "
                      f"{len(font_sims)} fonts")
            else:
                sims = torch.mm(query_emb.cpu(), self.index_embeddings.T).squeeze(0)
                font_sims = defaultdict(list)
                for i, fn in enumerate(self.index_font_names):
                    font_sims[fn].append(sims[i].item())
        else:
            sims = torch.mm(query_emb.cpu(), self.index_embeddings.T).squeeze(0)
            font_sims = defaultdict(list)
            for i, fn in enumerate(self.index_font_names):
                font_sims[fn].append(sims[i].item())
            print(f"No OCR text — full index ({len(self.index_font_names)} entries)")

        font_scores = {}
        for fn, sim_list in font_sims.items():
            top3 = sorted(sim_list, reverse=True)[:3]
            font_scores[fn] = sum(top3) / len(top3)

        family_best = {}
        for font_name, score in font_scores.items():
            font_dir = self.font_name_to_dir.get(font_name, font_name)
            if font_dir not in family_best or score > family_best[font_dir][1]:
                family_best[font_dir] = (font_name, score)

        ranked = sorted(family_best.values(), key=lambda x: x[1], reverse=True)

        SUPER_FAMILY_PREFIX_LEN = 6
        MAX_PER_SUPER_FAMILY = 2
        results = []
        for font_name, score in ranked:
            font_dir = self.font_name_to_dir.get(font_name, font_name)
            siblings = sum(
                1 for r in results
                if _common_prefix_len(font_dir,
                                      self.font_name_to_dir.get(r["font"], ""))
                   >= SUPER_FAMILY_PREFIX_LEN
            )
            if siblings >= MAX_PER_SUPER_FAMILY:
                continue
            results.append({
                "rank": len(results) + 1,
                "font": font_name,
                "family": self.font_name_to_dir.get(font_name, ""),
                "similarity": round(score, 4)
            })
            if len(results) >= top_k:
                break

        return results

    @modal.method()
    def find_similar(self, image_bytes: bytes, ocr_text: str = "",
                     top_k: int = 10, save_debug: bool = False,
                     preprocess: str = "rembg"):
        from PIL import Image
        import os

        if preprocess == "rembg":
            query_img, debug_data = self._preprocess(image_bytes)
            print(f"Preprocessing: rembg + crop/pad")
        else:
            query_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print(f"Preprocessing: none (raw photo)")

        if save_debug:
            os.makedirs("/data/debug_preprocess", exist_ok=True)
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(
                "/data/debug_preprocess/original.png")
            query_img.save("/data/debug_preprocess/preprocessed.png")
            volume.commit()
            print("Debug images saved to /data/debug_preprocess/")

        return self._match(query_img, ocr_text, top_k)

    @modal.method()
    def batch_find_similar(self, full_image_bytes: bytes, regions: list,
                           top_k: int = 10, debug: bool = False):
        """
        Simple pipeline: EasyOCR word boxes → generous crops from original
        image → _preprocess (rembg + alpha inversion) → _match_chars.
        Long words (>3 chars) are subdivided into equal-width groups.
        """
        import os
        from PIL import Image as _PilImg

        full_img = _PilImg.open(io.BytesIO(full_image_bytes)).convert("RGB")
        results = {}
        debug_images = {}

        for block in regions:
            block_id = block["id"]
            all_tiles = []
            text_parts = []
            block_debug = {"lines": []}

            for line in block.get("lines", [block]):
                words = line.get("words", [])
                if not words:
                    words = [{"x": line["x"], "y": line["y"],
                              "width": line["width"], "height": line["height"],
                              "text": line.get("text", "")}]
                groups = self._word_boxes_to_groups(words)
                ocr_line = line.get("text", "")
                text_parts.append(ocr_line)

                line_debug = {"id": f"{block_id}_line", "chars": [],
                              "raw_crops": []}

                for g in groups:
                    crop = self._generous_crop(full_img, g)
                    if crop.size[0] < 8 or crop.size[1] < 8:
                        continue
                    buf = io.BytesIO()
                    crop.save(buf, format="PNG")
                    tile, _ = self._preprocess(buf.getvalue())
                    if self._tile_quality_ok(tile):
                        all_tiles.append(tile)
                        if debug:
                            line_debug["raw_crops"].append(
                                self._img_to_base64(crop))
                            line_debug["chars"].append(
                                self._img_to_base64(tile))

                print(f"  Block {block_id}: {len(groups)} word groups "
                      f"→ {len(all_tiles)} tiles from '{ocr_line}'")
                if debug:
                    block_debug["lines"].append(line_debug)

            all_tiles = all_tiles[:self.MAX_TILES_PER_BLOCK]
            combined_ocr = " ".join(text_parts).strip()

            if all_tiles:
                block_results = self._match_chars(
                    all_tiles, combined_ocr, top_k)
            else:
                block_results = []

            results[block_id] = block_results

            if debug:
                block_debug["total_chars"] = len(all_tiles)
                debug_images[block_id] = block_debug

        response = {"results": results}
        if debug:
            response["debug_images"] = debug_images
        return response


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.function(image=image, timeout=10)
@modal.fastapi_endpoint(method="GET")
async def health():
    return {"status": "ok"}


@app.function(image=image, gpu="T4", timeout=60, volumes={"/data": volume})
@modal.fastapi_endpoint(method="POST")
async def detect_text(request: dict):
    """
    Step 1: Detect text regions + OCR.
    Request:  { "image": "<base64>", "debug": false }
    Response: { "regions": [{ "id": "0", "x": 100, "y": 50, "width": 200, "height": 80, "text": "OF" }],
                "debug_images": { "annotated": "<base64 png>" } }
    """
    import base64

    image_b64 = request.get("image", "")
    if not image_b64:
        return {"error": "No image provided"}

    image_bytes = base64.b64decode(image_b64)
    debug = request.get("debug", False)

    detector = TextDetector()
    return detector.detect.remote(image_bytes, debug=debug)


@app.function(image=image, gpu="T4", timeout=120, volumes={"/data": volume})
@modal.fastapi_endpoint(method="POST")
async def batch_match(request: dict):
    """
    Step 2: Font matching from full image + detected regions with word boxes.
    Request:  { "image": "<base64>", "regions": [...from detect_text...],
                "top_k": 10, "debug": false }
    Response: { "results": { "0": [{ "rank": 1, "font": "...", "similarity": 0.89 }] },
                "debug_images": { ... } }
    """
    import base64

    image_b64 = request.get("image", "")
    regions = request.get("regions", [])
    if not image_b64 or not regions:
        return {"error": "Provide both 'image' and 'regions'"}

    top_k = request.get("top_k", 10)
    debug = request.get("debug", False)
    image_bytes = base64.b64decode(image_b64)

    matcher = FontMatcher()
    return matcher.batch_find_similar.remote(
        image_bytes, regions, top_k=top_k, debug=debug)


@app.function(image=image, gpu="T4", timeout=120, volumes={"/data": volume})
@modal.fastapi_endpoint(method="POST")
async def similar_fonts(request: dict):
    """
    Legacy single-image endpoint (backward compat + CLI testing).
    Request:  { "image": "<base64>", "text": "OF", "top_k": 10 }
    Response: { "results": [{ "rank": 1, "font": "...", "similarity": 0.92 }] }
    """
    import base64

    image_b64 = request.get("image", "")
    ocr_text = request.get("text", "")
    top_k = request.get("top_k", 10)

    if not image_b64:
        return {"error": "No image provided"}

    image_bytes = base64.b64decode(image_b64)
    preprocess = request.get("preprocess", "rembg")
    matcher = FontMatcher()
    results = matcher.find_similar.remote(
        image_bytes, ocr_text=ocr_text, top_k=top_k, preprocess=preprocess
    )
    return {"results": results}


# ── Local CLI for testing ─────────────────────────────────────────────────────

@app.local_entrypoint()
def main(command: str = "test", image_path: str = "", text: str = "",
         top_k: int = 10, preprocess: str = "rembg"):
    """
    Usage:
      modal run inference.py --command build-index
      modal run inference.py --command test --image-path crop1.jpg --text "OF"
      modal run inference.py --command test-detect --image-path test_images/sign1.jpg
      modal run inference.py --command test-full --image-path test_images/sign1.jpg
    """
    if command == "build-index":
        build_index.remote()
        print("Index built successfully.")

    elif command == "test":
        if not image_path:
            print("Provide --image-path")
            return
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        matcher = FontMatcher()
        results = matcher.find_similar.remote(
            image_bytes, ocr_text=text, top_k=top_k,
            save_debug=True, preprocess=preprocess
        )
        print(f"\nTop {top_k} similar fonts{f' (OCR: {text})' if text else ''} [preprocess={preprocess}]:")
        print("-" * 40)
        for r in results:
            family = r.get('family', '')
            print(f"  #{r['rank']}  {r['font']}  [{family}]  (similarity: {r['similarity']})")
        print(f"\nDebug images saved — download with:")
        print(f"  python3 -m modal volume get font-data debug_preprocess/original.png ./debug_original.png")
        print(f"  python3 -m modal volume get font-data debug_preprocess/preprocessed.png ./debug_preprocessed.png")

    elif command == "test-detect":
        if not image_path:
            print("Provide --image-path")
            return
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        detector = TextDetector()
        result = detector.detect.remote(image_bytes, debug=True)
        regions = result["regions"]
        print(f"\nDetected {len(regions)} text block(s):")
        print("-" * 40)
        for r in regions:
            n_lines = r.get("line_count", 1)
            lines_label = f"  ({n_lines} line{'s' if n_lines > 1 else ''})" if n_lines > 1 else ""
            print(f"  Block {r['id']}: \"{r['text']}\"{lines_label}  "
                  f"(x={r['x']}, y={r['y']}, {r['width']}x{r['height']})")
            for li, line in enumerate(r.get("lines", [])):
                print(f"    Line {li}: \"{line['text']}\"  "
                      f"({line['width']}x{line['height']}, angle={line.get('angle', 0):.1f}°)")
        print(f"\nDebug image saved — download with:")
        print(f"  python3 -m modal volume get font-data debug_detect/annotated.png ./debug_annotated.png")

    elif command == "test-full":
        if not image_path:
            print("Provide --image-path (full photo, not a crop)")
            return
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        print("=" * 50)
        print("STEP 1: Detecting text regions (EasyOCR)")
        print("=" * 50)
        detector = TextDetector()
        detect_result = detector.detect.remote(image_bytes, debug=True)
        regions = detect_result["regions"]
        print(f"\nFound {len(regions)} text block(s):")
        for r in regions:
            n_lines = r.get("line_count", 1)
            print(f"  Block {r['id']}: \"{r['text']}\"  "
                  f"({n_lines} line{'s' if n_lines > 1 else ''})")
            for li, line in enumerate(r.get("lines", [])):
                n_words = len(line.get("words", []))
                print(f"    Line {li}: \"{line['text']}\"  "
                      f"({line['width']}x{line['height']}, "
                      f"{n_words} word(s), angle={line.get('angle', 0):.1f}°)")

        if not regions:
            print("\nNo text detected.")
            return

        print(f"\n{'=' * 50}")
        print("STEP 2: Word crops → preprocess → match")
        print("=" * 50)
        matcher = FontMatcher()
        match_result = matcher.batch_find_similar.remote(
            image_bytes, regions, top_k=top_k, debug=True
        )
        for block_id, matches in match_result["results"].items():
            block_text = next(
                (r["text"] for r in regions if r["id"] == block_id), "")
            total = match_result.get("debug_images", {}).get(
                block_id, {}).get("total_chars", "?")
            print(f"\n  Block {block_id} \"{block_text}\" "
                  f"({total} tiles) — Top {top_k}:")
            print(f"  {'-' * 38}")
            for m in matches:
                print(f"    #{m['rank']}  {m['font']}  [{m['family']}]  "
                      f"({m['similarity']})")

        import os, base64 as _b64
        from PIL import Image as _PImg
        from datetime import datetime

        test_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%m%d_%H%M")
        out_dir = f"debug_output/{test_name}_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)

        if "debug_images" in detect_result and "annotated" in detect_result["debug_images"]:
            raw = _b64.b64decode(detect_result["debug_images"]["annotated"])
            _PImg.open(io.BytesIO(raw)).save(f"{out_dir}/annotated.png")

        dbg = match_result.get("debug_images", {})
        for block_id, bd in dbg.items():
            for li, ld in enumerate(bd.get("lines", [])):
                line_dir = f"{out_dir}/block{block_id}_line{li}"
                os.makedirs(line_dir, exist_ok=True)
                for ci, crop_b64 in enumerate(ld.get("raw_crops", [])):
                    raw = _b64.b64decode(crop_b64)
                    _PImg.open(io.BytesIO(raw)).save(
                        f"{line_dir}/raw_crop_{ci}.png")
                for ci, char_b64 in enumerate(ld.get("chars", [])):
                    raw = _b64.b64decode(char_b64)
                    _PImg.open(io.BytesIO(raw)).save(
                        f"{line_dir}/tile_{ci}.png")

        print(f"\n{'=' * 50}")
        print(f"DEBUG IMAGES saved to: {out_dir}/")
        print("=" * 50)
        if os.path.exists(f"{out_dir}/annotated.png"):
            print(f"  {out_dir}/annotated.png")
        for block_id, bd in dbg.items():
            for li, ld in enumerate(bd.get("lines", [])):
                n_tiles = len(ld.get("chars", []))
                n_raw = len(ld.get("raw_crops", []))
                print(f"  {out_dir}/block{block_id}_line{li}/  "
                      f"({n_raw} crop(s) → {n_tiles} tile(s))")

    else:
        print(f"Unknown command: {command}")
        print("Available: build-index, test, test-detect, test-full")
