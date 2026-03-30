import modal
import io

app = modal.App("font-inference-base")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers",
        "Pillow", "numpy"
    )
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)


# ── Pre-compute font index (raw DINOv2, no fine-tuning) ──────────────────────

@app.function(image=image, gpu="A100-40GB", timeout=3600, volumes={"/data": volume})
def build_index():
    import glob, os, torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    GLYPHS = ['a', 'g', 'R', 'Q', 'e', 'G', 'S', 'n', 'f', 't', '0', '&', 'M', 'W', 'O', 'I']
    GLYPH_DIR = "/data/glyph_images"
    device = torch.device("cuda")

    backbone = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    backbone.eval()

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
    print(f"Scanning glyph images for {len(font_files)} font files ({len(all_font_files) - len(font_files)} filtered out)...")

    all_images = []
    all_font_indices = []
    font_names = []
    font_name_to_idx = {}

    for font_path in font_files:
        font_name = os.path.basename(font_path).replace(".ttf", "")
        glyph_paths = [f"{GLYPH_DIR}/{font_name}_{g}.png" for g in GLYPHS]
        existing = [p for p in glyph_paths if os.path.exists(p)]
        if not existing:
            continue

        if font_name not in font_name_to_idx:
            font_name_to_idx[font_name] = len(font_names)
            font_names.append(font_name)

        fidx = font_name_to_idx[font_name]
        for p in existing:
            all_images.append(p)
            all_font_indices.append(fidx)

    print(f"Found {len(all_images)} glyph images across {len(font_names)} fonts")

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
        for batch in tqdm(loader, desc="Embedding glyphs"):
            cls = backbone(pixel_values=batch.to(device)).last_hidden_state[:, 0]
            all_embeddings.append(F.normalize(cls, dim=1).cpu())

    all_embeddings = torch.cat(all_embeddings)
    font_indices_tensor = torch.tensor(all_font_indices)

    index_embeddings = []
    for fidx in range(len(font_names)):
        mask = font_indices_tensor == fidx
        avg = all_embeddings[mask].mean(dim=0)
        index_embeddings.append(F.normalize(avg, dim=0))

    index_tensor = torch.stack(index_embeddings)
    torch.save({"names": font_names, "embeddings": index_tensor}, "/data/font_index_base.pt")
    volume.commit()
    print(f"Index saved: {len(font_names)} fonts, shape {index_tensor.shape}")


# ── Query: find similar fonts ─────────────────────────────────────────────────

@app.function(image=image, gpu="T4", timeout=60, volumes={"/data": volume})
def find_similar(image_bytes: bytes, top_k: int = 10):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda")

    backbone = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    backbone.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls = backbone(pixel_values=tensor).last_hidden_state[:, 0]
        query_emb = F.normalize(cls, dim=1)

    index = torch.load("/data/font_index_base.pt", map_location="cpu")
    names = index["names"]
    embeddings = index["embeddings"]

    sims = torch.mm(query_emb.cpu(), embeddings.T).squeeze(0)
    sorted_indices = sims.argsort(descending=True)

    # Greedy dedup: skip fonts whose embedding is too close to an already-selected font
    DEDUP_THRESHOLD = 0.95
    selected_embs = []
    results = []
    for idx in sorted_indices.tolist():
        if len(results) >= top_k:
            break
        emb = embeddings[idx]
        if selected_embs:
            stacked = torch.stack(selected_embs)
            max_sim = torch.mm(emb.unsqueeze(0), stacked.T).max().item()
            if max_sim > DEDUP_THRESHOLD:
                continue
        selected_embs.append(emb)
        results.append({
            "rank": len(results) + 1,
            "font": names[idx],
            "similarity": round(sims[idx].item(), 4)
        })

    return results


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(command: str = "test", image_path: str = "", top_k: int = 10):
    """
    Usage:
      modal run inference_base.py --command build-index
      modal run inference_base.py --command test --image-path photo.jpg --top-k 10
    """
    if command == "build-index":
        build_index.remote()
        print("Base DINOv2 index built successfully.")

    elif command == "test":
        if not image_path:
            print("Provide --image-path, e.g.: modal run inference_base.py --command test --image-path photo.jpg")
            return
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        results = find_similar.remote(image_bytes, top_k=top_k)
        print(f"\nTop {top_k} similar fonts (base DINOv2):")
        print("-" * 50)
        for r in results:
            print(f"  #{r['rank']}  {r['font']}  (distance: {r['distance']})")

    else:
        print(f"Unknown command: {command}. Use 'build-index' or 'test'.")
