import modal

app = modal.App("font-finetune")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers",
        "Pillow", "tqdm", "numpy", "requests"
    )
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)
secret = modal.Secret.from_name("google-fonts-key")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/data": volume},
    secrets=[secret]
)
def finetune():
    import glob, os, random, torch, requests
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, Sampler
    from torch.amp import GradScaler, autocast
    from transformers import AutoModel, get_cosine_schedule_with_warmup
    from PIL import Image, ImageDraw, ImageFont
    from torchvision import transforms
    from tqdm import tqdm
    from collections import defaultdict

    GLYPHS = ['a', 'g', 'R', 'Q', 'e', 'G', 'S', 'n', 'f', 't', '0', '&']
    GLYPH_DIR = "/data/glyph_images"
    os.makedirs(GLYPH_DIR, exist_ok=True)

    # ── 1. Fetch font tags from Google Fonts API ──────────────────────────────
    api_key = os.environ["GOOGLE_FONTS_API_KEY"]
    url = (
        f"https://www.googleapis.com/webfonts/v1/webfonts"
        f"?key={api_key}&capability=FAMILY_TAGS&fields=items(family,category,tags)"
    )
    resp = requests.get(url)
    fonts_data = resp.json().get("items", [])
    print(f"Fetched {len(fonts_data)} font families from API")

    family_tags = {}
    family_category = {}

    for item in fonts_data:
        family = item["family"]
        family_category[family] = item.get("category", "")
        tags = item.get("tags", [])
        tag_names = []
        for t in tags:
            name = t.get("tag") or t.get("name") or ""
            if name:
                tag_names.append(name)
        family_tags[family] = tag_names

    all_tags = set()
    for tags in family_tags.values():
        all_tags.update(tags)
    print(f"Tags found: {len(all_tags)} unique tags")

    # ── 2. Render glyphs (all weights) ───────────────────────────────────────
    font_files = glob.glob("/data/google-fonts/**/*.ttf", recursive=True)
    print(f"Found {len(font_files)} total font files")

    def filename_to_family(font_path):
        basename = os.path.basename(font_path).replace(".ttf", "")
        clean_basename = basename.split("-")[0].split("[")[0].replace(" ", "").lower()
        for family in family_tags.keys():
            if family.replace(" ", "").lower() == clean_basename:
                return family
        for family in family_tags.keys():
            if clean_basename.startswith(family.replace(" ", "").lower()):
                return family
        return basename.split("-")[0].split("[")[0]

    rendered = []
    fontfile_to_family = {}

    for font_path in tqdm(font_files, desc="Rendering glyphs"):
        font_name = os.path.basename(font_path).replace(".ttf", "")
        family = filename_to_family(font_path)
        fontfile_to_family[font_name] = family

        for glyph in GLYPHS:
            out_path = f"{GLYPH_DIR}/{font_name}_{glyph}.png"
            if not os.path.exists(out_path):
                try:
                    img = Image.new("RGB", (224, 224), "white")
                    draw = ImageDraw.Draw(img)
                    font = ImageFont.truetype(font_path, 160)
                    bbox = draw.textbbox((0, 0), glyph, font=font)
                    x = (224 - (bbox[2] - bbox[0])) // 2 - bbox[0]
                    y = (224 - (bbox[3] - bbox[1])) // 2 - bbox[1]
                    draw.text((x, y), glyph, font=font, fill="black")
                    img.save(out_path)
                except Exception:
                    continue
            rendered.append((font_name, family, out_path))

    print(f"Rendered {len(rendered)} glyph images")

    # ── 3. Group by font file + assign primary tag ────────────────────────────
    fontfile_to_paths = defaultdict(list)
    for font_name, family, path in rendered:
        fontfile_to_paths[font_name].append(path)

    def get_primary_tag(font_name):
        fam = fontfile_to_family.get(font_name, "")
        tags = family_tags.get(fam, [])
        if tags:
            return tags[0]
        cat = family_category.get(fam, "")
        return f"_cat_{cat}" if cat else "_unknown"

    font_files_list = [f for f in fontfile_to_paths.keys()
                       if len(fontfile_to_paths[f]) >= 2]
    print(f"{len(font_files_list)} usable font files")

    # ── 4. Split by font family ───────────────────────────────────────────────
    family_to_fontfiles = defaultdict(list)
    for f in font_files_list:
        fam = fontfile_to_family.get(f, f)
        family_to_fontfiles[fam].append(f)

    families = list(family_to_fontfiles.keys())
    random.shuffle(families)
    split = int(len(families) * 0.9)
    train_families = set(families[:split])
    val_families = set(families[split:])

    train_fonts = [f for f in font_files_list
                   if fontfile_to_family.get(f, f) in train_families]
    val_fonts = [f for f in font_files_list
                 if fontfile_to_family.get(f, f) in val_families]
    print(f"Train: {len(train_fonts)} files ({len(train_families)} families), "
          f"Val: {len(val_fonts)} files ({len(val_families)} families)")

    # ── 5. Camera-robust augmentation ─────────────────────────────────────────
    class RandomBackground:
        def __call__(self, img):
            r, g, b = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
            bg = Image.new("RGB", img.size, (r, g, b))
            arr = np.array(img)
            gray = np.mean(arr, axis=2)
            mask = (gray < 240).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask, mode="L")
            bg.paste(img, mask=mask_img)
            return bg

    class GaussianNoise:
        def __init__(self, max_sigma=0.05):
            self.max_sigma = max_sigma

        def __call__(self, tensor):
            sigma = random.uniform(0, self.max_sigma)
            return tensor + torch.randn_like(tensor) * sigma

    augment = transforms.Compose([
        RandomBackground(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomInvert(p=0.15),
        transforms.ToTensor(),
        GaussianNoise(max_sigma=0.05),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── 6. PK dataset + batch sampler ─────────────────────────────────────────
    class FontDataset(Dataset):
        def __init__(self, fonts, fontfile_to_paths, transform):
            self.fonts = fonts
            self.fontfile_to_paths = fontfile_to_paths
            self.transform = transform

        def __len__(self):
            return len(self.fonts)

        def __getitem__(self, idx):
            font_name = self.fonts[idx]
            path = random.choice(self.fontfile_to_paths[font_name])
            img = Image.open(path).convert("RGB")
            return self.transform(img), idx

    class PKBatchSampler(Sampler):
        """Yields batches of P primary-tags x K font files per tag."""
        def __init__(self, fonts, get_primary_tag_fn, p=8, k=4, batches_per_epoch=1500):
            self.p = p
            self.k = k
            self.batches_per_epoch = batches_per_epoch

            self.tag_to_indices = defaultdict(list)
            for i, f in enumerate(fonts):
                tag = get_primary_tag_fn(f)
                self.tag_to_indices[tag].append(i)

            self.valid_tags = [t for t, idxs in self.tag_to_indices.items()
                               if len(idxs) >= k]
            if len(self.valid_tags) < p:
                self.valid_tags = list(self.tag_to_indices.keys())

        def __iter__(self):
            for _ in range(self.batches_per_epoch):
                if len(self.valid_tags) >= self.p:
                    chosen_tags = random.sample(self.valid_tags, self.p)
                else:
                    chosen_tags = random.choices(self.valid_tags, k=self.p)
                batch = []
                for tag in chosen_tags:
                    indices = self.tag_to_indices[tag]
                    if len(indices) >= self.k:
                        batch.extend(random.sample(indices, self.k))
                    else:
                        batch.extend(random.choices(indices, k=self.k))
                yield batch

        def __len__(self):
            return self.batches_per_epoch

    def get_tag_labels(fonts):
        """Build a tensor mapping font index -> primary tag index."""
        tag_set = sorted(set(get_primary_tag(f) for f in fonts))
        tag_to_idx = {t: i for i, t in enumerate(tag_set)}
        labels = torch.tensor([tag_to_idx[get_primary_tag(f)] for f in fonts])
        return labels, tag_to_idx

    train_labels, train_tag_to_idx = get_tag_labels(train_fonts)
    val_labels, _ = get_tag_labels(val_fonts)

    # ── 7. FontEmbedder: DINOv2 + projection head ────────────────────────────
    class FontEmbedder(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(768, 512),
                nn.GELU(),
                nn.Linear(512, 256),
            )

        def forward(self, pixel_values):
            cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
            return F.normalize(self.head(cls), dim=1)

    backbone = AutoModel.from_pretrained("facebook/dinov2-base")
    device = torch.device("cuda")

    for param in backbone.parameters():
        param.requires_grad = False

    total_blocks = len(backbone.encoder.layer)
    for i, block in enumerate(backbone.encoder.layer):
        if i >= total_blocks - 4:
            for param in block.parameters():
                param.requires_grad = True

    for param in backbone.layernorm.parameters():
        param.requires_grad = True

    model = FontEmbedder(backbone).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params")

    # ── 8. Semi-hard triplet mining ──────────────────────────────────────────
    def semi_hard_triplet_loss(embeddings, labels, margin=0.5):
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        labels_col = labels.unsqueeze(0)
        same_mask = (labels_col == labels_col.T)
        diff_mask = ~same_mask
        eye = torch.eye(len(embeddings), device=embeddings.device, dtype=torch.bool)
        pos_mask = same_mask & ~eye

        pos_dists = dist_mat.clone()
        pos_dists[~pos_mask] = -1.0
        hardest_pos_dist = pos_dists.max(dim=1).values

        losses = []
        for i in range(len(embeddings)):
            ap_dist = hardest_pos_dist[i]
            neg_dists = dist_mat[i][diff_mask[i]]
            if len(neg_dists) == 0:
                continue
            semi_hard = neg_dists[(neg_dists > ap_dist) & (neg_dists < ap_dist + margin)]
            if len(semi_hard) > 0:
                an_dist = semi_hard.min()
            else:
                an_dist = neg_dists.min()
            loss = F.relu(ap_dist - an_dist + margin)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return torch.stack(losses).mean()

    # ── 9. Dataloaders ────────────────────────────────────────────────────────
    def worker_init_fn(worker_id):
        random.seed(random.getrandbits(32) + worker_id)
        np.random.seed(random.getrandbits(32) + worker_id)

    train_dataset = FontDataset(train_fonts, fontfile_to_paths, transform=augment)
    val_dataset = FontDataset(val_fonts, fontfile_to_paths, transform=clean)

    train_sampler = PKBatchSampler(
        train_fonts, get_primary_tag, p=12, k=4, batches_per_epoch=1500
    )
    val_sampler = PKBatchSampler(
        val_fonts, get_primary_tag, p=12, k=4, batches_per_epoch=200
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )

    # ── 10. Optimizer + scheduler ─────────────────────────────────────────────
    EPOCHS = 30
    steps_per_epoch = len(train_sampler)
    total_steps = EPOCHS * steps_per_epoch

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    scaler = GradScaler("cuda")

    # ── 11. Recall@k evaluation (multi-glyph average per font) ─────────────
    class GlyphEvalDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    @torch.no_grad()
    def compute_recall_at_k(model, fonts, fontfile_to_paths, labels, ks=(1, 5)):
        model.eval()
        all_paths = []
        all_font_idx = []
        for fidx, font_name in enumerate(fonts):
            for p in fontfile_to_paths[font_name]:
                all_paths.append(p)
                all_font_idx.append(fidx)

        eval_dataset = GlyphEvalDataset(all_paths, clean)
        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False,
                                 num_workers=4, pin_memory=True)

        all_embs = []
        for batch in eval_loader:
            embs = model(pixel_values=batch.to(device))
            all_embs.append(embs.cpu())
        all_embs = torch.cat(all_embs)

        font_idx_tensor = torch.tensor(all_font_idx)
        font_embs = []
        for fidx in range(len(fonts)):
            mask = font_idx_tensor == fidx
            avg = all_embs[mask].mean(dim=0)
            font_embs.append(F.normalize(avg, dim=0))
        font_embs = torch.stack(font_embs)

        dist_mat = torch.cdist(font_embs, font_embs, p=2)
        dist_mat.fill_diagonal_(float('inf'))

        results = {}
        for k in ks:
            topk_indices = dist_mat.topk(k, largest=False).indices
            topk_labels = labels[topk_indices]
            query_labels = labels.unsqueeze(1).expand_as(topk_labels)
            hits = (topk_labels == query_labels).any(dim=1).float()
            results[f"R@{k}"] = hits.mean().item()
        return results

    # ── 12. Training loop ─────────────────────────────────────────────────────
    best_recall = 0.0
    patience_counter = 0
    PATIENCE = 7
    train_labels_device = train_labels.to(device)
    val_labels_device = val_labels.to(device)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train"):
            images = images.to(device)
            batch_labels = train_labels_device[indices]

            with autocast("cuda"):
                embeddings = model(pixel_values=images)
                loss = semi_hard_triplet_loss(embeddings, batch_labels, margin=0.5)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation loss
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, indices in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} val"):
                images = images.to(device)
                batch_labels = val_labels_device[indices]
                with autocast("cuda"):
                    embeddings = model(pixel_values=images)
                    loss = semi_hard_triplet_loss(embeddings, batch_labels, margin=0.5)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        recall = compute_recall_at_k(
            model, val_fonts, fontfile_to_paths, val_labels, ks=(1, 5)
        )
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} — train_loss: {avg_train_loss:.4f} | "
              f"val_loss: {avg_val_loss:.4f} | "
              f"R@1: {recall['R@1']:.4f} | R@5: {recall['R@5']:.4f} | "
              f"lr: {lr:.2e}")

        if recall["R@1"] > best_recall:
            best_recall = recall["R@1"]
            patience_counter = 0
            torch.save(model.state_dict(), "/data/font_embedder_best.pt")
            volume.commit()
            print(f"  -> Best model saved (R@1: {best_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping: R@1 hasn't improved for {PATIENCE} epochs")
                break

        torch.save(model.state_dict(), f"/data/checkpoint_epoch{epoch+1}.pt")
        volume.commit()

    print(f"Done. Best R@1: {best_recall:.4f}")

@app.local_entrypoint()
def main():
    finetune.remote()
