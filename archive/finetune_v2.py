import modal

app = modal.App("font-finetune-v2")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers",
        "Pillow", "tqdm", "numpy"
    )
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/data": volume},
)
def finetune():
    import glob, os, random, torch
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

    GLYPHS = ['a', 'g', 'R', 'Q', 'e', 'G', 'S', 'n', 'f', 't', '0', '&', 'M', 'W', 'O', 'I']
    GLYPH_DIR = "/data/glyph_images"
    os.makedirs(GLYPH_DIR, exist_ok=True)

    # ── 1. Render glyphs (all weights) ───────────────────────────────────────
    font_files = glob.glob("/data/google-fonts/**/*.ttf", recursive=True)
    print(f"Found {len(font_files)} total font files")

    rendered = []

    for font_path in tqdm(font_files, desc="Rendering glyphs"):
        font_name = os.path.basename(font_path).replace(".ttf", "")

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
            rendered.append((font_name, out_path))

    print(f"Rendered {len(rendered)} glyph images")

    # ── 2. Group by font file ─────────────────────────────────────────────────
    fontfile_to_paths = defaultdict(list)
    for font_name, path in rendered:
        fontfile_to_paths[font_name].append(path)

    font_files_list = [f for f in fontfile_to_paths.keys()
                       if len(fontfile_to_paths[f]) >= 6]
    print(f"{len(font_files_list)} usable font files (>= 6 glyphs)")

    # ── 3. Train/val split ────────────────────────────────────────────────────
    random.shuffle(font_files_list)
    split = int(len(font_files_list) * 0.9)
    train_fonts = font_files_list[:split]
    val_fonts = font_files_list[split:]
    print(f"Train: {len(train_fonts)}, Val: {len(val_fonts)}")

    # ── 4. Camera-robust augmentation ─────────────────────────────────────────
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
        transforms.RandomRotation(5),
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

    # ── 5. PK dataset: P fonts x K glyphs per font ───────────────────────────
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
        """P random fonts x K glyphs each. Label = font index."""
        def __init__(self, fonts, p=16, k=4, batches_per_epoch=1500):
            self.fonts = fonts
            self.p = p
            self.k = k
            self.batches_per_epoch = batches_per_epoch

        def __iter__(self):
            for _ in range(self.batches_per_epoch):
                if len(self.fonts) >= self.p:
                    chosen = random.sample(range(len(self.fonts)), self.p)
                else:
                    chosen = random.choices(range(len(self.fonts)), k=self.p)
                batch = []
                for idx in chosen:
                    for _ in range(self.k):
                        batch.append(idx)
                yield batch

        def __len__(self):
            return self.batches_per_epoch

    # ── 6. FontEmbedder: DINOv2 + projection head ────────────────────────────
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
        if i >= total_blocks - 6:
            for param in block.parameters():
                param.requires_grad = True

    for param in backbone.layernorm.parameters():
        param.requires_grad = True

    model = FontEmbedder(backbone).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params")

    # ── 7. Supervised contrastive loss (label = font file) ────────────────────
    def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
        """All same-font pairs are positive, all different-font pairs are negative."""
        sim_mat = torch.mm(embeddings, embeddings.T) / temperature
        labels_col = labels.unsqueeze(0)
        pos_mask = (labels_col == labels_col.T).float()
        eye = torch.eye(len(embeddings), device=embeddings.device)
        pos_mask = pos_mask - eye

        log_sum_exp = torch.logsumexp(sim_mat - eye * 1e9, dim=1)

        pos_sims = sim_mat * pos_mask
        num_pos = pos_mask.sum(dim=1).clamp(min=1)
        mean_pos_sim = pos_sims.sum(dim=1) / num_pos

        loss = log_sum_exp - mean_pos_sim
        valid = num_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return loss[valid].mean()

    # ── 8. Dataloaders ────────────────────────────────────────────────────────
    def worker_init_fn(worker_id):
        random.seed(random.getrandbits(32) + worker_id)
        np.random.seed(random.getrandbits(32) + worker_id)

    train_dataset = FontDataset(train_fonts, fontfile_to_paths, transform=augment)
    val_dataset = FontDataset(val_fonts, fontfile_to_paths, transform=clean)

    train_sampler = PKBatchSampler(train_fonts, p=16, k=4, batches_per_epoch=1500)
    val_sampler = PKBatchSampler(val_fonts, p=16, k=4, batches_per_epoch=200)

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )

    # ── 9. Optimizer + scheduler ──────────────────────────────────────────────
    EPOCHS = 20
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

    # ── 10. Evaluation: font retrieval ────────────────────────────────────────
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
    def evaluate(model, fonts, fontfile_to_paths):
        """
        For each val font, embed all glyphs, average, find nearest neighbor.
        Report what % of nearest neighbors are visually meaningful
        (same font family prefix as a proxy).
        """
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

        topk = dist_mat.topk(5, largest=False)
        nn_indices = topk.indices

        def font_base(name):
            return name.split("-")[0].split("[")[0].replace(" ", "").lower()

        family_hits_1 = 0
        family_hits_5 = 0
        for i in range(len(fonts)):
            base = font_base(fonts[i])
            nn1_base = font_base(fonts[nn_indices[i][0]])
            if base == nn1_base:
                family_hits_1 += 1
            for j in range(5):
                nnj_base = font_base(fonts[nn_indices[i][j]])
                if base == nnj_base:
                    family_hits_5 += 1
                    break

        fam_r1 = family_hits_1 / len(fonts)
        fam_r5 = family_hits_5 / len(fonts)

        avg_nn_dist = topk.values[:, 0].mean().item()
        avg_5nn_dist = topk.values.mean().item()

        return {
            "family_R@1": fam_r1,
            "family_R@5": fam_r5,
            "avg_nn_dist": avg_nn_dist,
            "avg_5nn_dist": avg_5nn_dist,
        }

    # ── 11. Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 5

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train"):
            images = images.to(device)
            font_labels = indices.to(device)

            with autocast("cuda"):
                embeddings = model(pixel_values=images)
                loss = supervised_contrastive_loss(embeddings, font_labels, temperature=0.07)

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

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, indices in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} val"):
                images = images.to(device)
                font_labels = indices.to(device)
                with autocast("cuda"):
                    embeddings = model(pixel_values=images)
                    loss = supervised_contrastive_loss(embeddings, font_labels, temperature=0.07)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        metrics = evaluate(model, val_fonts, fontfile_to_paths)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} — train_loss: {avg_train_loss:.4f} | "
              f"val_loss: {avg_val_loss:.4f} | "
              f"fam_R@1: {metrics['family_R@1']:.4f} | "
              f"fam_R@5: {metrics['family_R@5']:.4f} | "
              f"avg_nn_dist: {metrics['avg_nn_dist']:.4f} | "
              f"lr: {lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "/data/font_embedder_best.pt")
            volume.commit()
            print(f"  -> Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping: val_loss hasn't improved for {PATIENCE} epochs")
                break

        torch.save(model.state_dict(), f"/data/checkpoint_epoch{epoch+1}.pt")
        volume.commit()

    print(f"Done. Best val_loss: {best_val_loss:.4f}")

@app.local_entrypoint()
def main():
    finetune.remote()
