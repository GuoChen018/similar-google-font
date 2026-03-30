import modal

app = modal.App("font-finetune-v3")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers",
        "Pillow", "tqdm", "numpy", "requests", "peft"
    )
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)
secret = modal.Secret.from_name("google-fonts-key")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/data": volume},
    secrets=[secret],
)
def finetune():
    import glob, os, random, torch, requests
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader, Sampler
    from torch.amp import GradScaler, autocast
    from transformers import AutoModel, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    from PIL import Image, ImageDraw, ImageFont
    from torchvision import transforms
    from tqdm import tqdm
    from collections import defaultdict

    GLYPHS = ['a', 'g', 'R', 'Q', 'e', 'G', 'S', 'n', 'f', 't', '0', '&', 'M', 'W', 'O', 'I']
    GLYPH_DIR = "/data/glyph_images"
    os.makedirs(GLYPH_DIR, exist_ok=True)

    # ── 1. Fetch font tags with weights from Google Fonts API ─────────────────
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
    total_tags_parsed = 0
    total_tags_with_weight = 0

    for item in fonts_data:
        family = item["family"]
        family_category[family] = item.get("category", "")
        tags = item.get("tags", [])
        tag_dict = {}
        for t in tags:
            name = t.get("tag") or t.get("name") or ""
            weight = t.get("weight", 50)
            if name:
                tag_dict[name] = weight
                total_tags_parsed += 1
                if "weight" in t:
                    total_tags_with_weight += 1
        family_tags[family] = tag_dict

    print(f"Tag weight coverage: {total_tags_with_weight}/{total_tags_parsed} tags had explicit weights "
          f"({total_tags_parsed - total_tags_with_weight} defaulted to 50)")

    all_tag_names = sorted(set(
        tag for tags in family_tags.values() for tag in tags.keys()
    ))
    tag_to_idx = {t: i for i, t in enumerate(all_tag_names)}
    num_tags = len(all_tag_names)

    fonts_with_tags = sum(1 for tags in family_tags.values() if len(tags) > 0)
    avg_tags = np.mean([len(t) for t in family_tags.values() if len(t) > 0]) if fonts_with_tags else 0
    print(f"Tag vocabulary: {num_tags} unique tags")
    print(f"Fonts with tags: {fonts_with_tags} / {len(family_tags)} ({avg_tags:.1f} avg tags per font)")

    def get_tag_vector(family):
        tags = family_tags.get(family, {})
        vec = torch.zeros(num_tags)
        for tag_name, weight in tags.items():
            if tag_name in tag_to_idx:
                vec[tag_to_idx[tag_name]] = weight / 100.0
        return vec

    # ── 2. Filter font files (remove Guides, script variants, regional dupes) ─
    import re
    all_font_files = glob.glob("/data/google-fonts/**/*.ttf", recursive=True)
    print(f"Found {len(all_font_files)} total font files")

    SCRIPT_SUFFIXES = {
        'JP', 'KR', 'SC', 'TC', 'HK', 'Arabic', 'Hebrew', 'Thai', 'Bengali',
        'Devanagari', 'Tamil', 'Telugu', 'Kannada', 'Malayalam', 'Gujarati',
        'Gurmukhi', 'Oriya', 'Sinhala', 'Myanmar', 'Khmer', 'Lao', 'Georgian',
        'Armenian', 'Ethiopic', 'Cherokee', 'Tibetan', 'Mongolian', 'Symbols',
        'Math', 'Music', 'SignWriting',
    }
    NOTO_SCRIPT_RE = re.compile(r'NotoSans(' + '|'.join(SCRIPT_SUFFIXES) + r')')
    NOTO_SERIF_SCRIPT_RE = re.compile(r'NotoSerif(' + '|'.join(SCRIPT_SUFFIXES) + r')')

    REGIONAL_PREFIXES = {}

    def should_skip_font(font_path):
        name = os.path.basename(font_path).replace(".ttf", "")
        if "Guides" in name:
            return True
        if NOTO_SCRIPT_RE.search(name) or NOTO_SERIF_SCRIPT_RE.search(name):
            return True
        # Deduplicate regional super-families (Playwrite, etc.)
        # Keep the first variant encountered, skip the rest
        for prefix in ["Playwrite"]:
            if name.startswith(prefix):
                variant = name.split("-")[0].split("[")[0]
                if prefix not in REGIONAL_PREFIXES:
                    REGIONAL_PREFIXES[prefix] = variant
                if variant != REGIONAL_PREFIXES[prefix]:
                    return True
        return False

    font_files = [f for f in all_font_files if not should_skip_font(f)]
    skipped = len(all_font_files) - len(font_files)
    print(f"After filtering: {len(font_files)} font files ({skipped} removed: Guides, script variants, regional dupes)")

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
            rendered.append((font_name, out_path))

    print(f"Rendered {len(rendered)} glyph images")

    # ── 3. Group by font, filter ──────────────────────────────────────────────
    fontfile_to_paths = defaultdict(list)
    for font_name, path in rendered:
        fontfile_to_paths[font_name].append(path)

    font_files_list = [
        f for f in fontfile_to_paths.keys()
        if len(fontfile_to_paths[f]) >= 6
        and len(family_tags.get(fontfile_to_family.get(f, ""), {})) > 0
    ]
    print(f"{len(font_files_list)} usable font files (>= 6 glyphs, has tags)")

    fontfile_tag_vectors = {}
    for f in font_files_list:
        family = fontfile_to_family.get(f, "")
        fontfile_tag_vectors[f] = get_tag_vector(family)

    # ── 4. Train/val split by family ──────────────────────────────────────────
    family_to_fontfiles = defaultdict(list)
    for f in font_files_list:
        fam = fontfile_to_family.get(f, f)
        family_to_fontfiles[fam].append(f)

    families = list(family_to_fontfiles.keys())
    random.seed(42)
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

    # ── 5. Camera-robust augmentation (proportion-preserving) ─────────────────
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

    # ── 6. Dataset + category-aware PK batch sampler ──────────────────────────
    class FontDataset(Dataset):
        def __init__(self, fonts, fontfile_to_paths, fontfile_tag_vectors, transform):
            self.fonts = fonts
            self.fontfile_to_paths = fontfile_to_paths
            self.fontfile_tag_vectors = fontfile_tag_vectors
            self.transform = transform

        def __len__(self):
            return len(self.fonts)

        def __getitem__(self, idx):
            font_name = self.fonts[idx]
            path = random.choice(self.fontfile_to_paths[font_name])
            img = Image.open(path).convert("RGB")
            tag_vec = self.fontfile_tag_vectors[font_name]
            return self.transform(img), idx, tag_vec

    class CategoryPKSampler(Sampler):
        """P category groups x K fonts per group."""
        def __init__(self, fonts, fontfile_to_family, family_category,
                     p=16, k=4, batches_per_epoch=1500):
            self.p = p
            self.k = k
            self.batches_per_epoch = batches_per_epoch

            self.cat_to_indices = defaultdict(list)
            for i, f in enumerate(fonts):
                fam = fontfile_to_family.get(f, "")
                cat = family_category.get(fam, "_unknown")
                self.cat_to_indices[cat].append(i)

            self.valid_cats = [c for c, idxs in self.cat_to_indices.items()
                               if len(idxs) >= k]
            if not self.valid_cats:
                self.valid_cats = list(self.cat_to_indices.keys())

        def __iter__(self):
            for _ in range(self.batches_per_epoch):
                chosen = random.choices(self.valid_cats, k=self.p)
                batch = []
                for cat in chosen:
                    indices = self.cat_to_indices[cat]
                    if len(indices) >= self.k:
                        batch.extend(random.sample(indices, self.k))
                    else:
                        batch.extend(random.choices(indices, k=self.k))
                yield batch

        def __len__(self):
            return self.batches_per_epoch

    # ── 7. Model: DINOv2 + LoRA + projection head + tag classifier ───────────
    class FontSimilarityModel(nn.Module):
        def __init__(self, backbone, num_tags):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(768, 512),
                nn.GELU(),
                nn.Linear(512, 256),
            )
            self.tag_classifier = nn.Linear(768, num_tags)

        def forward(self, pixel_values):
            cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
            embedding = F.normalize(self.head(cls), dim=1)
            tag_logits = self.tag_classifier(cls)
            return embedding, tag_logits

        def embed(self, pixel_values):
            cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
            return F.normalize(self.head(cls), dim=1)

    backbone = AutoModel.from_pretrained("facebook/dinov2-base")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    device = torch.device("cuda")
    model = FontSimilarityModel(backbone, num_tags).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} params")

    # ── 8. Loss functions ─────────────────────────────────────────────────────
    _supcon_log_counter = [0]

    def tag_similarity_supcon(embeddings, tag_vectors, temperature=0.07):
        tag_norms = F.normalize(tag_vectors, dim=1)
        tag_sim = torch.mm(tag_norms, tag_norms.T)

        pos_mask = (tag_sim > 0.65).float()
        neg_mask = (tag_sim < 0.40).float()
        eye = torch.eye(len(embeddings), device=embeddings.device)
        pos_mask = pos_mask - eye

        num_pos = pos_mask.sum(dim=1)

        if _supcon_log_counter[0] < 5:
            n = len(embeddings)
            total_pairs = n * (n - 1)
            n_pos = pos_mask.sum().item()
            n_neg = neg_mask.sum().item()
            n_ambig = total_pairs - n_pos - max(n_neg, 0)
            print(f"  [batch {_supcon_log_counter[0]}] tag_sim stats: "
                  f"mean={tag_sim.mean().item():.3f} median={tag_sim.median().item():.3f} | "
                  f"pos={n_pos:.0f} neg={max(n_neg,0):.0f} ambig={n_ambig:.0f} / {total_pairs:.0f} pairs")
            _supcon_log_counter[0] += 1

        if (num_pos > 0).sum() == 0:
            print("WARNING: No positive pairs in batch — SupCon loss is zero (no gradient)")
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        sim_mat = torch.mm(embeddings, embeddings.T) / temperature

        valid_mask = (pos_mask + neg_mask).clamp(max=1)
        masked_sim = sim_mat - (1 - valid_mask) * 1e4 - eye * 1e4
        log_sum_exp = torch.logsumexp(masked_sim, dim=1)

        pos_sims = sim_mat * pos_mask
        mean_pos_sim = pos_sims.sum(dim=1) / num_pos.clamp(min=1)

        loss = log_sum_exp - mean_pos_sim
        valid = num_pos > 0
        return loss[valid].mean()

    tag_pos_counts = torch.zeros(num_tags)
    for f in train_fonts:
        tv = fontfile_tag_vectors[f]
        tag_pos_counts += (tv > 0).float()
    tag_neg_counts = len(train_fonts) - tag_pos_counts
    bce_pos_weight = (tag_neg_counts / tag_pos_counts.clamp(min=1)).clamp(max=20).to(device)
    print(f"BCE pos_weight range: {bce_pos_weight.min().item():.1f} - {bce_pos_weight.max().item():.1f}")

    def multilabel_tag_bce(tag_logits, tag_vectors):
        targets = (tag_vectors > 0).float()
        return F.binary_cross_entropy_with_logits(tag_logits, targets, pos_weight=bce_pos_weight)

    # ── 9. Dataloaders ────────────────────────────────────────────────────────
    def worker_init_fn(worker_id):
        random.seed(random.getrandbits(32) + worker_id)
        np.random.seed(random.getrandbits(32) + worker_id)

    train_dataset = FontDataset(train_fonts, fontfile_to_paths, fontfile_tag_vectors, augment)
    val_dataset = FontDataset(val_fonts, fontfile_to_paths, fontfile_tag_vectors, clean)

    train_sampler = CategoryPKSampler(
        train_fonts, fontfile_to_family, family_category,
        p=16, k=4, batches_per_epoch=1500
    )
    val_sampler = CategoryPKSampler(
        val_fonts, fontfile_to_family, family_category,
        p=16, k=4, batches_per_epoch=200
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn
    )

    # ── 10. Optimizer + scheduler ─────────────────────────────────────────────
    EPOCHS = 20
    steps_per_epoch = len(train_sampler)
    total_steps = EPOCHS * steps_per_epoch

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=total_steps
    )
    scaler = GradScaler("cuda")

    # ── 11. Evaluation ────────────────────────────────────────────────────────
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
    def evaluate(model, fonts, fontfile_to_paths, fontfile_tag_vectors):
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
            embs = model.embed(batch.to(device))
            all_embs.append(embs.cpu())
        all_embs = torch.cat(all_embs)

        font_idx_tensor = torch.tensor(all_font_idx)
        font_embs = []
        font_tag_vecs = []
        for fidx in range(len(fonts)):
            mask = font_idx_tensor == fidx
            avg = all_embs[mask].mean(dim=0)
            font_embs.append(F.normalize(avg, dim=0))
            font_tag_vecs.append(fontfile_tag_vectors[fonts[fidx]])
        font_embs = torch.stack(font_embs)
        font_tag_vecs = torch.stack(font_tag_vecs)

        dist_mat = torch.cdist(font_embs, font_embs, p=2)
        dist_mat.fill_diagonal_(float('inf'))
        topk = dist_mat.topk(5, largest=False)
        nn_indices = topk.indices

        cat_hits = 0
        for i in range(len(fonts)):
            fam_i = fontfile_to_family.get(fonts[i], "")
            cat_i = family_category.get(fam_i, "")
            fam_nn = fontfile_to_family.get(fonts[nn_indices[i][0]], "")
            cat_nn = family_category.get(fam_nn, "")
            if cat_i == cat_nn and cat_i:
                cat_hits += 1
        cat_acc = cat_hits / len(fonts) if fonts else 0

        tag_overlaps = []
        for i in range(len(fonts)):
            tags_i = set(j for j in range(num_tags) if font_tag_vecs[i][j] > 0)
            if not tags_i:
                continue
            overlaps = []
            for k_idx in range(5):
                nn_idx = nn_indices[i][k_idx].item()
                tags_nn = set(j for j in range(num_tags) if font_tag_vecs[nn_idx][j] > 0)
                union = tags_i | tags_nn
                if union:
                    overlaps.append(len(tags_i & tags_nn) / len(union))
            if overlaps:
                tag_overlaps.append(sum(overlaps) / len(overlaps))
        avg_tag_overlap = sum(tag_overlaps) / len(tag_overlaps) if tag_overlaps else 0

        known = ["Roboto-Bold", "PlayfairDisplay-Regular", "CourierPrime-Regular"]
        for kf in known:
            if kf in fonts:
                idx = fonts.index(kf)
                neighbors = [fonts[nn_indices[idx][j]] for j in range(3)]
                print(f"    {kf} -> {neighbors}")

        return {"cat_acc": cat_acc, "tag_overlap@5": avg_tag_overlap}

    # ── 12. Training loop ─────────────────────────────────────────────────────
    best_tag_overlap = 0.0
    patience_counter = 0
    PATIENCE = 7

    for epoch in range(EPOCHS):
        model.train()
        epoch_supcon = 0.0
        epoch_bce = 0.0
        num_batches = 0

        for images, indices, tag_vecs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train"):
            images = images.to(device)
            tag_vecs = tag_vecs.to(device)

            with autocast("cuda"):
                embeddings, tag_logits = model(pixel_values=images)
                loss_supcon = tag_similarity_supcon(embeddings, tag_vecs)
                loss_bce = multilabel_tag_bce(tag_logits, tag_vecs)
                loss = loss_supcon + 0.5 * loss_bce

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_supcon += loss_supcon.item()
            epoch_bce += loss_bce.item()
            num_batches += 1

        avg_supcon = epoch_supcon / max(num_batches, 1)
        avg_bce = epoch_bce / max(num_batches, 1)

        model.eval()
        val_supcon = 0.0
        val_bce = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, indices, tag_vecs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} val"):
                images = images.to(device)
                tag_vecs = tag_vecs.to(device)
                with autocast("cuda"):
                    embeddings, tag_logits = model(pixel_values=images)
                    loss_supcon = tag_similarity_supcon(embeddings, tag_vecs)
                    loss_bce = multilabel_tag_bce(tag_logits, tag_vecs)
                val_supcon += loss_supcon.item()
                val_bce += loss_bce.item()
                val_batches += 1

        avg_val_supcon = val_supcon / max(val_batches, 1)
        avg_val_bce = val_bce / max(val_batches, 1)
        avg_val_loss = avg_val_supcon + 0.5 * avg_val_bce

        metrics = evaluate(model, val_fonts, fontfile_to_paths, fontfile_tag_vectors)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} — supcon: {avg_supcon:.4f} | bce: {avg_bce:.4f} | "
              f"val_supcon: {avg_val_supcon:.4f} | val_bce: {avg_val_bce:.4f} | "
              f"cat_acc: {metrics['cat_acc']:.4f} | tag@5: {metrics['tag_overlap@5']:.4f} | "
              f"lr: {lr:.2e}")

        tag_overlap = metrics["tag_overlap@5"]
        if tag_overlap > best_tag_overlap:
            best_tag_overlap = tag_overlap
            patience_counter = 0
            torch.save(model.state_dict(), "/data/font_v3_best_full.pt")
            volume.commit()
            print(f"  -> Best model saved (tag@5: {best_tag_overlap:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best tag@5: {best_tag_overlap:.4f})")
                break

        torch.save(model.state_dict(), f"/data/checkpoint_v3_epoch{epoch+1}.pt")
        volume.commit()

    # ── 13. Merge LoRA and save inference model ───────────────────────────────
    print("Merging LoRA weights for inference...")
    best_state = torch.load("/data/font_v3_best_full.pt", map_location=device)
    model.load_state_dict(best_state)

    model.backbone = model.backbone.merge_and_unload()

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

    inference_model = FontEmbedder(model.backbone).to(device)
    inference_model.head.load_state_dict(model.head.state_dict())

    torch.save(inference_model.state_dict(), "/data/font_embedder_v3.pt")
    volume.commit()
    print(f"Inference model saved to /data/font_embedder_v3.pt")
    print(f"Done. Best tag@5: {best_tag_overlap:.4f}")

@app.local_entrypoint()
def main():
    finetune.remote()
