import modal

app = modal.App("font-finetune-v7")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers",
        "Pillow", "tqdm", "numpy", "requests", "peft"
    )
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=18000,
    volumes={"/data": volume},
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
    TEXT_STRINGS = [
        "Hamburgevons", "Handgloves",
        "AB", "OF", "AC", "THE", "MR", "GO", "NYC",
        "Quick", "Gazing", "Sphinx", "Rhythm",
        "QUALITY", "DESIGN",
        "typography", "elegant",
        "0123456789", "No. 42",
        "Aa Bb Gg", "Quick fox",
    ]
    GLYPH_DIR = "/data/glyph_images"
    os.makedirs(GLYPH_DIR, exist_ok=True)

    # ── 1. Fetch font categories from Google Fonts API ────────────────────────
    api_key = os.environ.get("GOOGLE_FONTS_API_KEY", "")
    family_category = {}
    if api_key:
        url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}&fields=items(family,category)"
        resp = requests.get(url)
        for item in resp.json().get("items", []):
            family_category[item["family"]] = item.get("category", "")
        print(f"Fetched categories for {len(family_category)} font families")
    else:
        print("No API key — cat_acc metric will be empty")

    # ── 2. Filter font files ──────────────────────────────────────────────────
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
    print(f"After filtering: {len(font_files)} font files ({skipped} removed)")

    def filename_to_family(font_path):
        basename = os.path.basename(font_path).replace(".ttf", "")
        clean_basename = basename.split("-")[0].split("[")[0].replace(" ", "").lower()
        for family in family_category.keys():
            if family.replace(" ", "").lower() == clean_basename:
                return family
        for family in family_category.keys():
            if clean_basename.startswith(family.replace(" ", "").lower()):
                return family
        return basename.split("-")[0].split("[")[0]

    def render_text_image(font_path, text, size=224):
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

    # ── 3. Render glyphs + text strings ───────────────────────────────────────
    rendered = []
    fontfile_to_family = {}
    fontfile_to_fontpath = {}

    for font_path in tqdm(font_files, desc="Rendering glyphs + text"):
        font_name = os.path.basename(font_path).replace(".ttf", "")
        family = filename_to_family(font_path)
        fontfile_to_family[font_name] = family
        fontfile_to_fontpath[font_name] = font_path

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

        for text in TEXT_STRINGS:
            safe_text = text.replace(" ", "_").replace(".", "_")
            out_path = f"{GLYPH_DIR}/{font_name}_text_{safe_text}.png"
            if not os.path.exists(out_path):
                img = render_text_image(font_path, text)
                if img is None:
                    continue
                img.save(out_path)
            if os.path.exists(out_path):
                rendered.append((font_name, out_path))

        combo = "".join(random.sample(GLYPHS, min(5, len(GLYPHS))))
        combo_path = f"{GLYPH_DIR}/{font_name}_text_combo.png"
        if not os.path.exists(combo_path):
            img = render_text_image(font_path, combo)
            if img:
                img.save(combo_path)
        if os.path.exists(combo_path):
            rendered.append((font_name, combo_path))

    print(f"Rendered {len(rendered)} images (glyphs + text strings)")

    # ── 4. Group by font, filter ──────────────────────────────────────────────
    fontfile_to_paths = defaultdict(list)
    for font_name, path in rendered:
        fontfile_to_paths[font_name].append(path)

    font_files_list = [
        f for f in fontfile_to_paths.keys()
        if len(fontfile_to_paths[f]) >= 6
    ]
    print(f"{len(font_files_list)} usable font files (>= 6 images)")

    # ── 5. Train/val split by family ──────────────────────────────────────────
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

    # ── 6. Transform: clean only (no augmentation) ────────────────────────────
    clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── 7. Dataset + family-aware PK batch sampler ────────────────────────────
    family_name_to_id = {}
    font_family_id_map = {}
    for f in font_files_list:
        fam = fontfile_to_family.get(f, f)
        if fam not in family_name_to_id:
            family_name_to_id[fam] = len(family_name_to_id)
        font_family_id_map[f] = family_name_to_id[fam]

    class FontDataset(Dataset):
        def __init__(self, fonts, fontfile_to_paths, font_family_id_map, transform):
            self.fonts = fonts
            self.fontfile_to_paths = fontfile_to_paths
            self.font_family_id_map = font_family_id_map
            self.transform = transform

        def __len__(self):
            return len(self.fonts)

        def __getitem__(self, idx):
            font_name = self.fonts[idx]
            paths = self.fontfile_to_paths[font_name]
            p1, p2 = random.choices(paths, k=2)
            img1 = Image.open(p1).convert("RGB")
            img2 = Image.open(p2).convert("RGB")
            family_id = self.font_family_id_map[font_name]
            return self.transform(img1), self.transform(img2), idx, family_id

    class FamilyPKSampler(Sampler):
        def __init__(self, fonts, fontfile_to_family, p=24, k=3,
                     batches_per_epoch=1500):
            self.p = p
            self.k = k
            self.batches_per_epoch = batches_per_epoch

            self.family_to_indices = defaultdict(list)
            for i, f in enumerate(fonts):
                fam = fontfile_to_family.get(f, f)
                self.family_to_indices[fam].append(i)

            self.valid_families = list(self.family_to_indices.keys())

        def __iter__(self):
            for _ in range(self.batches_per_epoch):
                chosen = random.sample(
                    self.valid_families,
                    min(self.p, len(self.valid_families))
                )
                batch = []
                for fam in chosen:
                    indices = self.family_to_indices[fam]
                    if len(indices) >= self.k:
                        batch.extend(random.sample(indices, self.k))
                    else:
                        batch.extend(random.choices(indices, k=self.k))
                yield batch

        def __len__(self):
            return self.batches_per_epoch

    # ── 8. Model: DINOv2 + LoRA + projection head ────────────────────────────
    class FontSimilarityModel(nn.Module):
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
    model = FontSimilarityModel(backbone).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} params")

    # ── 9. Loss functions ─────────────────────────────────────────────────────
    _log_counter = [0]

    def family_identity_supcon(embeddings, family_ids, temperature=0.07):
        n = len(embeddings)
        sim_mat = torch.mm(embeddings, embeddings.T) / temperature
        ids = torch.tensor(family_ids, device=embeddings.device)
        pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
        eye = torch.eye(n, device=embeddings.device)
        pos_mask = pos_mask - eye
        num_pos = pos_mask.sum(dim=1)

        if _log_counter[0] < 5:
            total_pairs = n * (n - 1)
            n_pos = pos_mask.sum().item()
            n_neg = total_pairs - n_pos
            n_families = len(set(family_ids))
            print(f"  [batch {_log_counter[0]}] family_supcon: "
                  f"{n_families} families | pos={n_pos:.0f} neg={n_neg:.0f} / {total_pairs:.0f} pairs")
            _log_counter[0] += 1

        if (num_pos > 0).sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        masked_sim = sim_mat - eye * 1e4
        log_sum_exp = torch.logsumexp(masked_sim, dim=1)
        mean_pos_sim = (sim_mat * pos_mask).sum(dim=1) / num_pos.clamp(min=1)
        loss = log_sum_exp - mean_pos_sim
        valid = num_pos > 0
        return loss[valid].mean()

    _file_log_counter = [0]

    def file_identity_supcon(embeddings, file_indices, temperature=0.1):
        n = len(embeddings)
        sim_mat = torch.mm(embeddings, embeddings.T) / temperature
        ids = torch.tensor(file_indices, device=embeddings.device)
        pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
        eye = torch.eye(n, device=embeddings.device)
        pos_mask = pos_mask - eye
        num_pos = pos_mask.sum(dim=1)

        if _file_log_counter[0] < 3:
            n_pos = pos_mask.sum().item()
            n_with_pos = (num_pos > 0).sum().item()
            print(f"  [batch {_file_log_counter[0]}] file_supcon: "
                  f"{n_with_pos}/{n} samples have same-file positives | {n_pos:.0f} pos pairs")
            _file_log_counter[0] += 1

        if (num_pos > 0).sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        masked_sim = sim_mat - eye * 1e4
        log_sum_exp = torch.logsumexp(masked_sim, dim=1)
        mean_pos_sim = (sim_mat * pos_mask).sum(dim=1) / num_pos.clamp(min=1)
        loss = log_sum_exp - mean_pos_sim
        valid = num_pos > 0
        return loss[valid].mean()

    # ── 10. Dataloaders ───────────────────────────────────────────────────────
    def worker_init_fn(worker_id):
        random.seed(random.getrandbits(32) + worker_id)
        np.random.seed(random.getrandbits(32) + worker_id)

    train_dataset = FontDataset(train_fonts, fontfile_to_paths,
                                font_family_id_map, clean)
    val_dataset = FontDataset(val_fonts, fontfile_to_paths,
                              font_family_id_map, clean)

    train_sampler = FamilyPKSampler(
        train_fonts, fontfile_to_family,
        p=24, k=3, batches_per_epoch=1500
    )
    val_sampler = FamilyPKSampler(
        val_fonts, fontfile_to_family,
        p=24, k=3, batches_per_epoch=200
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn
    )

    # ── 11. Optimizer + scheduler ─────────────────────────────────────────────
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

    # ── 12. Evaluation ────────────────────────────────────────────────────────
    class GlyphEvalDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    print("Pre-rendering Recall@k query images for val fonts...")
    recall_base_images = {}
    recall_text_strings = ["Hamburgevons", "Aa Bb Gg", "Quick fox", "AB", "DESIGN"]
    for font_name in val_fonts:
        font_path = fontfile_to_fontpath.get(font_name)
        if font_path is None:
            continue
        images = []
        for text in recall_text_strings:
            img = render_text_image(font_path, text)
            if img is not None:
                images.append(img)
        if images:
            recall_base_images[font_name] = images
    print(f"Pre-rendered {sum(len(v) for v in recall_base_images.values())} recall query images "
          f"for {len(recall_base_images)} val fonts")

    @torch.no_grad()
    def evaluate(model, fonts, fontfile_to_paths):
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
            embs = model(batch.to(device))
            all_embs.append(embs.cpu())
        all_embs = torch.cat(all_embs)

        font_idx_tensor = torch.tensor(all_font_idx)
        font_embs = []
        for fidx in range(len(fonts)):
            mask = font_idx_tensor == fidx
            avg = all_embs[mask].mean(dim=0)
            font_embs.append(F.normalize(avg, dim=0))
        font_embs = torch.stack(font_embs)

        sim_mat = torch.mm(font_embs, font_embs.T)
        sim_mat.fill_diagonal_(-1.0)
        topk_vals, nn_indices = sim_mat.topk(5, largest=True)

        cat_hits = 0
        for i in range(len(fonts)):
            fam_i = fontfile_to_family.get(fonts[i], "")
            cat_i = family_category.get(fam_i, "")
            fam_nn = fontfile_to_family.get(fonts[nn_indices[i][0]], "")
            cat_nn = family_category.get(fam_nn, "")
            if cat_i == cat_nn and cat_i:
                cat_hits += 1
        cat_acc = cat_hits / len(fonts) if fonts else 0

        font_to_family_idx = {}
        family_names = []
        for fidx, font_name in enumerate(fonts):
            fam = fontfile_to_family.get(font_name, font_name)
            if fam not in font_to_family_idx:
                font_to_family_idx[fam] = len(family_names)
                family_names.append(fam)
        font_family_ids = [font_to_family_idx[fontfile_to_family.get(f, f)] for f in fonts]

        recall_at = {1: 0, 5: 0, 10: 0}
        total_queries = 0

        query_tensors = []
        query_family_ids = []
        for fidx, font_name in enumerate(fonts):
            family_id = font_family_ids[fidx]
            for base_img in recall_base_images.get(font_name, []):
                query_tensors.append(clean(base_img.copy()))
                query_family_ids.append(family_id)

        if query_tensors:
            query_batch = torch.stack(query_tensors)
            query_embs = []
            for i in range(0, len(query_batch), 128):
                chunk = query_batch[i:i+128].to(device)
                query_embs.append(model(chunk).cpu())
            query_embs = torch.cat(query_embs)

            all_sims = torch.mm(query_embs, font_embs.T)
            for qi in range(len(query_embs)):
                ranked = all_sims[qi].argsort(descending=True)
                family_id = query_family_ids[qi]
                for k in [1, 5, 10]:
                    top_families = set(font_family_ids[r.item()] for r in ranked[:k])
                    if family_id in top_families:
                        recall_at[k] += 1
                total_queries += 1

        r1 = recall_at[1] / max(total_queries, 1)
        r5 = recall_at[5] / max(total_queries, 1)
        r10 = recall_at[10] / max(total_queries, 1)

        known = ["Roboto-Bold", "PlayfairDisplay-Regular", "CourierPrime-Regular"]
        for kf in known:
            if kf in fonts:
                idx = fonts.index(kf)
                neighbors = [fonts[nn_indices[idx][j]] for j in range(3)]
                print(f"    {kf} -> {neighbors}")

        return {"cat_acc": cat_acc, "R@1": r1, "R@5": r5, "R@10": r10}

    # ── 13. Training loop ─────────────────────────────────────────────────────
    best_recall5 = -1.0
    patience_counter = 0
    PATIENCE = 7

    for epoch in range(EPOCHS):
        model.train()
        epoch_family = 0.0
        epoch_file = 0.0
        num_batches = 0

        for view1, view2, indices, fam_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train"):
            images = torch.cat([view1, view2], dim=0).to(device)
            fam_ids_list = fam_ids.tolist() * 2
            indices_list = indices.tolist() * 2

            with autocast("cuda"):
                embeddings = model(pixel_values=images)
                loss_family = family_identity_supcon(embeddings, fam_ids_list)
                loss_file = file_identity_supcon(embeddings, indices_list)
                loss = loss_family + 0.2 * loss_file

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_family += loss_family.item()
            epoch_file += loss_file.item()
            num_batches += 1

        avg_family = epoch_family / max(num_batches, 1)
        avg_file = epoch_file / max(num_batches, 1)

        model.eval()
        val_family = 0.0
        val_file = 0.0
        val_batches = 0
        with torch.no_grad():
            for view1, view2, indices, fam_ids in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} val"):
                images = torch.cat([view1, view2], dim=0).to(device)
                fam_ids_list = fam_ids.tolist() * 2
                indices_list = indices.tolist() * 2
                with autocast("cuda"):
                    embeddings = model(pixel_values=images)
                    lf = family_identity_supcon(embeddings, fam_ids_list)
                    lfi = file_identity_supcon(embeddings, indices_list)
                val_family += lf.item()
                val_file += lfi.item()
                val_batches += 1

        avg_val_family = val_family / max(val_batches, 1)
        avg_val_file = val_file / max(val_batches, 1)

        metrics = evaluate(model, val_fonts, fontfile_to_paths)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} — fam: {avg_family:.4f} file: {avg_file:.4f} | "
              f"v_fam: {avg_val_family:.4f} v_file: {avg_val_file:.4f} | "
              f"R@1: {metrics['R@1']:.4f} R@5: {metrics['R@5']:.4f} R@10: {metrics['R@10']:.4f} | "
              f"cat: {metrics['cat_acc']:.4f} | lr: {lr:.2e}")

        r5 = metrics["R@5"]
        if r5 > best_recall5:
            best_recall5 = r5
            patience_counter = 0
            torch.save(model.state_dict(), "/data/font_v7_best_full.pt")
            volume.commit()
            print(f"  -> Best model saved (R@5: {best_recall5:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best R@5: {best_recall5:.4f})")
                break

        torch.save(model.state_dict(), f"/data/checkpoint_v7_epoch{epoch+1}.pt")
        volume.commit()

    # ── 14. Merge LoRA and save inference model ───────────────────────────────
    if not os.path.exists("/data/font_v7_best_full.pt"):
        print("ERROR: No best model was saved. Saving current model as fallback.")
        torch.save(model.state_dict(), "/data/font_v7_best_full.pt")
        volume.commit()

    print("Merging LoRA weights for inference...")
    best_state = torch.load("/data/font_v7_best_full.pt", map_location=device)
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

    torch.save(inference_model.state_dict(), "/data/font_embedder_v7.pt")
    volume.commit()
    print(f"Inference model saved to /data/font_embedder_v7.pt")
    print(f"Done. Best R@5: {best_recall5:.4f}")

@app.local_entrypoint()
def main():
    finetune.remote()
