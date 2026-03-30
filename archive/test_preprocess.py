import modal
import io

app = modal.App("test-preprocess")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision",
        "Pillow", "numpy", "opencv-python-headless",
        "tqdm", "matplotlib",
        "pycocotools", "einops", "Polygon3",
        "pyclipper", "python-Levenshtein",
        "scikit-image", "scipy", "shapely", "timm",
        "huggingface_hub",
    )
    .run_commands(
        "git clone https://github.com/ymy-k/Hi-SAM.git /opt/Hi-SAM",
        "mkdir -p /opt/Hi-SAM/pretrained_checkpoint",
    )
    .run_commands(
        'python3 -c "'
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download('GoGiants1/Hi-SAM', 'sam_vit_b_01ec64.pth', local_dir='/opt/Hi-SAM/pretrained_checkpoint'); "
        "hf_hub_download('GoGiants1/Hi-SAM', 'sam_tss_b_hiertext.pth', local_dir='/opt/Hi-SAM/pretrained_checkpoint'); "
        "hf_hub_download('GoGiants1/Hi-SAM', 'sam_tss_b_textseg.pth', local_dir='/opt/Hi-SAM/pretrained_checkpoint')"
        '"',
    )
)

rembg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("rembg[cpu]", "Pillow", "numpy")
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=600, volumes={"/data": volume})
def test_hisam(image_bytes: bytes):
    import sys, os
    os.chdir("/opt/Hi-SAM")
    sys.path.insert(0, "/opt/Hi-SAM")

    import numpy as np
    import torch
    from PIL import Image
    from argparse import Namespace

    from hi_sam.modeling.build import model_registry
    from hi_sam.modeling.predictor import SamPredictor

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(img)
    os.makedirs("/data/preprocess_test", exist_ok=True)

    checkpoints = {
        "hiertext": "/opt/Hi-SAM/pretrained_checkpoint/sam_tss_b_hiertext.pth",
        "textseg": "/opt/Hi-SAM/pretrained_checkpoint/sam_tss_b_textseg.pth",
    }

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n=== Testing SAM-TS-B ({ckpt_name}) ===")
        args = Namespace(
            checkpoint=ckpt_path,
            model_type="vit_b",
            device="cuda",
            hier_det=False,
            input_size=[1024, 1024],
            patch_mode=False,
            attn_layers=1,
            prompt_len=12,
        )

        model = model_registry[args.model_type](args)
        model.eval()
        model.to(args.device)
        predictor = SamPredictor(model)

        predictor.set_image(image_np)

        # Get raw logits first
        mask_logits, hr_logits, score, hr_score = predictor.predict(
            multimask_output=False, return_logits=True
        )
        print(f"Logit stats — mask: min={mask_logits.min():.3f} max={mask_logits.max():.3f} "
              f"mean={mask_logits.mean():.3f}")
        print(f"Logit stats — hr:   min={hr_logits.min():.3f} max={hr_logits.max():.3f} "
              f"mean={hr_logits.mean():.3f}")
        print(f"Scores: mask_iou={score}, hr_iou={hr_score}")

        # Apply threshold (SAM default is 0.0 for logits)
        text_mask = (hr_logits[0] > 0.0).astype(np.uint8)
        print(f"Mask: text pixels={text_mask.sum()}, total={text_mask.size}, "
              f"ratio={text_mask.sum()/text_mask.size:.2%}")

        h, w = text_mask.shape
        result = np.full((h, w), 255, dtype=np.uint8)
        result[text_mask == 1] = 0

        mask_img = Image.fromarray(text_mask * 255)
        mask_img.save(f"/data/preprocess_test/hisam_{ckpt_name}_mask.png")

        result_img = Image.fromarray(result)
        result_img.save(f"/data/preprocess_test/hisam_{ckpt_name}_result.png")

        # Also save logit heatmap for debugging
        hr_logit_map = hr_logits[0]
        norm_logits = ((hr_logit_map - hr_logit_map.min()) /
                       (hr_logit_map.max() - hr_logit_map.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(norm_logits)
        heatmap_img.save(f"/data/preprocess_test/hisam_{ckpt_name}_logits.png")

        del model, predictor
        torch.cuda.empty_cache()

    volume.commit()
    print("\nHi-SAM results saved for both checkpoints.")


@app.function(image=rembg_image, timeout=600, volumes={"/data": volume})
def test_rembg(image_bytes: bytes):
    from rembg import remove, new_session
    from PIL import Image
    import numpy as np
    import os

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    os.makedirs("/data/preprocess_test", exist_ok=True)

    for model_name in ["u2net", "birefnet-general", "isnet-general-use"]:
        print(f"Testing rembg model: {model_name} ...")
        session = new_session(model_name)

        output = remove(img, session=session)  # RGBA
        output.save(f"/data/preprocess_test/rembg_{model_name}_rgba.png")

        output_np = np.array(output)
        alpha = output_np[:, :, 3]

        h, w = alpha.shape
        result = np.full((h, w), 255, dtype=np.uint8)
        result[alpha > 128] = 0

        result_img = Image.fromarray(result)
        result_img.save(f"/data/preprocess_test/rembg_{model_name}_bw.png")

        print(f"  {model_name} done — foreground pixels: {(alpha > 128).sum()}, "
              f"ratio: {(alpha > 128).sum() / alpha.size:.2%}")
        volume.commit()

    print("All rembg results saved.")


@app.local_entrypoint()
def main(image_path: str = "crop1.jpg"):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print("Spawning Hi-SAM and rembg tests in parallel...")
    hisam_call = test_hisam.spawn(image_bytes)
    rembg_call = test_rembg.spawn(image_bytes)

    hisam_call.get()
    print("[Hi-SAM] Done.")

    rembg_call.get()
    print("[rembg] Done.")

    print("\nAll results saved! Download with:")
    print("  python3 -m modal volume get font-data preprocess_test/ ./preprocess_test/")
