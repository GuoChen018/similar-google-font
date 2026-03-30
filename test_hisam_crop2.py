import modal
import io

app = modal.App("test-hisam-crop2")

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
        "hf_hub_download('GoGiants1/Hi-SAM', 'sam_tss_b_textseg.pth', local_dir='/opt/Hi-SAM/pretrained_checkpoint')"
        '"',
    )
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
    os.makedirs("/data/preprocess_test_crop2", exist_ok=True)

    ckpt_path = "/opt/Hi-SAM/pretrained_checkpoint/sam_tss_b_textseg.pth"
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

    mask_logits, hr_logits, score, hr_score = predictor.predict(
        multimask_output=False, return_logits=True
    )
    print(f"Logit stats — hr: min={hr_logits.min():.3f} max={hr_logits.max():.3f}")
    print(f"Scores: mask_iou={score}, hr_iou={hr_score}")

    text_mask = (hr_logits[0] > 0.0).astype(np.uint8)
    print(f"Mask: text pixels={text_mask.sum()}, ratio={text_mask.sum()/text_mask.size:.2%}")

    h, w = text_mask.shape
    result = np.full((h, w), 255, dtype=np.uint8)
    result[text_mask == 1] = 0
    Image.fromarray(result).save("/data/preprocess_test_crop2/hisam_textseg_bw.png")

    volume.commit()
    print("Hi-SAM result saved.")


@app.local_entrypoint()
def main(image_path: str = "test_images/crop2.jpg"):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    test_hisam.remote(image_bytes)
    print("Done! Download with:")
    print("  python3 -m modal volume get font-data preprocess_test_crop2/hisam_textseg_bw.png ./debug/hisam_crop2.png")
