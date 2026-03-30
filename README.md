# similar-google-font

Find the closest matching Google Font from a photo of text. Uses a fine-tuned DINOv2 backbone to compare neural embeddings of the photographed text against a pre-computed index of ~3,600 rendered Google Font samples.

## How it works

1. **Training** (`finetune_v8.py`): Fine-tunes a DINOv2-base backbone with a projection head (CLS + patch token pooling → 256-dim) using supervised contrastive loss on rendered glyph images from ~3,600 Google Fonts.

2. **Index** (`inference.py :: build_index`): Renders each font at 68 individual characters and 26 text strings (e.g., "OF", "AC", "Hamburgevons"), embeds them all, and saves the index.

3. **Inference** (`inference.py :: FontMatcher`): Preprocesses a photo using rembg background removal, crops and pads to a square, embeds it, and finds the nearest fonts via cosine similarity — optionally filtered by OCR text to narrow the search space.

## Usage

Requires a [Modal](https://modal.com) account. All training and inference runs on Modal's cloud GPUs.

```bash
# Train the model (~4-5 hours on A100)
python3 -m modal run --detach finetune_v8.py

# Build the font index (~10 min)
python3 -m modal run inference.py --command build-index

# Test on a cropped photo of text
python3 -m modal run inference.py --command test --image-path test_images/crop1.jpg --text "OF"

# Full pipeline: detect text regions + match fonts from a complete photo
python3 -m modal run inference.py --command test-full --image-path test_images/sign1.jpg
```

## Project structure

```
inference.py        Main inference pipeline (text detection, preprocessing, matching)
finetune_v8.py      Current training script (DINOv2 + CLS/patch pooling)
finetune_v7.py      Previous training script (CLS-only pooling)
test_images/        Sample photos for testing
debug/              Reference preprocessed outputs
archive/            Earlier training script versions (v1-v6)
```
