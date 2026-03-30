import modal

app = modal.App("font-merge-v7")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("torch", "torchvision", "transformers", "peft")
)

volume = modal.Volume.from_name("font-data", create_if_missing=True)

@app.function(image=image, gpu="A100-40GB", timeout=600, volumes={"/data": volume})
def merge():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")

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
        r=8, lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05, bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    model = FontSimilarityModel(backbone).to(device)

    print("Loading best checkpoint...")
    state = torch.load("/data/font_v7_best_full.pt", map_location=device)
    model.load_state_dict(state)

    print("Merging LoRA weights...")
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
    print("Saved /data/font_embedder_v7.pt")

@app.local_entrypoint()
def main():
    merge.remote()
