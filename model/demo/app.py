"""Offline Gradio demo: image → class + confidence + Grad-CAM overlay (M4).

Runs entirely locally (no network, no analytics) — built for a judged live
demo where robustness beats features.

Usage:
  python demo/app.py [--checkpoint checkpoints/<run>/ensemble.json]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never phone home mid-demo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.analysis import find_last_conv  # noqa: E402
from src.data import build_eval_transform  # noqa: E402
from src.evaluate import load_models  # noqa: E402
from src.utils import CLASSES, get_device  # noqa: E402

DEFAULT_CKPT = "checkpoints/baseline_resnet50/fold0/best.pth"


def build_predictor(checkpoint: str):
    device = get_device()
    models, cfg, _ = load_models(checkpoint, device)
    transform = build_eval_transform(cfg["img_size"])

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    cam = GradCAM(model=models[0], target_layers=[find_last_conv(models[0])])

    def predict_fn(img: Image.Image):
        if img is None:
            return {}, None
        img = img.convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.stack(
                [torch.softmax(m(x).float(), dim=1)[0] for m in models]
            ).mean(0).cpu().numpy()
        confidences = {c: float(p) for c, p in zip(CLASSES, probs)}

        gray = cam(input_tensor=x)[0]  # Grad-CAM from the first ensemble member
        base = np.asarray(img.resize((cfg["img_size"], cfg["img_size"]))).astype(np.float32) / 255
        overlay = show_cam_on_image(base, gray, use_rgb=True)
        return confidences, overlay

    return predict_fn, cfg, len(models)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    predict_fn, cfg, n_models = build_predictor(args.checkpoint)

    import gradio as gr
    demo = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="pil", label="trash photo"),
        outputs=[gr.Label(num_top_classes=6, label="prediction"),
                 gr.Image(label="Grad-CAM: where the model looked")],
        title="TrashNet 6-class classifier",
        description=(f"{cfg['model']} · {n_models} model(s) · offline · "
                     "leakage-free pipeline (split-before-augment, group-aware, "
                     "test touched once)"),
        flagging_mode="never",
    )
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False,
                show_api=False)


if __name__ == "__main__":
    main()
