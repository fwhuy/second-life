# Dual-model guide

The active project has exactly two model tracks: a ConvNet and a Transformer.
Their training programs are independent, while `website/app.py` provides one
shared inference interface.

## 1. Run and verify the website

```bash
cd website
./start.sh
```

Useful checks while it is running:

```bash
curl http://127.0.0.1:5001/api/model
curl -F image=@/path/to/photo.jpg -F model=convnet \
  http://127.0.0.1:5001/api/identify
curl -F image=@/path/to/photo.jpg -F model=transformer \
  http://127.0.0.1:5001/api/identify
```

`/api/model` reports which models are available. `/api/identify` always returns
the same six probability keys, regardless of the selected architecture.

## 2. Train the ConvNet

The ConvNet program downloads and deduplicates its multi-source six-class
corpus, trains ConvNeXt V2-Tiny at 384px, and writes its checkpoint and metadata
to the current directory.

```bash
cd "model/convnextv2_tiny_cnn"
python "train_and_upload .py" --no-upload
```

Place the two outputs in `results/` if training wrote them beside the script:

- `best_convnextv2.pt`
- `best_convnextv2_metadata.json`

The metadata is the source of truth for architecture, label order, image size,
parameter count, and validation accuracy.

## 3. Train the Transformer

```bash
cd "model/Swing B Transformer"
python train_swin_b.py
```

This writes `best_swin_b.pt` and `swin_b_results.json`. Swin-B is trained on ten
labels. The app keeps only the six labels shared with the ConvNet, then applies
softmax to those six logits.

## 4. Runtime contract

Do not duplicate checkpoints into `website/`. The canonical paths are:

```text
model/convnextv2_tiny_cnn/results/best_convnextv2.pt
model/convnextv2_tiny_cnn/results/best_convnextv2_metadata.json
model/Swing B Transformer/best_swin_b.pt
model/Swing B Transformer/swin_b_results.json
```

If a model, class order, transform, or checkpoint name changes, update its
loader in `website/app.py` and its adjacent metadata/results file together.
Keep the public API keys `convnet` and `transformer` stable.
