# Second Life AI website

Offline bilingual UI and Flask inference API for the project's two models:

- `convnet`: ConvNeXt V2-Tiny, 384px, six output classes (default)
- `transformer`: Swin-B, 224px, restricted to the same six displayed classes

## Run

**macOS:** double-click `run.command`

**Windows:** double-click `run.bat`

**macOS/Linux terminal:** `./start.sh`

The first launch creates `.venv` and installs dependencies. Inference is local
after setup; the app reads both checkpoints directly from `../model/`.

## API

- `GET /api/model` — runtime details, class order, and available model keys
- `POST /api/identify` — multipart `image` plus `model=convnet|transformer`

A successful prediction contains `cls`, `conf`, `probs`, `model`, and guard
status. `probs` always uses the shared order: cardboard, glass, metal, paper,
plastic, trash.

The displayed confidence is the selected model's raw softmax output. No
temperature scaling is applied. The bundled feature bank belongs to an older
architecture, so the current model loaders correctly report the open-set guard
as off rather than mixing incompatible embeddings.
