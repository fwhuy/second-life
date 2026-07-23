# Second Life AI website

Offline bilingual UI and Flask inference API for the project's two models:

- `convnet`: TrashNeXt (ConvNeXt V2-Tiny architecture), 384px, six output classes (default)
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
temperature scaling is applied.

The optional Swin-based five-signal OOD guard is calibrated by
`ood_guard_calib.npz`. Send `guard=on` with an identify request to enable it;
ordinary requests leave it off. The retired ResNet feature-bank experiment has
been removed from the active website.

The two Remotion projects also have separate roles: `website/remotion/` produces
the short website documentary, while the repository-level `video/` directory
produces the presentation video.
