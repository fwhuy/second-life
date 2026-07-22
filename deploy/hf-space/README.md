---
title: Second Life AI
emoji: ♻️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Second Life AI

Photograph a piece of waste and the model names its material, the bin it belongs
in, and the two futures waiting for it — recycled or landfilled. Bilingual
(中文 / English).

Classification runs a ResNet-50 (`resnet50.tv2_in1k`, 224px) fine-tuned on a
deduplicated unified waste corpus, with a leakage firewall between the training
and test splits. Validation accuracy is 94.93% with four-view test-time
augmentation. Six classes: cardboard, glass, metal, paper, plastic, trash.

Predictions on this page are the model's real softmax output — the same
inference path as the training repo's `evaluate.py`, not canned demo data.

## Notes

- Free CPU tier: roughly 0.3s per image, ~1.2s in accuracy mode (4-view TTA).
- The Space sleeps after 48 hours idle and takes ~30s to wake on the next visit.
- A kNN feature-distance guard flags images that fall outside the six classes
  rather than forcing a confident wrong answer.

Source: https://github.com/fwhuy/second-life
