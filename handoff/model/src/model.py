"""timm model wrapper: head swap, freeze/unfreeze, discriminative LR groups."""

from pathlib import Path

import timm
import torch

from .utils import NUM_CLASSES

# Competition rule: the pretrained model may not exceed 30M parameters. Enforced in
# code rather than trusted to config review — an over-cap run wastes GPU hours and
# produces a disqualified number. Override per-config with `max_params` (0 disables).
MAX_PARAMS = 30_000_000


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(cfg: dict, pretrained: bool = True) -> torch.nn.Module:
    model = timm.create_model(cfg["model"], pretrained=pretrained,
                              num_classes=NUM_CLASSES, **cfg.get("model_kwargs") or {})
    limit = cfg.get("max_params", MAX_PARAMS)
    n_params = count_parameters(model)
    if limit and n_params > limit:
        raise ValueError(
            f"{cfg['model']} has {n_params / 1e6:.1f}M parameters, over the "
            f"{limit / 1e6:.0f}M competition cap"
        )
    print(f"model={cfg['model']} params={n_params / 1e6:.1f}M (cap {limit / 1e6:.0f}M)")
    init = cfg.get("init_checkpoint")
    if init:
        state = torch.load(Path(init), map_location="cpu", weights_only=False)
        sd = state.get("ema") or state.get("model") if isinstance(state, dict) else state
        # progressive resizing: 224→384 checkpoints may differ in pos-embed shapes
        missing, unexpected = model.load_state_dict(sd, strict=False)
        skipped = [k for k in missing + unexpected]
        if skipped:
            print(f"init_checkpoint: skipped {len(skipped)} mismatched tensors (expected for 224→384)")
    return model


def head_parameters(model: torch.nn.Module):
    return list(model.get_classifier().parameters())


def backbone_parameters(model: torch.nn.Module):
    head_ids = {id(p) for p in head_parameters(model)}
    return [p for p in model.parameters() if id(p) not in head_ids]


def freeze_backbone(model: torch.nn.Module, frozen: bool = True) -> None:
    for p in backbone_parameters(model):
        p.requires_grad = not frozen


def param_groups(model: torch.nn.Module, lr_head: float, lr_backbone: float, weight_decay: float):
    """Discriminative LRs: backbone LR << head LR. Too-high backbone LR is the
    #1 cause of a 93%→75% collapse on this dataset."""
    return [
        {"params": [p for p in backbone_parameters(model) if p.requires_grad],
         "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_parameters(model), "lr": lr_head, "weight_decay": weight_decay},
    ]
