"""BF16 network operations, FP32 logits/Loss/master parameters/Adam."""
import torch


def forward_image(model, tokens, fraction, public, precision="FP32"):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "BF16"):
        result = model(tokens, fraction, public)
    # Loss, including probability-space BCE, always runs outside autocast.
    return {key: value.float() for key, value in result.items()}
