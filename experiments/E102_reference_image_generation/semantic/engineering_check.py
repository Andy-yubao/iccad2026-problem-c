"""Four next train members, FP32 loss reference and one actual BF16 update."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate

from .data import CachedSemanticImages
from .generation import corrupt
from .loss import semantic_loss
from .model import SemanticGenerator, MASK
from .precision import forward_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.parent / "latest.pt", map_location="cpu", weights_only=False)
    model = SemanticGenerator(**saved["model_config"]).cuda()
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    optimizer.load_state_dict(saved["optimizer"])
    dataset = CachedSemanticImages(args.cache, "train")
    cursor = saved["state"]["cursor"]
    indices = saved["state"]["order"][cursor:cursor + 4]
    batch = default_collate([dataset[i] for i in indices])
    public = {k: batch[k].cuda() for k in ("features", "relations", "valid")}
    target = batch["labels"].cuda()
    generator = torch.Generator(device="cuda")
    generator.set_state(saved["corruption_rng"])
    noisy, fraction, hidden = corrupt(target, generator)
    with torch.no_grad():
        model.eval()
        baseline = torch.zeros(4, device="cuda")
        for begin in (0, 2):
            part = {k: v[begin:begin + 2] for k, v in public.items()}
            truth = target[begin:begin + 2]
            a, _ = semantic_loss(forward_image(model, torch.full_like(truth, MASK), torch.ones(2, device="cuda"), part, "FP32"),
                                 truth, torch.ones_like(truth, dtype=torch.bool))
            b, _ = semantic_loss(forward_image(model, noisy[begin:begin + 2], fraction[begin:begin + 2], part, "FP32"), truth, hidden[begin:begin + 2])
            baseline[begin:begin + 2] = .5 * (a + b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    began = time.perf_counter()
    full, _ = semantic_loss(forward_image(model, torch.full_like(target, MASK), torch.ones(4, device="cuda"), public, "BF16"),
                            target, torch.ones_like(target, dtype=torch.bool))
    assert full.dtype == torch.float32
    (.5 * full.mean()).backward()
    full_values = full.detach().clone()
    del full
    masked, _ = semantic_loss(forward_image(model, noisy, fraction, public, "BF16"), target, hidden)
    assert masked.dtype == torch.float32
    (.5 * masked.mean()).backward()
    new_loss = .5 * (full_values + masked.detach())
    relative = float((new_loss.mean() - baseline.mean()).abs() / baseline.mean().abs().clamp_min(1e-6))
    if not torch.isfinite(new_loss).all() or relative > .02:
        raise FloatingPointError(f"BF16 Loss drift exceeds registered2% gate: {relative}")
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    finite = bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all())
    assert finite and all(p.dtype == torch.float32 for p in model.parameters())
    assert all(v.dtype == torch.float32 for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    report = {"status": "passed", "train_instances": 4, "members": [dataset.records[i]["member"] for i in indices],
        "parent_step": saved["state"]["steps"], "actual_bf16_updates": 1,
        "FP32_reference_microbatch": 2, "BF16_batch": 4, "FP32_mean_loss": float(baseline.mean()),
        "BF16_mean_loss": float(new_loss.mean()), "relative_loss_difference": relative,
        "loss_difference_gate": .02, "gradient_norm_before_clip": float(grad), "finite_parameters": finite,
        "master_weights_loss_adam_dtype": "FP32", "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
        "first_update_seconds": seconds, "first_update_samples_per_second": 4 / seconds,
        "timing_note": "single engineering update, not steady-state throughput or epoch speedup",
        "parent_checkpoint_modified": False, "check_weights_used_for_training": False, "test_read": False,
        "parent_latest_sha256": hashlib.sha256((args.parent / "latest.pt").read_bytes()).hexdigest()}
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
