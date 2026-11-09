"""One bounded training-image fit after the observed gray-output failure."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import zipfile

import torch
from torch.utils.data import default_collate

from visualization.raster_layouts import comparison_sheet, save_rgb
from visualization.semantic_images import ids_to_rgb
from ..data import five_indices
from .data import SemanticImages
from .model import SemanticGenerator, MASK
from .loss import semantic_loss, image_metrics
from .generation import corrupt, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(__file__).with_name("plan.json").read_text(encoding="utf-8"))
    spec = plan["necessary_learning_check"]
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    ds = SemanticImages("train_monitor500", 512)
    index = five_indices(ds)[0]
    data = default_collate([ds[index]])
    public = {k: data[k].to(device) for k in ("features", "relations", "valid")}
    target = data["labels"].to(device)
    model = SemanticGenerator(**plan["architecture"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=.0001)
    generator = torch.Generator(device=device).manual_seed(0)
    full_mask = torch.full_like(target, MASK)

    @torch.no_grad()
    def evaluate():
        model.eval()
        result = model(full_mask, torch.ones(1, device=device), public)
        loss, parts = semantic_loss(result, target, torch.ones_like(target, dtype=torch.bool))
        prediction = result["logits"].argmax(1)
        metrics = image_metrics(prediction, target)
        row = {"all_masked_loss": float(loss[0]), **{k: float(v[0]) for k, v in metrics.items()},
               **{k: float(v[0]) for k, v in parts.items()}}
        model.train()
        return row, prediction

    initial, _ = evaluate()
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"step": 0, **initial}), flush=True)
    begin = time.perf_counter()
    steps = 0
    for step in range(spec["steps"]):
        if time.perf_counter() - begin >= spec["max_seconds"]:
            break
        optimizer.zero_grad(set_to_none=True)
        unconditional_output = model(full_mask, torch.ones(1, device=device), public)
        unconditional_loss, _ = semantic_loss(unconditional_output, target, torch.ones_like(target, dtype=torch.bool))
        (.5 * unconditional_loss.mean()).backward()
        all_masked_training_loss = float(unconditional_loss.detach().mean())
        del unconditional_output, unconditional_loss
        noisy, fraction, hidden = corrupt(target, generator)
        output = model(noisy, fraction, public)
        loss, parts = semantic_loss(output, target, hidden)
        loss = loss.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite semantic loss")
        (.5 * loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if not torch.stack([p.isfinite().all() for p in model.parameters()]).all():
            raise FloatingPointError("Nonfinite parameters")
        steps = step + 1
        if steps % 25 == 0:
            row, _ = evaluate()
            row = {"step": steps, "training_loss": .5 * (float(loss.detach()) + all_masked_training_loss), **row}
            with (args.out / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    final, fitted = evaluate()
    model.eval()
    generated = sample(model, public, resolution=512, steps=16, seed=0)
    generated_metrics = {k: float(v[0]) for k, v in image_metrics(generated, target).items()}
    rgb = [ids_to_rgb(v[0].cpu().numpy()) for v in (target, fitted, generated)]
    for name, image in zip(("reference", "all_masked_prediction", "iterative_generation"), rgb):
        save_rgb(args.out / f"{name}.png", image)
    note = f"ONE TRAIN MEMBER | {ds.records[index]['member']} | fitting diagnostic, not generalization"
    comparison_sheet(rgb[:2], ["Original reference", "All-masked prediction after fitting"],
                     args.out / "all_masked_comparison.png", caption=note)
    comparison_sheet([rgb[0], rgb[2]], ["Original reference", "16-round generated image"],
                     args.out / "generation_comparison.png", caption=note)
    report = {"status": "completed", "actual_updates": steps, "training_member_count": 1,
        "member": ds.records[index]["member"], "registry_sha256": ds.reg["manifest_sha256"],
        "resolution": 512, "precision": "FP32", "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - begin, "initial": initial, "final": final,
        "sampled_image": generated_metrics, "learning_gate_passed": final["module_mean_IoU"] >= .75,
        "sampler_takes_reference": False, "formal_training_started": False, "dev_read": False,
        "test_read": False, "geometry_extraction": False, "repair": False, "Q_computed": False}
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"model": model.state_dict(), "model_config": model.config, "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "corruption_rng": generator.get_state(),
        "steps": steps, "plan": plan, "result": report}, args.out / "fit_final.pt")
    paths = list(Path(__file__).parent.glob("*.py")) + [Path(__file__).with_name("plan.json"),
        Path("visualization/semantic_images.py"), Path("visualization/raster_layouts.py"),
        Path("experiments/E102_reference_image_generation/data.py"), Path("experiments/E102_reference_image_generation/model.py")]
    with zipfile.ZipFile(args.out / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.as_posix())
    (args.out / "source_sha256.json").write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
