"""Batched gold HyperLoRA inference; saves predictions without scoring metrics."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from multi_hyper import (
    clear_hyperlora_state,
    generate_edited_text_batch_st,
    load_hyper_from_pt_st,
)
from Train_Light.hyper_light_train_gold import encode_prompts_with_st


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def run(args):
    files = [Path(p).resolve() for p in args.inputs]
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Input filenames must have distinct stems")
    datasets = []
    for path in files:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data["data"]
        if not isinstance(data, list) or not data:
            raise ValueError(f"Expected a nonempty record list: {path}")
        for i, row in enumerate(data):
            if not isinstance(row, dict) or "id" not in row or any(
                not isinstance(row.get(k), str) for k in ("context", "edit_request")
            ):
                raise ValueError(f"Invalid record: {path}:{i}")
        datasets.append((path, data[:args.limit] if args.limit else data))
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.ckpt).resolve()
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "inputs": [{"path": str(p), "num_samples": len(rows),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p, rows in datasets],
        "mode": "single_turn_without_request_splitting",
        "generation_function": "multi_hyper.generate_edited_text_batch_st",
        "prompt_encoder": "Train_Light.hyper_light_train_gold.encode_prompts_with_st",
        "batch_size": args.batch_size,
        "max_chunk_tokens": args.max_chunk_tokens,
        "max_new_tokens_per_chunk": args.max_new_tokens,
        "do_sample": False, "num_beams": 1, "padding_side": "left",
        "metrics": [], "limit_per_file": args.limit,
    }
    write_json(output / "manifest.json", manifest)
    completed = 0
    total = sum(len(rows) for _, rows in datasets)

    def status(state, **extra):
        write_json(output / "status.json", {
            "state": state, "pid": os.getpid(), "completed": completed,
            "total": total, "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        })

    targets, hooks = [], []
    status("loading")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok, lm, st, hyper, targets, hooks, _, alpha, sent_max_len = load_hyper_from_pt_st(
            str(checkpoint), device
        )
        manifest.update(sent_max_len=sent_max_len, target_layers=len(targets),
                        hypernetwork_loaded_strictly=True)
        write_json(output / "manifest.json", manifest)
        print(f"Hypernetwork loaded strictly; {len(targets)} target layers; "
              f"batch_size={args.batch_size}; no metrics", flush=True)
        for path, records in datasets:
            rows = []
            with (output / f"{path.stem}_predictions.jsonl").open("x") as stream:
                for start in range(0, len(records), args.batch_size):
                    batch = records[start:start + args.batch_size]
                    status("generating", dataset=path.stem, batch_start=start,
                           batch_size=len(batch))
                    print(f"{path.stem} starting {start + 1}-{start + len(batch)}"
                          f"/{len(records)}", flush=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    begin = time.monotonic()
                    generated = generate_edited_text_batch_st(
                        [r["context"] for r in batch], [r["edit_request"] for r in batch],
                        lm, tok, st, hyper, targets, alpha,
                        sent_max_len=sent_max_len, max_chunk_tokens=args.max_chunk_tokens,
                        max_new_tokens=args.max_new_tokens,
                        prompt_encoder=encode_prompts_with_st,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.monotonic() - begin
                    for offset, (record, text) in enumerate(zip(batch, generated)):
                        row = {
                            "id": record["id"], "input_index": start + offset,
                            "context": record["context"], "edit_request": record["edit_request"],
                            "generated_edit": text, "batch_index": start // args.batch_size,
                            "batch_size": len(batch), "batch_time_sec": elapsed,
                        }
                        if "edit_content" in record:
                            row["reference"] = record["edit_content"]
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
                    stream.flush()
                    completed += len(batch)
                    status("running", dataset=path.stem, dataset_completed=len(rows))
                    print(f"{path.stem} {len(rows)}/{len(records)} "
                          f"batch_time={elapsed:.1f}s", flush=True)
            write_json(output / f"{path.stem}_predictions.json", rows)
        status("completed")
    except BaseException as error:
        status("failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        clear_hyperlora_state(targets)
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_chunk_tokens", type=int, default=2000)
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size, args.max_chunk_tokens, args.max_new_tokens) <= 0 or args.limit < 0:
        parser.error("Batch size/token budgets must be positive and limit nonnegative")
    run(args)


if __name__ == "__main__":
    main()
