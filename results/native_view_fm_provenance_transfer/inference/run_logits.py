#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


MAPPINGS = (
    ("map0", "A = NOT READY; B = READY."),
    ("map1", "A = READY; B = NOT READY."),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def messages(row: dict[str, Any], root: Path, mapping: str) -> list[dict[str, Any]]:
    image = root / str(row["image"])
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {
                    "type": "text",
                    "text": f"{row['prompt_text']}\n{mapping}\nAnswer with one letter only:",
                },
            ],
        }
    ]


def score(model: Any, processor: Any, conversations: list[list[dict[str, Any]]], token_ids: list[int]) -> list[list[float]]:
    import torch
    from qwen_vl_utils import process_vision_info

    rendered = [
        processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
        for item in conversations
    ]
    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(
        text=rendered,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(str(next(model.parameters()).device))
    with torch.inference_mode():
        logits = model(**inputs).logits[:, -1, token_ids].float().cpu().tolist()
    return logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompt-pack", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    prompts = read_jsonl(args.prompt_pack)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    if not prompts or len({str(row["id"]) for row in prompts}) != len(prompts):
        raise RuntimeError("prompt pack is empty or contains duplicate IDs")
    for row in prompts:
        image = args.root / str(row["image"])
        if not image.is_file() or sha256_file(image) != row["image_sha256"]:
            raise RuntimeError(f"missing or changed image: {image}")

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    token_ids = []
    for token in ("A", "B"):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"{token!r} is not one tokenizer token: {encoded}")
        token_ids.append(int(encoded[0]))
    if token_ids[0] == token_ids[1]:
        raise RuntimeError("A and B share one tokenizer token")
    processor.tokenizer.padding_side = "left"

    max_memory: dict[int | str, str] = {
        index: "44GiB" for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = "96GiB"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

    existing = read_jsonl(args.outputs) if args.resume else []
    done = {str(row["id"]) for row in existing if row.get("status") == "ok"}
    pending = [row for row in prompts if str(row["id"]) not in done]
    environment = {
        "model_id": args.model_id,
        "revision": args.revision,
        "model_path": str(args.model_path),
        "model_config_sha256": sha256_file(args.model_path / "config.json"),
        "prompt_pack_sha256": sha256_file(args.prompt_pack),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_visible_devices": args.cuda_visible_devices,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "batch_size": args.batch_size,
        "option_token_ids": {"A": token_ids[0], "B": token_ids[1]},
    }
    args.environment_output.parent.mkdir(parents=True, exist_ok=True)
    args.environment_output.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        conversations = [
            messages(row, args.root, mapping)
            for row in batch
            for _, mapping in MAPPINGS
        ]
        started = time.time()
        logits = score(model, processor, conversations, token_ids)
        elapsed = (time.time() - started) / len(batch)
        if len(logits) != 2 * len(batch):
            raise RuntimeError("mapping output cardinality mismatch")
        outputs = []
        for index, row in enumerate(batch):
            map0, map1 = logits[2 * index], logits[2 * index + 1]
            ready_log_odds = 0.5 * ((map0[1] - map0[0]) + (map1[0] - map1[1]))
            p_ready = sigmoid(ready_log_odds)
            outputs.append(
                {
                    "id": row["id"],
                    "status": "ok",
                    "model_id": args.model_id,
                    "revision": args.revision,
                    "event_id": row["event_id"],
                    "episode_id": row["episode_id"],
                    "sid": row["sid"],
                    "window": row["window"],
                    "reference_ready": row["reference_ready"],
                    "physical_view_id": row["physical_view_id"],
                    "parent_id": row["parent_id"],
                    "surface_id": row["surface_id"],
                    "image_sha256": row["image_sha256"],
                    "prompt_record_sha256": row["prompt_record_sha256"],
                    "mapping_logits": [
                        {"mapping": "map0", "logit_A": map0[0], "logit_B": map0[1]},
                        {"mapping": "map1", "logit_A": map1[0], "logit_B": map1[1]},
                    ],
                    "ready_log_odds": ready_log_odds,
                    "p_ready": p_ready,
                    "evidence": [2.0 * (1.0 - p_ready), 2.0 * p_ready],
                    "latency_seconds": elapsed,
                }
            )
        append_jsonl(args.outputs, outputs)
        done.update(str(row["id"]) for row in outputs)
        print(json.dumps({"completed": len(done), "target": len(prompts)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
