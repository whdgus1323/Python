import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(config):
    return SimpleMlp(config["input_dim"], config["hidden_dim"], config["num_classes"])


def flatten_state_dict(state_dict):
    flat = []
    for key in state_dict:
        flat.extend(state_dict[key].detach().cpu().view(-1).tolist())
    return flat


def apply_flattened_state_dict(model, flat_values):
    state_dict = model.state_dict()
    cursor = 0
    for key, tensor in state_dict.items():
        count = tensor.numel()
        values = torch.tensor(flat_values[cursor:cursor + count], dtype=tensor.dtype).view_as(tensor)
        state_dict[key] = values
        cursor += count
    model.load_state_dict(state_dict)


def state_file(state_dir: Path) -> Path:
    return state_dir / "global_state.json"


def config_file(state_dir: Path) -> Path:
    return state_dir / "config.json"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def generate_vehicle_dataset(node_id: int, sample_count: int, input_dim: int, num_classes: int, seed: int):
    generator = torch.Generator().manual_seed(seed + node_id * 97 + sample_count)
    base_centers = torch.randn(num_classes, input_dim, generator=generator) * 0.75
    heterogeneity = (node_id % max(1, num_classes)) * 0.35
    features = []
    labels = []
    for index in range(sample_count):
        label = (index + node_id) % num_classes
        center = base_centers[label] + heterogeneity
        noise = torch.randn(input_dim, generator=generator) * (0.45 + node_id * 0.005)
        features.append(center + noise)
        labels.append(label)
    x = torch.stack(features).float()
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def generate_validation_dataset(input_dim: int, num_classes: int, seed: int, sample_count: int = 512):
    generator = torch.Generator().manual_seed(seed + 2026)
    centers = torch.randn(num_classes, input_dim, generator=generator) * 0.8
    features = []
    labels = []
    for index in range(sample_count):
        label = index % num_classes
        features.append(centers[label] + torch.randn(input_dim, generator=generator) * 0.55)
        labels.append(label)
    x = torch.stack(features).float()
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def evaluate_model(model, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_count += batch_x.size(0)
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def compress_delta(delta, compression_ratio: float):
    vector = torch.tensor(delta, dtype=torch.float32)
    total = vector.numel()
    keep = max(1, int(math.ceil(total * compression_ratio)))
    _, indices = torch.topk(vector.abs(), k=min(keep, total))
    sorted_indices = torch.sort(indices).values.tolist()
    values = vector[sorted_indices].tolist()
    return sorted_indices, values


def serialize_int_list(values):
    return ",".join(str(int(value)) for value in values)


def serialize_float_list(values):
    return ",".join(f"{float(value):.8f}" for value in values)


def parse_int_list(value: str):
    if not value:
        return []
    return [int(item) for item in value.split(",") if item]


def parse_float_list(value: str):
    if not value:
        return []
    return [float(item) for item in value.split(",") if item]


def init_state(args):
    state_dir = Path(args.state_dir)
    ensure_dir(state_dir)
    ensure_dir(state_dir / "updates")
    config = {
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": args.num_classes,
        "seed": args.seed,
    }
    save_json(config_file(state_dir), config)
    seed_everything(args.seed)
    model = build_model(config)
    validation = generate_validation_dataset(args.input_dim, args.num_classes, args.seed)
    global_loss, global_accuracy = evaluate_model(model, validation)
    payload = {
        "round": 0,
        "model": flatten_state_dict(model.state_dict()),
        "global_loss": global_loss,
        "global_accuracy": global_accuracy,
        "model_score": global_accuracy,
    }
    save_json(state_file(state_dir), payload)
    return {
        "status": "ok",
        "global_loss": global_loss,
        "global_accuracy": global_accuracy,
        "model_score": global_accuracy,
    }


def load_model_and_config(state_dir: Path):
    config = load_json(config_file(state_dir))
    payload = load_json(state_file(state_dir))
    model = build_model(config)
    apply_flattened_state_dict(model, payload["model"])
    return model, config, payload


def ensure_initialized_state(state_dir: Path, seed: int, input_dim: int, hidden_dim: int, num_classes: int):
    if config_file(state_dir).exists() and state_file(state_dir).exists():
        return
    ensure_dir(state_dir)
    args = argparse.Namespace(
        state_dir=str(state_dir),
        output=str(state_dir / "outputs" / "auto_init.txt"),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        seed=seed,
    )
    init_state(args)


def local_train(args):
    state_dir = Path(args.state_dir)
    model, config, payload = load_model_and_config(state_dir)
    dataset = generate_vehicle_dataset(
        args.node_id,
        args.sample_count,
        config["input_dim"],
        config["num_classes"],
        config["seed"] + args.round * 31,
    )
    loader = DataLoader(dataset, batch_size=min(32, args.sample_count), shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    baseline_model = build_model(config)
    apply_flattened_state_dict(baseline_model, payload["model"])

    model.train()
    for _ in range(args.local_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    local_loss, local_accuracy = evaluate_model(model, dataset)
    baseline_flat = flatten_state_dict(baseline_model.state_dict())
    trained_flat = flatten_state_dict(model.state_dict())
    delta = [trained_flat[i] - baseline_flat[i] for i in range(len(trained_flat))]
    update_norm = math.sqrt(sum(value * value for value in delta))
    sparse_indices, sparse_values = compress_delta(delta, args.compression_ratio)

    update_payload = {
        "round": args.round,
        "node_id": args.node_id,
        "sample_count": args.sample_count,
        "local_loss": local_loss,
        "local_accuracy": local_accuracy,
        "update_norm": update_norm,
        "delta": delta,
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
    }
    update_path = Path(args.output_update)
    ensure_dir(update_path.parent)
    save_json(update_path, update_payload)
    return {
        "status": "ok",
        "round": args.round,
        "node_id": args.node_id,
        "sample_count": args.sample_count,
        "local_loss": local_loss,
        "local_accuracy": local_accuracy,
        "update_norm": update_norm,
        "update_path": str(update_path),
        "sparse_indices": serialize_int_list(sparse_indices),
        "sparse_values": serialize_float_list(sparse_values),
        "model_score": local_accuracy,
    }


def aggregate(args):
    state_dir = Path(args.state_dir)
    model, config, payload = load_model_and_config(state_dir)
    manifest = load_json(Path(args.manifest))
    updates = manifest.get("updates", [])
    if not updates:
        result = {
            "status": "ok",
            "received_updates": 0,
            "global_loss": payload["global_loss"],
            "global_accuracy": payload["global_accuracy"],
            "model_score": payload["model_score"],
        }
        return result

    weighted_delta = None
    total_samples = 0
    for entry in updates:
        sample_count = int(entry["sample_count"])
        if "sparse_indices" in entry and "sparse_values" in entry:
            indices = parse_int_list(entry["sparse_indices"])
            values = parse_float_list(entry["sparse_values"])
            delta = torch.zeros(len(payload["model"]), dtype=torch.float32)
            for index, value in zip(indices, values):
                if 0 <= index < delta.numel():
                    delta[index] = float(value)
        else:
            update_payload = load_json(Path(entry["update_path"]))
            delta = torch.tensor(update_payload["delta"], dtype=torch.float32)
        if weighted_delta is None:
            weighted_delta = delta * sample_count
        else:
            weighted_delta += delta * sample_count
        total_samples += sample_count

    current_flat = torch.tensor(payload["model"], dtype=torch.float32)
    averaged_delta = weighted_delta / max(1, total_samples)
    new_flat = current_flat + averaged_delta
    apply_flattened_state_dict(model, new_flat.tolist())
    sparse_indices, sparse_values = compress_delta(averaged_delta.tolist(), args.compression_ratio)

    validation = generate_validation_dataset(config["input_dim"], config["num_classes"], config["seed"])
    global_loss, global_accuracy = evaluate_model(model, validation)
    new_payload = {
        "round": args.round,
        "model": flatten_state_dict(model.state_dict()),
        "global_loss": global_loss,
        "global_accuracy": global_accuracy,
        "model_score": global_accuracy,
    }
    save_json(state_file(state_dir), new_payload)
    return {
        "status": "ok",
        "round": args.round,
        "received_updates": len(updates),
        "global_loss": global_loss,
        "global_accuracy": global_accuracy,
        "sparse_indices": serialize_int_list(sparse_indices),
        "sparse_values": serialize_float_list(sparse_values),
        "model_score": global_accuracy,
    }


def apply_global(args):
    state_dir = Path(args.state_dir)
    ensure_initialized_state(
        state_dir,
        args.seed,
        args.input_dim,
        args.hidden_dim,
        args.num_classes,
    )
    model, config, payload = load_model_and_config(state_dir)
    current_flat = torch.tensor(payload["model"], dtype=torch.float32)
    delta = torch.zeros_like(current_flat)
    indices = parse_int_list(args.sparse_indices)
    values = parse_float_list(args.sparse_values)
    for index, value in zip(indices, values):
        if 0 <= index < delta.numel():
            delta[index] = float(value)
    new_flat = current_flat + delta
    apply_flattened_state_dict(model, new_flat.tolist())

    validation = generate_validation_dataset(config["input_dim"], config["num_classes"], config["seed"])
    global_loss, global_accuracy = evaluate_model(model, validation)
    new_payload = {
        "round": args.round,
        "model": flatten_state_dict(model.state_dict()),
        "global_loss": global_loss,
        "global_accuracy": global_accuracy,
        "model_score": global_accuracy,
    }
    save_json(state_file(state_dir), new_payload)
    return {
        "status": "ok",
        "round": args.round,
        "global_loss": global_loss,
        "global_accuracy": global_accuracy,
        "model_score": global_accuracy,
    }


def write_key_value_output(path: Path, payload):
    ensure_dir(path.parent)
    lines = []
    for key, value in payload.items():
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--state-dir", required=True)
    init_parser.add_argument("--output", required=True)
    init_parser.add_argument("--input-dim", type=int, default=16)
    init_parser.add_argument("--hidden-dim", type=int, default=32)
    init_parser.add_argument("--num-classes", type=int, default=4)
    init_parser.add_argument("--seed", type=int, default=20260312)

    local_parser = subparsers.add_parser("local-train")
    local_parser.add_argument("--state-dir", required=True)
    local_parser.add_argument("--output", required=True)
    local_parser.add_argument("--output-update", required=True)
    local_parser.add_argument("--node-id", type=int, required=True)
    local_parser.add_argument("--round", type=int, required=True)
    local_parser.add_argument("--sample-count", type=int, required=True)
    local_parser.add_argument("--local-epochs", type=int, default=1)
    local_parser.add_argument("--learning-rate", type=float, default=0.001)
    local_parser.add_argument("--compression-ratio", type=float, default=0.05)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--state-dir", required=True)
    aggregate_parser.add_argument("--output", required=True)
    aggregate_parser.add_argument("--manifest", required=True)
    aggregate_parser.add_argument("--round", type=int, required=True)
    aggregate_parser.add_argument("--compression-ratio", type=float, default=0.05)

    apply_parser = subparsers.add_parser("apply-global")
    apply_parser.add_argument("--state-dir", required=True)
    apply_parser.add_argument("--output", required=True)
    apply_parser.add_argument("--round", type=int, required=True)
    apply_parser.add_argument("--sparse-indices", required=True)
    apply_parser.add_argument("--sparse-values", required=True)
    apply_parser.add_argument("--seed", type=int, default=20260312)
    apply_parser.add_argument("--input-dim", type=int, default=16)
    apply_parser.add_argument("--hidden-dim", type=int, default=32)
    apply_parser.add_argument("--num-classes", type=int, default=4)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "init":
        payload = init_state(args)
    elif args.command == "local-train":
        payload = local_train(args)
    elif args.command == "aggregate":
        payload = aggregate(args)
    elif args.command == "apply-global":
        payload = apply_global(args)
    else:
        raise ValueError(args.command)
    write_key_value_output(Path(args.output), payload)


if __name__ == "__main__":
    main()
