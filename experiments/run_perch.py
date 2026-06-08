"""Preflight Perch 2.0 / Perch-Hoplite evaluation on BioDCASE labels."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

from experiments import runtime
from experiments.run_baseline import resolve_dataset_root


MPL_CONFIG_DIR = runtime.PROJECT_ROOT / "output/matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

REQUIRED_MODULES = [
  "jax",
  "flax",
  "tensorflow",
  "tf_keras",
  "ml_collections",
  "etils",
  "apache_beam",
  "aqt",
  "perch_hoplite",
]

PERCH_MODULES = [
  "chirp",
  "chirp.models.perch_2",
  "chirp.inference.embed_lib",
]

HOPLITE_MODULES = [
  "perch_hoplite.zoo.model_configs",
  "perch_hoplite.zoo.zoo_interface",
]


class Tee:
  """Write stream output to multiple destinations."""

  def __init__(self, *streams):
    self.streams = streams

  def write(self, data):
    for stream in self.streams:
      stream.write(data)
      stream.flush()

  def flush(self):
    for stream in self.streams:
      stream.flush()

  def close(self):
    # absl logging may call close() on sys.stderr at interpreter shutdown.
    # The real streams are owned by the process, so keep this as a no-op.
    pass


@contextlib.contextmanager
def tee_to_file(path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", buffering=1) as f:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, f)
    sys.stderr = Tee(original_stderr, f)
    try:
      yield
    finally:
      sys.stdout = original_stdout
      sys.stderr = original_stderr


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    default="experiments/configs/perch2_eval.yaml",
    help="Perch evaluation YAML config.",
  )
  parser.add_argument(
    "--mode",
    choices=["preflight", "label-map", "hoplite-smoke", "train-head"],
    default="preflight",
    help="Which Perch setup stage to run.",
  )
  parser.add_argument(
    "--dataset-root",
    default=None,
    help="Override dataset root. Parent dirs with one nested Train/Validation child are auto-resolved.",
  )
  parser.add_argument(
    "--perch-repo",
    default=None,
    help="Override local google-research/perch checkout.",
  )
  parser.add_argument(
    "--model-choice",
    default=None,
    help="Override Perch-Hoplite preset, for example perch_8.",
  )
  parser.add_argument(
    "--num-epochs",
    type=int,
    default=10,
    help="Number of epochs for the frozen-Perch embedding classifier head.",
  )
  parser.add_argument(
    "--max-files-per-class",
    type=int,
    default=None,
    help="Optional balanced cap per class and split for local CPU smoke runs.",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for the classifier head.",
  )
  return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
  config = runtime.expand_config(runtime.load_yaml(runtime.resolve_project_path(args.config)))
  if args.dataset_root:
    config.setdefault("dataset", {})["root_path"] = args.dataset_root
  if args.perch_repo:
    config.setdefault("perch", {})["repo_root"] = args.perch_repo
  if args.model_choice:
    config.setdefault("perch", {})["model_choice"] = args.model_choice

  root = config.get("dataset", {}).get("root_path")
  resolved_root = resolve_dataset_root(root)
  if resolved_root is not None:
    config.setdefault("dataset", {})["root_path"] = str(resolved_root)
  return config


def add_perch_repo_to_path(config: dict[str, Any]) -> Path | None:
  repo_root = config.get("perch", {}).get("repo_root")
  if not repo_root:
    return None
  path = runtime.resolve_project_path(repo_root)
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))
  return path


def import_status(modules: list[str]) -> dict[str, str]:
  status = {}
  for module in modules:
    try:
      imported = importlib.import_module(module)
      status[module] = f"ok {getattr(imported, '__version__', '')}".strip()
    except Exception as exc:
      status[module] = f"missing: {exc.__class__.__name__}: {exc}"
  return status


def missing_modules(status: dict[str, str]) -> list[str]:
  return [name for name, value in status.items() if not value.startswith("ok")]


def split_labels(dataset_path: Path | None, split: str) -> list[str]:
  if dataset_path is None:
    return []
  split_dir = dataset_path / split
  if not split_dir.is_dir():
    return []
  return sorted(child.name for child in split_dir.iterdir() if child.is_dir())


def split_examples(
    dataset_path: Path,
    split: str,
    labels: list[str],
    max_files_per_class: int | None = None,
) -> list[tuple[Path, int]]:
  examples = []
  for label_idx, label in enumerate(labels):
    wavs = sorted((dataset_path / split / label).glob("*.wav"))
    if max_files_per_class is not None:
      wavs = wavs[:max_files_per_class]
    examples.extend((path, label_idx) for path in wavs)
  return examples


def dataset_status(config: dict[str, Any]) -> dict[str, Any]:
  dataset_root = config.get("dataset", {}).get("root_path")
  dataset_path = resolve_dataset_root(dataset_root)
  return {
    "root": str(dataset_path) if dataset_path else None,
    "exists": dataset_path.is_dir() if dataset_path else False,
    "train_labels": split_labels(dataset_path, "Train"),
    "validation_labels": split_labels(dataset_path, "Validation"),
  }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
  perch_repo = add_perch_repo_to_path(config)
  core_status = import_status(REQUIRED_MODULES)
  perch_status = import_status(PERCH_MODULES)
  hoplite_status = import_status(HOPLITE_MODULES)
  combined_status = {**core_status, **perch_status, **hoplite_status}
  return {
    "python_supported_by_upstream_pyproject": sys.version_info < (3, 12),
    "python_version": sys.version,
    "perch_repo": str(perch_repo) if perch_repo else None,
    "perch_repo_exists": perch_repo.is_dir() if perch_repo else False,
    "dataset": dataset_status(config),
    "module_status": combined_status,
    "missing_modules": missing_modules(combined_status),
    "imports_ok": not missing_modules(combined_status),
    "model_choice": config.get("perch", {}).get("model_choice"),
  }


def label_map(config: dict[str, Any]) -> dict[str, Any]:
  dataset = dataset_status(config)
  mapping = config.get("label_mapping", {})
  train_labels = dataset["train_labels"]
  validation_labels = dataset["validation_labels"]
  dataset_labels = sorted(set(train_labels) | set(validation_labels))
  missing_mapping = [label for label in dataset_labels if label not in mapping]
  unused_mapping = [label for label in sorted(mapping) if label not in dataset_labels]
  mapped_species = {
    label: {
      "scientific_name": mapping[label].get("scientific_name"),
      "common_name": mapping[label].get("common_name"),
      "perch_strategy": mapping[label].get("perch_strategy", "species_logit"),
    }
    for label in dataset_labels
    if label in mapping
  }
  return {
    "dataset": dataset,
    "num_dataset_labels": len(dataset_labels),
    "dataset_labels": dataset_labels,
    "mapped_species": mapped_species,
    "missing_mapping": missing_mapping,
    "unused_mapping": unused_mapping,
    "mapping_ok": not missing_mapping,
  }


def hoplite_smoke(config: dict[str, Any]) -> dict[str, Any]:
  add_perch_repo_to_path(config)
  status = import_status(HOPLITE_MODULES)
  missing = missing_modules(status)
  if missing:
    return {
      "status": "blocked",
      "module_status": status,
      "missing_modules": missing,
    }

  from perch_hoplite.zoo import model_configs

  model_choice = config.get("perch", {}).get("model_choice", "perch_8")
  try:
    model_config = model_configs.get_preset_model_config(model_choice)
  except Exception as exc:
    return {
      "status": "failed",
      "model_choice": model_choice,
      "error": f"{exc.__class__.__name__}: {exc}",
    }

  public_attrs = {
    name: repr(value)
    for name, value in vars(model_config).items()
    if not name.startswith("_") and isinstance(value, (str, int, float, bool, tuple, list, dict, type(None)))
  }
  return {
    "status": "ok",
    "model_choice": model_choice,
    "model_config_type": type(model_config).__name__,
    "model_config_attrs": public_attrs,
  }


def dataset_cache_key(
    dataset_path: Path,
    model_choice: str,
    max_files_per_class: int | None,
) -> str:
  raw = f"{dataset_path.resolve()}|{model_choice}|{max_files_per_class}"
  return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def read_audio(path: Path, sample_rate: int):
  import librosa
  import numpy as np
  import soundfile as sf

  wav, native_sr = sf.read(path)
  if wav.ndim == 2:
    wav = wav.mean(axis=1)
  wav = wav.astype(np.float32)
  if native_sr != sample_rate:
    wav = librosa.resample(wav, orig_sr=native_sr, target_sr=sample_rate)
  return wav.astype(np.float32), int(native_sr)


def embed_examples(
    model,
    examples: list[tuple[Path, int]],
    split_name: str,
) -> tuple[Any, Any, dict[str, Any]]:
  import numpy as np

  embeddings = []
  labels = []
  native_sample_rates = set()
  for idx, (path, label_idx) in enumerate(examples, start=1):
    if idx == 1 or idx % 25 == 0 or idx == len(examples):
      print(f"[{split_name}] embedding {idx}/{len(examples)}: {path}")
    wav, native_sr = read_audio(path, model.sample_rate)
    native_sample_rates.add(native_sr)
    outputs = model.batch_embed(wav[np.newaxis, :])
    if outputs.embeddings is None:
      raise RuntimeError(f"Perch produced no embeddings for {path}")
    pooled = outputs.embeddings.mean(axis=(1, 2))[0]
    embeddings.append(pooled.astype(np.float32))
    labels.append(label_idx)
  return (
    np.stack(embeddings, axis=0),
    np.array(labels, dtype=np.int64),
    {
      "num_examples": len(examples),
      "native_sample_rates": sorted(native_sample_rates),
    },
  )


def load_or_create_embeddings(
    config: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Any, Any, Any, Any, list[str], dict[str, Any]]:
  import numpy as np
  from perch_hoplite.zoo import model_configs

  dataset_root = config.get("dataset", {}).get("root_path")
  dataset_path = resolve_dataset_root(dataset_root)
  if dataset_path is None or not dataset_path.is_dir():
    raise RuntimeError(f"Dataset root not found: {dataset_root}")

  labels = split_labels(dataset_path, "Train")
  validation_labels = split_labels(dataset_path, "Validation")
  if labels != validation_labels:
    raise RuntimeError(
      "Train and Validation labels differ; refusing to train a head with ambiguous class ids."
    )

  model_choice = config.get("perch", {}).get("model_choice", "perch_8")
  cache_key = dataset_cache_key(dataset_path, model_choice, args.max_files_per_class)
  cache_dir = runtime.resolve_project_path(config["experiment"]["output_root"])
  cache_path = cache_dir / "perch2_eval" / "embedding_cache" / f"{cache_key}.npz"
  metadata = {
    "cache_path": str(cache_path),
    "cache_hit": cache_path.is_file(),
    "model_choice": model_choice,
    "max_files_per_class": args.max_files_per_class,
  }
  if cache_path.is_file():
    print(f"Loading cached Perch embeddings: {cache_path}")
    cached = np.load(cache_path, allow_pickle=True)
    metadata["train_info"] = cached["train_info"].item()
    metadata["validation_info"] = cached["validation_info"].item()
    return (
      cached["x_train"],
      cached["y_train"],
      cached["x_validation"],
      cached["y_validation"],
      list(cached["labels"]),
      metadata,
    )

  print(f"Loading Perch model preset: {model_choice}")
  preset = model_configs.get_preset_model_config(model_choice)
  model = preset.load_model()
  print(
    f"Loaded Perch model: sample_rate={model.sample_rate}, "
    f"embedding_dim={preset.embedding_dim}"
  )

  train_examples = split_examples(
    dataset_path, "Train", labels, args.max_files_per_class
  )
  validation_examples = split_examples(
    dataset_path, "Validation", labels, args.max_files_per_class
  )
  x_train, y_train, train_info = embed_examples(model, train_examples, "Train")
  x_validation, y_validation, validation_info = embed_examples(
    model, validation_examples, "Validation"
  )

  cache_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    cache_path,
    x_train=x_train,
    y_train=y_train,
    x_validation=x_validation,
    y_validation=y_validation,
    labels=np.array(labels),
    train_info=train_info,
    validation_info=validation_info,
  )
  metadata["train_info"] = train_info
  metadata["validation_info"] = validation_info
  print(f"Saved Perch embedding cache: {cache_path}")
  return x_train, y_train, x_validation, y_validation, labels, metadata


def train_head(
    config: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
  import numpy as np
  from sklearn.metrics import accuracy_score, roc_auc_score
  import tensorflow as tf

  add_perch_repo_to_path(config)
  x_train, y_train, x_validation, y_validation, labels, embedding_metadata = (
    load_or_create_embeddings(config, args, run_dir)
  )
  tf.random.set_seed(1337)
  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1],)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(labels), activation="softmax"),
  ])
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
  )
  print("=== Perch Embedding Head ===")
  print(f"train shape: {x_train.shape}, validation shape: {x_validation.shape}")
  print(f"labels: {labels}")
  history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=args.num_epochs,
    batch_size=args.batch_size,
    verbose=2,
  )
  y_prob = model.predict(x_validation, batch_size=args.batch_size, verbose=0)
  y_pred = np.argmax(y_prob, axis=1)
  accuracy = float(accuracy_score(y_validation, y_pred))
  try:
    roc_auc = float(
      roc_auc_score(
        y_validation,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=list(range(len(labels))),
      )
    )
  except ValueError as exc:
    roc_auc = None
    print(f"Validation ROC-AUC unavailable: {exc}")

  model_path = run_dir / "models" / "perch_embedding_head.keras"
  model_path.parent.mkdir(parents=True, exist_ok=True)
  model.save(model_path)
  return {
    "status": "ok",
    "model_choice": config.get("perch", {}).get("model_choice", "perch_8"),
    "num_epochs": args.num_epochs,
    "batch_size": args.batch_size,
    "max_files_per_class": args.max_files_per_class,
    "labels": labels,
    "num_classes": len(labels),
    "embedding_shape": list(x_train.shape[1:]),
    "dataset_lengths": {
      "train": int(len(y_train)),
      "validation": int(len(y_validation)),
    },
    "history": {
      key: [float(value) for value in values]
      for key, values in history.history.items()
    },
    "final_epoch": {
      key: float(values[-1])
      for key, values in history.history.items()
      if values
    },
    "validation_accuracy": accuracy,
    "validation_roc_auc_macro_ovr": roc_auc,
    "model_path": str(model_path),
    "embedding_metadata": embedding_metadata,
  }


def train_log_header(config: dict[str, Any], args: argparse.Namespace, run_dir: Path):
  header = {
    "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    "command": " ".join(sys.argv),
    "mode": args.mode,
    "run_dir": str(run_dir),
    "git": {
      "branch": runtime.git_branch(),
      "commit": runtime.git_commit(),
      "dirty": runtime.git_dirty(),
    },
    "host": {
      "platform": platform.platform(),
      "python": platform.python_version(),
    },
    "dataset": config.get("dataset", {}),
    "perch": config.get("perch", {}),
    "training": {
      "num_epochs": args.num_epochs,
      "batch_size": args.batch_size,
      "max_files_per_class": args.max_files_per_class,
    },
  }
  import yaml

  print("=== Perch Head Training Run ===")
  print(yaml.dump(header, default_flow_style=False, sort_keys=False))
  print("=== Terminal Output ===")


def write_run(
    config: dict[str, Any],
    mode: str,
    results: dict[str, Any],
    run_dir: Path | None = None,
) -> Path:
  run_dir = run_dir or runtime.make_run_dir(config, mode)
  record = runtime.run_metadata(config, mode)
  record["results"] = results
  runtime.dump_yaml(record, run_dir / "run.yaml")

  module_status = results.get("module_status", {})
  missing = results.get("missing_modules") or missing_modules(module_status)
  dataset = results.get("dataset", {})
  row = {
    "timestamp": record["timestamp"],
    "experiment_id": config["experiment"]["id"],
    "mode": mode,
    "branch": record["git"]["branch"],
    "commit": record["git"]["commit"],
    "run_dir": str(run_dir),
    "perch.imports_ok": results.get("imports_ok"),
    "perch.dataset_exists": dataset.get("exists"),
    "perch.model_choice": results.get("model_choice") or config.get("perch", {}).get("model_choice"),
    "perch.mapped_labels": len(results.get("mapped_species", {})) or None,
    "perch.mapping_ok": results.get("mapping_ok"),
    "perch.missing_modules": ", ".join(missing),
    "perch.status": results.get("status"),
    "perch.train.validation_accuracy": results.get("validation_accuracy"),
    "perch.train.validation_roc_auc_macro_ovr": results.get("validation_roc_auc_macro_ovr"),
    "perch.train.num_epochs": results.get("num_epochs"),
    "perch.train.dataset_lengths": results.get("dataset_lengths"),
    "perch.train.max_files_per_class": results.get("max_files_per_class"),
    "perch.train.model_path": results.get("model_path"),
  }
  runtime.append_summary(row, config)
  return run_dir


def main() -> None:
  args = parse_args()
  config = load_config(args)
  if args.mode == "preflight":
    results = preflight(config)
  elif args.mode == "label-map":
    results = label_map(config)
  elif args.mode == "hoplite-smoke":
    results = hoplite_smoke(config)
  else:
    run_dir = runtime.make_run_dir(config, args.mode)
    log_path = run_dir / "train.log"
    with tee_to_file(log_path):
      train_log_header(config, args, run_dir)
      try:
        results = train_head(config, args, run_dir)
      except Exception as exc:
        failure = {
          "status": "failed",
          "error": repr(exc),
          "traceback": traceback.format_exc(),
          "log_path": str(log_path),
        }
        write_run(config, args.mode, failure, run_dir=run_dir)
        print("=== Perch Head Training Failed ===")
        traceback.print_exc()
        print(f"Feedback log: {log_path}")
        raise
      results["log_path"] = str(log_path)
      write_run(config, args.mode, results, run_dir=run_dir)
    print(f"Wrote run record: {run_dir / 'run.yaml'}")
    return

  run_dir = write_run(config, args.mode, results)
  print(f"Wrote run record: {run_dir / 'run.yaml'}")


if __name__ == "__main__":
  main()
