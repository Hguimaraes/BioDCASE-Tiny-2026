"""Run Mel+MSS experiments with Perch teacher soft-logit distillation."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torchsummary import summary

from experiments import runtime
from experiments.mel_mss_datamodule import DatamoduleMelMSS
from experiments.run_mel_mss import (
  feature_smoke,
  load_config as load_mel_mss_config,
  model_smoke,
  preflight as mel_mss_preflight,
  tee_to_file,
  training_log_footer,
  training_log_header,
  write_run,
)
from pipeline_pytorch.model_training import run_model_testing, run_model_training
from pipeline_pytorch.pytorch_datamodule import DataloaderPytorch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    default="experiments/configs/mel_mss_logit_distill.yaml",
    help="Mel+MSS logit-distillation experiment YAML config.",
  )
  parser.add_argument(
    "--mode",
    choices=["preflight", "feature-smoke", "model-smoke", "train"],
    default="preflight",
    help="Which experiment stage to run.",
  )
  parser.add_argument(
    "--dataset-root",
    default=None,
    help="Override dataset root. Parent dirs with one nested Train/Validation child are auto-resolved.",
  )
  parser.add_argument(
    "--teacher-soft-labels",
    default=None,
    help="Path to teacher_soft_labels.npz exported by experiments.run_perch.",
  )
  parser.add_argument(
    "--num-epochs",
    type=int,
    default=None,
    help="Override pytorch_framework.model_training.num_epochs.",
  )
  return parser.parse_args()


def relative_dataset_key(path: str | Path) -> str:
  """Return a stable Train/... or Validation/... key independent of root path."""
  parts = Path(path).parts
  for split in ("Train", "Validation", "Test"):
    if split in parts:
      split_index = parts.index(split)
      return str(Path(*parts[split_index:]))
  return str(Path(path).name)


def ordered_labels(label_dict: dict[str, int]) -> list[str]:
  return [label for label, _ in sorted(label_dict.items(), key=lambda item: item[1])]


def resolve_teacher_path(config: dict[str, Any], cli_path: str | None) -> Path:
  teacher_path = cli_path or config.get("distillation", {}).get("teacher_soft_labels_path")
  if not teacher_path or "${" in str(teacher_path):
    raise ValueError(
      "Teacher soft labels are required. Pass --teacher-soft-labels or set "
      "distillation.teacher_soft_labels_path in the config."
    )
  return runtime.resolve_project_path(teacher_path)


def load_teacher_archive(path: Path) -> dict[str, Any]:
  if not path.is_file():
    raise FileNotFoundError(f"Teacher soft-label archive not found: {path}")
  with np.load(path, allow_pickle=False) as data:
    return {key: data[key].copy() for key in data.files}


def validate_teacher_labels(teacher: dict[str, Any], label_dict: dict[str, int]) -> None:
  teacher_labels = [str(label) for label in teacher["labels"].tolist()]
  student_labels = ordered_labels(label_dict)
  if teacher_labels != student_labels:
    raise ValueError(
      "Teacher label order does not match the student datamodule.\n"
      f"Teacher: {teacher_labels}\nStudent: {student_labels}"
    )


def teacher_split_for_datamodule(datamodule: DatamoduleMelMSS) -> str:
  split = datamodule.cfg["load_set_on_init"]
  if split == "test" and datamodule.cfg["test_folder"] == datamodule.cfg["validation_folder"]:
    return "validation"
  return split


def teacher_logits_by_sid(datamodule: DatamoduleMelMSS, teacher: dict[str, Any]) -> np.ndarray:
  split = teacher_split_for_datamodule(datamodule)
  path_key = f"{split}_paths"
  logits_key = f"{split}_logits"
  if path_key not in teacher or logits_key not in teacher:
    raise KeyError(f"Teacher archive does not contain {path_key!r} and {logits_key!r}")

  teacher_by_key = {
    relative_dataset_key(path): logits
    for path, logits in zip(teacher[path_key], teacher[logits_key])
  }

  num_samples = len(datamodule.get_cache_info()["files"]["dataset"])
  num_classes = teacher[logits_key].shape[1]
  logits_by_sid = np.empty((num_samples, num_classes), dtype=np.float32)
  found = np.zeros((num_samples,), dtype=bool)

  for sid, dataset_path in enumerate(datamodule.get_cache_info()["files"]["dataset"]):
    key = relative_dataset_key(dataset_path)
    if key in teacher_by_key:
      logits_by_sid[sid] = teacher_by_key[key]
      found[sid] = True

  loaded_sids = np.asarray(datamodule.sample_ids, dtype=np.int64)
  missing = [int(sid) for sid in loaded_sids if not found[int(sid)]]
  if missing:
    examples = [datamodule.get_file_names_by_single_sid(sid)[0] for sid in missing[:5]]
    raise ValueError(
      f"Missing teacher logits for {len(missing)} {split} samples. "
      f"Examples: {examples}"
    )

  return logits_by_sid


class DataloaderPytorchWithTeacherLogits(DataloaderPytorch):
  """Datamodule wrapper that appends teacher logits to each batch item."""

  def __init__(self, datamodule, teacher_logits):
    super().__init__(datamodule)
    self.teacher_logits = torch.from_numpy(teacher_logits.astype(np.float32))

  def __getitem__(self, idx):
    x, y, sid = super().__getitem__(idx)
    return x, y, sid, self.teacher_logits[int(sid)]


def build_dataloader(datamodule, teacher, dataloader_kwargs):
  logits_by_sid = teacher_logits_by_sid(datamodule, teacher)
  dataset = DataloaderPytorchWithTeacherLogits(datamodule, logits_by_sid)
  return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def preflight(config: dict[str, Any], teacher_path: Path) -> dict[str, Any]:
  results = mel_mss_preflight(config)
  results["distillation"] = {
    "teacher_soft_labels_path": str(teacher_path),
    "teacher_soft_labels_exists": teacher_path.is_file(),
    "loss": config["training_config"]["pytorch_framework"]["model"]["kwargs"].get("distillation", {}),
  }
  if teacher_path.is_file():
    teacher = load_teacher_archive(teacher_path)
    results["distillation"].update({
      "teacher_archive_keys": sorted(teacher.keys()),
      "teacher_labels": [str(label) for label in teacher["labels"].tolist()],
      "train_logits_shape": list(teacher["train_logits"].shape),
      "validation_logits_shape": list(teacher["validation_logits"].shape),
    })
  return results


def train(config: dict[str, Any], teacher_path: Path, run_dir: Path) -> dict[str, Any]:
  training_cfg = config["training_config"]
  training_cfg["pytorch_framework"]["model"]["kwargs"]["save_path"] = str(run_dir / "models")
  teacher = load_teacher_archive(teacher_path)

  datamodule_train = DatamoduleMelMSS(training_cfg["datamodule"], load_set_on_init="train")
  datamodule_validation = DatamoduleMelMSS(training_cfg["datamodule"], load_set_on_init="validation")
  datamodule_test = DatamoduleMelMSS(training_cfg["datamodule"], load_set_on_init="test")
  validate_teacher_labels(teacher, datamodule_train.get_label_dict())

  framework_cfg = training_cfg["pytorch_framework"]
  dataloader_train = build_dataloader(datamodule_train, teacher, framework_cfg["dataloader_train_kwargs"])
  dataloader_validation = build_dataloader(
    datamodule_validation,
    teacher,
    framework_cfg["dataloader_validation_and_test_kwargs"],
  )
  dataloader_test = build_dataloader(datamodule_test, teacher, framework_cfg["dataloader_validation_and_test_kwargs"])

  input_shape = datamodule_train.get_feature_shape_at_load()
  model_cfg = framework_cfg["model"]
  model_class = getattr(importlib.import_module(model_cfg["module"]), model_cfg["attr"])
  model = model_class(
    *model_cfg.get("args", []),
    **{
      "save_path": str(run_dir / "models"),
      **model_cfg.get("kwargs", {}),
      "input_shape": input_shape,
      "num_classes": len(datamodule_train.get_label_dict()),
    },
  )
  summary(model, input_size=input_shape, device=model.get_device_type_str())

  training_history = run_model_training(
    framework_cfg,
    model,
    dataloader_train,
    dataloader_validation,
    label_dict=datamodule_train.get_label_dict(),
  )

  evaluate_checkpoint = framework_cfg["model_training"].get("evaluate_checkpoint", "final")
  if evaluate_checkpoint != "final":
    checkpoint_info = model.best_checkpoints.get(evaluate_checkpoint)
    if checkpoint_info and Path(checkpoint_info["path"]).is_file():
      print(f"Load {evaluate_checkpoint} checkpoint for testing: {checkpoint_info['path']}")
      model.load(checkpoint_info["path"])
      model.evaluated_checkpoint = {"name": evaluate_checkpoint, **checkpoint_info}
    else:
      print(f"***Requested checkpoint [{evaluate_checkpoint}] was not found; testing final model.")
      model.evaluated_checkpoint = {"name": "final", "path": str(model.get_model_file_path())}
  else:
    model.evaluated_checkpoint = {"name": "final", "path": str(model.get_model_file_path())}

  test_metrics = run_model_testing(
    framework_cfg,
    model,
    dataloader_test,
    label_dict=datamodule_test.get_label_dict(),
  )
  test_metrics["evaluated_checkpoint"] = model.evaluated_checkpoint
  model.training_history = training_history
  model.test_metrics = test_metrics

  cache_info = datamodule_train.get_cache_info()
  return {
    "teacher_soft_labels_path": str(teacher_path),
    "distillation": framework_cfg["model"]["kwargs"].get("distillation", {}),
    "input_shape": list(datamodule_train.get_feature_shape_at_load()),
    "mel_shape": cache_info.get("mel_shape"),
    "mss_shape": cache_info.get("mss_shape"),
    "split_index": cache_info.get("split_index"),
    "num_classes": len(datamodule_train.get_label_dict()),
    "dataset_lengths": {
      "train": len(datamodule_train),
      "validation": len(datamodule_validation),
      "test": len(datamodule_test),
    },
    "model_path": str(model.get_model_file_path()),
    "tflite_path": str(model.get_tflite_model_file_path()),
    "num_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    "training_history": getattr(model, "training_history", []),
    "final_epoch": getattr(model, "training_history", [])[-1] if getattr(model, "training_history", []) else None,
    "test_metrics": getattr(model, "test_metrics", {}),
    "best_checkpoints": getattr(model, "best_checkpoints", {}),
  }


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
  config = load_mel_mss_config(args.config, args.dataset_root, args.num_epochs)
  teacher_path = resolve_teacher_path(config, args.teacher_soft_labels)
  config.setdefault("distillation", {})["teacher_soft_labels_path"] = str(teacher_path)
  return config, teacher_path


def main() -> None:
  args = parse_args()
  config, teacher_path = load_config(args)

  if args.mode == "preflight":
    results = preflight(config, teacher_path)
    write_run(config, args.mode, results)
  elif args.mode == "feature-smoke":
    results = feature_smoke(config)
    write_run(config, args.mode, results)
  elif args.mode == "model-smoke":
    results = model_smoke(config)
    write_run(config, args.mode, results)
  else:
    run_dir = runtime.make_run_dir(config, args.mode)
    log_path = run_dir / "train.log"
    with tee_to_file(log_path):
      training_log_header(config, args, run_dir)
      print("=== Distillation Teacher ===")
      print(yaml.dump({"teacher_soft_labels_path": str(teacher_path)}, sort_keys=False))
      try:
        results = train(config, teacher_path, run_dir)
      except Exception:
        import traceback

        failure = {
          "status": "failed",
          "teacher_soft_labels_path": str(teacher_path),
          "traceback": traceback.format_exc(),
          "log_path": str(log_path),
        }
        write_run(config, args.mode, failure, run_dir=run_dir)
        print("=== Training Failed ===")
        traceback.print_exc()
        print(f"Run YAML: {run_dir / 'run.yaml'}")
        print(f"Feedback log: {log_path}")
        raise
      results["log_path"] = str(log_path)
      training_log_footer(results, run_dir)
      write_run(config, args.mode, results, run_dir=run_dir)


if __name__ == "__main__":
  sys.exit(main())
