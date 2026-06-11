# --
# model training

import sys
import math
import copy
import yaml
import torch
import numpy as np
import importlib

from torchsummary import summary
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

# add root path of project if called as main
if __name__ == '__main__': [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from plots import plot_confusion_matrix
from pipeline_pytorch.paths import MODELS_DIR, CM_FIG_PATH
from pipeline_pytorch.pytorch_datamodule import DataloaderPytorch
from pipeline_pytorch.augmentation import FeatureAugmentDataset, mixup_batch
from pipeline_pytorch.distillation import TeacherLogitDataset, ContextDataset, load_teacher_logits, load_msab, mixup_with_teacher, distillation_loss
from pipeline_pytorch.model_tiny_ml import Baseline


# default training recipe (overridable via config: pytorch_framework.training_recipe)
RECIPE_DEFAULTS = {
  'augmentation': {'enabled': True},
  'mixup': {'alpha': 0.2, 'p': 0.5},
  'scheduler': {'name': 'cosine', 'warmup_epochs': 5, 'min_lr_factor': 0.05},
  'best_checkpoint_metric': 'val_auc',
  'early_stopping_patience': 0,
  # logit distillation from the Perch teacher (track C1); disabled by default
  'distillation': {'enabled': False, 'teacher_dir': '', 'alpha': 0.5, 'temperature': 3.0,
                   'label_smoothing': 0.1, 'train_split': 'Train'},
  # MSAB context for FiLM students (track C2); used only if the model needs it
  'context': {'msab_dir': '', 'train_split': 'Train', 'eval_split': 'Validation'},
}


def context_of(model, data):
  """
  MSAB context tensor (always the last batch element) for FiLM students that
  declare needs_context, else None. The dataset only appends ctx for such
  models, so needs_context <=> ctx is data[-1].
  """
  if getattr(model, 'needs_context', False):
    return data[-1].to(device=model.device, dtype=torch.float32)
  return None


def forward_with_ctx(model, x, ctx):
  """call model.forward(x) or model.forward(x, ctx) depending on the model"""
  return model.forward(x, ctx) if ctx is not None else model.forward(x)


def make_scheduler(cfg_scheduler, optimizer, num_epochs):
  """
  lr scheduler: linear warmup + cosine decay (per epoch), or none
  """

  if cfg_scheduler.get('name') != 'cosine': return None

  warmup = cfg_scheduler.get('warmup_epochs', 0)
  min_factor = cfg_scheduler.get('min_lr_factor', 0.0)

  def lr_lambda(epoch):
    if epoch < warmup: return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(num_epochs - warmup, 1)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_validation_epoch(model, dataloader_validation):
  """
  validation pass: returns mean loss, accuracy and macro auc
  """

  # targets and raw outputs
  y_targets, y_outputs, losses = [], [], []

  # validation loader
  for data in dataloader_validation:

    # forward (with MSAB context for FiLM students), then loss + metrics
    with torch.no_grad():
      x = data[0].to(device=model.device, dtype=torch.float32)
      y = data[1].to(device=model.device)
      ctx = context_of(model, data)
      y_hat = forward_with_ctx(model, x, ctx)
      loss = model.criterion(y_hat, y).item()
      y_hat = model.prediction_post_processing(y_hat)

    # collect
    y_targets.append(data[1].numpy())
    y_outputs.append(y_hat)
    losses.append(loss)

  # stack
  y_targets = np.concatenate(y_targets)
  y_outputs = np.concatenate(y_outputs)

  # metrics
  acc = float(np.mean(y_targets == np.argmax(y_outputs, axis=-1)))
  try: auc = float(roc_auc_score(y_targets, softmax(y_outputs, axis=1), multi_class='ovr', average='macro'))
  except ValueError: auc = float('nan')

  return float(np.mean(losses)), acc, auc


def run_model_training(cfg, model, dataloader_train, dataloader_validation, label_dict, run_logger=None):
  """
  run model training
  """

  # recipe config with defaults
  recipe = {**RECIPE_DEFAULTS, **cfg.get('training_recipe', {})}
  num_epochs = cfg['model_training']['num_epochs']
  num_classes = len(label_dict)

  # scheduler
  scheduler = make_scheduler(recipe.get('scheduler', {}), model.optimizer, num_epochs)

  # mixup config
  cfg_mixup = recipe.get('mixup', {}) or {}
  mixup_alpha, mixup_p = cfg_mixup.get('alpha', 0.0), cfg_mixup.get('p', 0.0)

  # distillation config
  cfg_distill = recipe.get('distillation', {}) or {}
  distill_on = cfg_distill.get('enabled', False)

  # best checkpoint tracking
  best_metric_name = recipe.get('best_checkpoint_metric', 'val_auc')
  best_metric, best_epoch, best_state = -np.inf, -1, None
  patience = recipe.get('early_stopping_patience', 0)

  # info
  print("\nTrain model on device: {} | epochs: {} | mixup(alpha={}, p={}) | scheduler: {} | distill: {} | best on: {}\n".format(
    model.get_device_full_str(), num_epochs, mixup_alpha, mixup_p, recipe.get('scheduler', {}).get('name'),
    'alpha={} T={}'.format(cfg_distill.get('alpha'), cfg_distill.get('temperature')) if distill_on else 'off', best_metric_name))

  # epochs
  for epoch in range(num_epochs):

    # set to train mode
    model.set_model_to_training_mode()

    # epoch loss
    epoch_train_loss = []

    # train loader
    for data in dataloader_train:

      if distill_on:
        # data: (x, y, sid, teacher_logits[, ctx]) -> joint mixup, then KL+CE
        # (ctx is the last element for FiLM students; mixup is off in the C2
        #  recipe, so x and ctx stay aligned)
        x, y_soft, t_logits = mixup_with_teacher(data[0], data[1], data[3], num_classes=num_classes, alpha=mixup_alpha, p=mixup_p)
        x = x.to(device=model.device, dtype=torch.float32)
        y_soft = y_soft.to(device=model.device)
        t_logits = t_logits.to(device=model.device)
        ctx = context_of(model, data)

        model.optimizer.zero_grad()
        student_logits = forward_with_ctx(model, x, ctx)
        loss_t, _ = distillation_loss(student_logits, t_logits, y_soft,
                                      alpha=cfg_distill.get('alpha', 0.5),
                                      temperature=cfg_distill.get('temperature', 3.0),
                                      label_smoothing=cfg_distill.get('label_smoothing', 0.0))
        loss_t.backward()
        model.optimizer.step()
        loss = float(loss_t.detach())

      else:
        # batch-level mixup -> soft targets (also without mixing, targets become one-hot)
        x, y_soft = mixup_batch(data[0], data[1], num_classes=num_classes, alpha=mixup_alpha, p=mixup_p)

        # training step
        loss = model.train_step((x, y_soft))

      # loss update
      epoch_train_loss.append(loss)

    # scheduler step (per epoch)
    current_lr = model.optimizer.param_groups[0]['lr']
    if scheduler is not None: scheduler.step()

    # evaluation mode
    model.set_model_to_evaluation_mode()

    # validation metrics
    val_loss, val_acc, val_auc = run_validation_epoch(model, dataloader_validation)

    # epoch info
    print("Epoch {:03} - lr: {:.2e}, train loss: {:.4f}, val: [loss: {:.4f}, acc: {:.4f}, auc: {:.4f}]".format(
      epoch + 1, current_lr, np.mean(epoch_train_loss), val_loss, val_acc, val_auc))

    # run logging
    if run_logger is not None: run_logger.log_epoch(epoch + 1, lr=current_lr, train_loss=float(np.mean(epoch_train_loss)), val_loss=val_loss, val_acc=val_acc, val_auc=val_auc)

    # best checkpoint tracking
    epoch_metric = {'val_auc': val_auc, 'val_acc': val_acc, 'val_loss': -val_loss}[best_metric_name]
    if not np.isnan(epoch_metric) and epoch_metric > best_metric:
      best_metric, best_epoch = epoch_metric, epoch + 1
      best_state = copy.deepcopy(model.state_dict())

    # early stopping
    if patience and (epoch + 1 - best_epoch) >= patience:
      print("Early stopping at epoch {} (no {} improvement for {} epochs)".format(epoch + 1, best_metric_name, patience))
      break

  # restore best checkpoint
  if best_state is not None:
    model.load_state_dict(best_state)
    print("Training finished - restored best checkpoint from epoch {} ({}: {:.4f})".format(best_epoch, best_metric_name, best_metric))
  else:
    print("Training finished - no best checkpoint tracked, keeping last weights")

  # run logging
  if run_logger is not None: run_logger.log_metrics(best_epoch=best_epoch, **{'best_{}'.format(best_metric_name): float(best_metric)})

  # remove any stale .tflite from a previous run so its existence is an
  # authoritative signal of whether THIS run exported one (on a node without
  # litert-torch the export is skipped and no .tflite should be present)
  model.get_tflite_model_file_path().unlink(missing_ok=True)

  # save model
  model.save(save_also_as_tflite=True)

  # save also label dict
  yaml.dump({'label_dict': label_dict}, open(Path(model.get_save_path()) / 'label_dict.yaml', 'w'))


def run_model_testing(cfg, model, dataloader_test, label_dict, run_logger=None):
  """
  run model testing (best checkpoint, float pytorch model)
  """

  # info
  print("\nTest model on device: {}...\n".format(model.get_device_full_str()))

  # targets and raw outputs
  y_targets, y_outputs = [], []

  # test loader
  model.set_model_to_evaluation_mode()
  for data in dataloader_test:

    # prediction (with MSAB context for FiLM students)
    with torch.no_grad():
      x = data[0].to(device=model.device, dtype=torch.float32)
      ctx = context_of(model, data)
      y_hat = model.prediction_post_processing(forward_with_ctx(model, x, ctx))

    # collect
    y_targets.append(data[1].numpy())
    y_outputs.append(y_hat)

  # stack
  y_targets = np.concatenate(y_targets)
  y_outputs = np.concatenate(y_outputs)
  y_predictions = np.argmax(y_outputs, axis=-1)

  # metrics
  acc = float(np.mean(y_targets == y_predictions))
  try: auc = float(roc_auc_score(y_targets, softmax(y_outputs, axis=1), multi_class='ovr', average='macro'))
  except ValueError: auc = float('nan')

  # report path
  plot_path_cm = CM_FIG_PATH.parent / 'cm_{}.png'.format(model.get_model_name())

  # confusion matrix
  plot_confusion_matrix(y_targets.tolist(), y_predictions.tolist(), labels=list(label_dict.keys()), plot_path=plot_path_cm)

  # test info
  print("Test accuracy: {:.4f}, macro auc: {:.4f}".format(acc, auc))
  print("Confusion matrix is saved in [{}]".format(plot_path_cm))

  # run logging
  if run_logger is not None:
    run_logger.log_metrics(val_acc=acc, val_auc=auc)
    run_logger.log_artifact_size('pth', model.get_model_file_path())

  # info
  print("Testing of model finished!\n")

  return {'acc': acc, 'auc': auc}


def pytorch_model_taining(cfg_framework, datamodule_train, datamodule_validation, datamodule_test, run_logger=None):
  """
  pytorch model training
  """

  # recipe config with defaults
  recipe = {**RECIPE_DEFAULTS, **cfg_framework.get('training_recipe', {})}
  num_classes = len(datamodule_train.get_label_dict())
  input_shape = datamodule_train.get_feature_shape_at_load()

  # build the model first (so we can tell if it needs MSAB context for FiLM)
  model_class = getattr(importlib.import_module(cfg_framework['model']['module']), cfg_framework['model']['attr'])
  model_kwargs_overwrite = {'input_shape': input_shape, 'num_classes': num_classes, 'save_path': str(MODELS_DIR)}
  model = model_class(*cfg_framework['model']['args'], **{**cfg_framework['model']['kwargs'], **model_kwargs_overwrite})
  needs_context = getattr(model, 'needs_context', False)

  # MSAB context (track C2 FiLM students): load per-split stem -> ctx
  cfg_context = recipe.get('context', {}) or {}
  ctx_train = ctx_eval = None
  if needs_context:
    msab_dir = cfg_context['msab_dir']
    ctx_train = load_msab(msab_dir, cfg_context.get('train_split', 'Train'))
    ctx_eval = load_msab(msab_dir, cfg_context.get('eval_split', 'Validation'))
    print('FiLM context enabled - MSAB from: {}'.format(msab_dir))

  # train dataset with dynamic augmentation (train split only)
  dataset_train = FeatureAugmentDataset(DataloaderPytorch(datamodule_train), cfg=recipe.get('augmentation', {}))

  # attach teacher logits for distillation (track C1) + optional MSAB ctx (C2)
  cfg_distill = recipe.get('distillation', {}) or {}
  if cfg_distill.get('enabled', False):
    stem_to_logits = load_teacher_logits(cfg_distill['teacher_dir'], cfg_distill.get('train_split', 'Train'))
    dataset_train = TeacherLogitDataset(dataset_train, datamodule_train, stem_to_logits, num_classes, stem_to_ctx=ctx_train)
    print('Distillation enabled - teacher logits from: {} ({} train samples aligned)'.format(cfg_distill['teacher_dir'], len(dataset_train)))
  elif needs_context:
    dataset_train = ContextDataset(dataset_train, datamodule_train, ctx_train)

  # validation / test datasets (attach MSAB ctx for FiLM students)
  dataset_val = DataloaderPytorch(datamodule_validation)
  dataset_test = DataloaderPytorch(datamodule_test)
  if needs_context:
    dataset_val = ContextDataset(dataset_val, datamodule_validation, ctx_eval)
    dataset_test = ContextDataset(dataset_test, datamodule_test, ctx_eval)

  # dataloaders
  dataloader_train = torch.utils.data.DataLoader(dataset_train, **cfg_framework['dataloader_train_kwargs'])
  dataloader_validation = torch.utils.data.DataLoader(dataset_val, **cfg_framework['dataloader_validation_and_test_kwargs'])
  dataloader_test = torch.utils.data.DataLoader(dataset_test, **cfg_framework['dataloader_validation_and_test_kwargs'])

  # summary (single-input torchsummary doesn't support FiLM's 2-input forward)
  if not needs_context:
    summary(model, input_size=input_shape, device=model.get_device_type_str())

  # run logging
  if run_logger is not None:
    run_logger.log_data_info(
      n_train=len(datamodule_train), n_validation=len(datamodule_validation), n_test=len(datamodule_test),
      feature_shape=list(input_shape), num_classes=len(datamodule_train.get_label_dict()))
    run_logger.log_model_info(model_name=model.get_model_name(), params=int(model.count_params()), macs=int(model.count_operations()))

  # run model training
  run_model_training(cfg_framework, model, dataloader_train, dataloader_validation, label_dict=datamodule_train.get_label_dict(), run_logger=run_logger)

  # run model testing
  run_model_testing(cfg_framework, model, dataloader_test, label_dict=datamodule_test.get_label_dict(), run_logger=run_logger)

  return model
