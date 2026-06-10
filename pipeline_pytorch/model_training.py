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
from pipeline_pytorch.model_tiny_ml import Baseline


# default training recipe (overridable via config: pytorch_framework.training_recipe)
RECIPE_DEFAULTS = {
  'augmentation': {'enabled': True},
  'mixup': {'alpha': 0.2, 'p': 0.5},
  'scheduler': {'name': 'cosine', 'warmup_epochs': 5, 'min_lr_factor': 0.05},
  'best_checkpoint_metric': 'val_auc',
  'early_stopping_patience': 0,
}


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

    # validation step
    y_hat, loss = model.validation_step(data)

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

  # best checkpoint tracking
  best_metric_name = recipe.get('best_checkpoint_metric', 'val_auc')
  best_metric, best_epoch, best_state = -np.inf, -1, None
  patience = recipe.get('early_stopping_patience', 0)

  # info
  print("\nTrain model on device: {} | epochs: {} | mixup(alpha={}, p={}) | scheduler: {} | best on: {}\n".format(
    model.get_device_full_str(), num_epochs, mixup_alpha, mixup_p, recipe.get('scheduler', {}).get('name'), best_metric_name))

  # epochs
  for epoch in range(num_epochs):

    # set to train mode
    model.set_model_to_training_mode()

    # epoch loss
    epoch_train_loss = []

    # train loader
    for data in dataloader_train:

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
  for data in dataloader_test:

    # prediction
    y_hat = model.predict(data[0])

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

  # train dataset with dynamic augmentation (train split only)
  dataset_train = FeatureAugmentDataset(DataloaderPytorch(datamodule_train), cfg=recipe.get('augmentation', {}))

  # dataloader
  dataloader_train = torch.utils.data.DataLoader(dataset_train, **cfg_framework['dataloader_train_kwargs'])
  dataloader_validation = torch.utils.data.DataLoader(DataloaderPytorch(datamodule_validation), **cfg_framework['dataloader_validation_and_test_kwargs'])
  dataloader_test = torch.utils.data.DataLoader(DataloaderPytorch(datamodule_test), **cfg_framework['dataloader_validation_and_test_kwargs'])

  # model
  input_shape = datamodule_train.get_feature_shape_at_load()

  # model class
  model_class = getattr(importlib.import_module(cfg_framework['model']['module']), cfg_framework['model']['attr'])

  # model kwargs
  model_kwargs_overwrite = {'input_shape': input_shape, 'num_classes': len(datamodule_train.get_label_dict()), 'save_path': str(MODELS_DIR)}

  # model
  model = model_class(*cfg_framework['model']['args'], **{**cfg_framework['model']['kwargs'], **model_kwargs_overwrite})

  # summary
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
