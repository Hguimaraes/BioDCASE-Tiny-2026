# --
# model training

import sys
import yaml
import torch
import numpy as np
import importlib

from torchsummary import summary
from pathlib import Path

# add root path of project if called as main
if __name__ == '__main__': [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from plots import plot_confusion_matrix
from pipeline_pytorch.paths import MODELS_DIR, CM_FIG_PATH
from pipeline_pytorch.pytorch_datamodule import DataloaderPytorch
from pipeline_pytorch.model_tiny_ml import Baseline


def early_stopping_improved(value, best_value, mode, min_delta):
  if mode == 'min':
    return value < best_value - min_delta
  return value > best_value + min_delta


def run_model_training(cfg, model, dataloader_train, dataloader_validation, label_dict):
  """
  run model training
  """

  # info
  print("\nTrain model on device: {}...\n".format(model.get_device_full_str()))

  # history
  history = []
  best_accuracy = {
    'epoch': None,
    'validation_accuracy': -float('inf'),
    'validation_loss': None,
    'path': str(model.get_model_file_path().with_name(model.get_model_file_path().stem + '_best_accuracy.pth')),
  }
  best_loss = {
    'epoch': None,
    'validation_accuracy': None,
    'validation_loss': float('inf'),
    'path': str(model.get_model_file_path().with_name(model.get_model_file_path().stem + '_best_loss.pth')),
  }
  early_cfg = cfg['model_training'].get('early_stopping', {})
  early_enabled = early_cfg.get('enabled', False)
  early_monitor = early_cfg.get('monitor', 'validation_accuracy')
  early_mode = early_cfg.get('mode', 'max')
  early_patience = early_cfg.get('patience', 15)
  early_min_delta = early_cfg.get('min_delta', 0.0)
  early_best = float('inf') if early_mode == 'min' else -float('inf')
  early_bad_epochs = 0
  early_summary = {
    'enabled': bool(early_enabled),
    'monitor': early_monitor,
    'mode': early_mode,
    'patience': early_patience,
    'min_delta': early_min_delta,
    'stopped_early': False,
    'stop_epoch': None,
    'best_epoch': None,
    'best_value': None,
  }

  # epochs
  for epoch in range(cfg['model_training']['num_epochs']):

    # set to train mode
    model.set_model_to_training_mode()

    # epoch loss
    epoch_train_loss = []
    epoch_validation_loss = []

    # train loader
    for data in dataloader_train: 

      # trainign step
      loss = model.train_step(data)

      # loss update
      epoch_train_loss.append(loss)

    # evaluation mode
    model.set_model_to_evaluation_mode()

    # targets and predictions
    y_targets = np.empty(shape=0, dtype=np.int8)
    y_predictions = np.empty(shape=0, dtype=np.int8)

    # validation loader
    for data in dataloader_validation: 

      # validation step
      y_pred, loss = model.validation_step(data)

      # argmax for acc
      y_pred = np.argmax(y_pred, axis=-1)

      # append targets and predictions
      y_targets = np.append(y_targets, data[1].numpy().astype(np.int8))
      y_predictions = np.append(y_predictions, y_pred.astype(np.int8))

      # loss update
      epoch_validation_loss.append(loss)

    # epoch metrics
    train_loss = float(np.mean(epoch_train_loss))
    validation_loss = float(np.mean(epoch_validation_loss))
    validation_accuracy = float(np.mean(y_targets == y_predictions))
    history.append({
      'epoch': epoch + 1,
      'train_loss': train_loss,
      'validation_loss': validation_loss,
      'validation_accuracy': validation_accuracy,
    })

    if validation_accuracy > best_accuracy['validation_accuracy']:
      best_accuracy.update({
        'epoch': epoch + 1,
        'validation_accuracy': validation_accuracy,
        'validation_loss': validation_loss,
      })
      torch.save(model.state_dict(), best_accuracy['path'])
      print("  saved best-accuracy checkpoint: {}".format(best_accuracy['path']))

    if validation_loss < best_loss['validation_loss']:
      best_loss.update({
        'epoch': epoch + 1,
        'validation_accuracy': validation_accuracy,
        'validation_loss': validation_loss,
      })
      torch.save(model.state_dict(), best_loss['path'])
      print("  saved best-loss checkpoint: {}".format(best_loss['path']))

    # epoch info
    print("Epoch {:03} - train loss: {:.4f}, val: [loss: {:.4f}, acc: {:.4f}]".format(epoch + 1, train_loss, validation_loss, validation_accuracy))

    if early_enabled:
      monitored_value = history[-1][early_monitor]
      if early_stopping_improved(monitored_value, early_best, early_mode, early_min_delta):
        early_best = monitored_value
        early_bad_epochs = 0
        early_summary.update({
          'best_epoch': epoch + 1,
          'best_value': monitored_value,
        })
      else:
        early_bad_epochs += 1
        print("  early stopping patience: {}/{}".format(early_bad_epochs, early_patience))

      if early_bad_epochs >= early_patience:
        early_summary.update({
          'stopped_early': True,
          'stop_epoch': epoch + 1,
        })
        print("Early stopping at epoch {:03}; best {} was {:.4f} at epoch {:03}.".format(
          epoch + 1,
          early_monitor,
          early_best,
          early_summary['best_epoch'],
        ))
        break

  # info
  print("Training of model finished!")

  # save model in case of no early stopping
  model.save(save_also_as_tflite=True)

  # save also label dict
  yaml.dump({'label_dict': label_dict}, open(Path(model.get_save_path()) / 'label_dict.yaml', 'w'))
  yaml.dump(
    {
      'best_accuracy': best_accuracy,
      'best_loss': best_loss,
      'final_model_path': str(model.get_model_file_path()),
      'early_stopping': early_summary,
    },
    open(Path(model.get_save_path()) / 'checkpoint_summary.yaml', 'w'),
  )

  model.best_checkpoints = {
    'best_accuracy': best_accuracy,
    'best_loss': best_loss,
    'final_model_path': str(model.get_model_file_path()),
    'checkpoint_summary_path': str(Path(model.get_save_path()) / 'checkpoint_summary.yaml'),
    'early_stopping': early_summary,
  }

  return history


def run_model_testing(cfg, model, dataloader_test, label_dict):
  """
  run model training
  """

  # info
  print("\nTest model on device: {}...\n".format(model.get_device_full_str()))

  # predictions and targets
  y_predictions = []
  y_targets = []

  # test loader
  for data in dataloader_test: 

    # validation step
    y_pred = model.predict(data[0])

    # argmax for acc
    y_pred = np.argmax(y_pred, axis=-1)

    # add data
    y_predictions.extend(y_pred.tolist())
    y_targets.extend(data[1].tolist())

  # accuracy
  acc = np.mean(np.array(y_predictions) == np.array(y_targets)).item()

  # report path
  plot_path_cm = CM_FIG_PATH.parent / 'cm_{}.png'.format(model.get_model_name())

  # confusion matrix
  plot_confusion_matrix(y_targets, y_predictions, labels=list(label_dict.keys()), plot_path=plot_path_cm)

  # test info
  print("Test accuracy: {:.4f}".format(acc))
  print("Confusion matrix is saved in [{}]".format(plot_path_cm))

  # info
  print("Testing of model finished!\n")

  return {
    'test_accuracy': float(acc),
    'num_test_samples': int(len(y_targets)),
    'confusion_matrix_path': str(plot_path_cm),
  }


def pytorch_model_taining(cfg_framework, datamodule_train, datamodule_validation, datamodule_test):
  """
  pytorch model training
  """
  
  # dataloader
  dataloader_train = torch.utils.data.DataLoader(DataloaderPytorch(datamodule_train), **cfg_framework['dataloader_train_kwargs'])
  dataloader_validation = torch.utils.data.DataLoader(DataloaderPytorch(datamodule_validation), **cfg_framework['dataloader_validation_and_test_kwargs'])
  dataloader_test = torch.utils.data.DataLoader(DataloaderPytorch(datamodule_test), **cfg_framework['dataloader_validation_and_test_kwargs'])

  # model
  input_shape = datamodule_train.get_feature_shape_at_load()
  
  # model class
  model_class = getattr(importlib.import_module(cfg_framework['model']['module']), cfg_framework['model']['attr'])

  # model kwargs
  model_kwargs_defaults = {'save_path': str(MODELS_DIR)}
  model_kwargs_overwrite = {'input_shape': input_shape, 'num_classes': len(datamodule_train.get_label_dict())}

  # model
  model = model_class(*cfg_framework['model']['args'], **{**model_kwargs_defaults, **cfg_framework['model']['kwargs'], **model_kwargs_overwrite})

  # summary
  summary(model, input_size=input_shape, device=model.get_device_type_str())

  # run model training
  training_history = run_model_training(cfg_framework, model, dataloader_train, dataloader_validation, label_dict=datamodule_train.get_label_dict())

  # optionally evaluate the best checkpoint instead of the final epoch
  evaluate_checkpoint = cfg_framework['model_training'].get('evaluate_checkpoint', 'final')
  if evaluate_checkpoint != 'final':
    checkpoint_info = model.best_checkpoints.get(evaluate_checkpoint)
    if checkpoint_info and Path(checkpoint_info['path']).is_file():
      print("Load {} checkpoint for testing: {}".format(evaluate_checkpoint, checkpoint_info['path']))
      model.load(checkpoint_info['path'])
      model.evaluated_checkpoint = {
        'name': evaluate_checkpoint,
        **checkpoint_info,
      }
    else:
      print("***Requested checkpoint [{}] was not found; testing final model.".format(evaluate_checkpoint))
      model.evaluated_checkpoint = {'name': 'final', 'path': str(model.get_model_file_path())}
  else:
    model.evaluated_checkpoint = {'name': 'final', 'path': str(model.get_model_file_path())}

  # run model testing
  test_metrics = run_model_testing(cfg_framework, model, dataloader_test, label_dict=datamodule_test.get_label_dict())
  test_metrics['evaluated_checkpoint'] = model.evaluated_checkpoint

  # attach metrics without changing the existing return contract
  model.training_history = training_history
  model.test_metrics = test_metrics

  return model
