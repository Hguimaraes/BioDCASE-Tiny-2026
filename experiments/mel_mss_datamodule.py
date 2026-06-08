"""Datamodule variant that caches baseline mel plus MSS side-channel features."""

from __future__ import annotations

import numpy as np

from datamodule import DatamoduleTinyMl
from experiments.features.modulation_spectrum import ModulationSpectrum2DFeatureHandler
from feature_handler import FeatureHandler


class DatamoduleMelMSS(DatamoduleTinyMl):
  """Cache a flat [mel, mss] feature vector for two-branch PyTorch models."""

  def at_caching_add_something_before_file_processing(self):
    self.cache_info.update({
      "x_len": None,
      "fs": None,
      "feature_size_origin": None,
      "mel_shape": None,
      "mss_shape": None,
      "mel_size": None,
      "mss_size": None,
      "split_index": None,
    })
    self.mel_feature_handler = FeatureHandler(
      **{**self.cfg["feature_extraction"], **self.cfg["feature_handler_add_kwargs"]}
    )
    self.mss_feature_handler = ModulationSpectrum2DFeatureHandler(
      **{
        **self.cfg["mss_feature_extraction"],
        **self.cfg.get("mss_feature_handler_add_kwargs", {}),
      }
    )

  def at_caching_add_something_after_file_processing(self):
    self.cache_info.update({
      "feature_type": "mel_plus_modulation_spectrum",
      "feature_extraction": self.cfg["feature_extraction"],
      "mss_feature_extraction": self.cfg["mss_feature_extraction"],
      "sample_rate": self.cfg["target_sample_rate"],
    })

  def at_caching_extract_features_from_file(self, file):
    data = np.load(str(file))
    x, fs = data["x"], data["fs"]

    mel = self.mel_feature_handler.extract(x).astype(np.float32)
    mss = self.mss_feature_handler.extract(x).astype(np.float32)
    mel_flat = mel.flatten()
    mss_flat = mss.flatten()
    features = np.concatenate([mel_flat, mss_flat]).astype(np.float32)

    if self.cache_info["x_len"] is None:
      self.cache_info["x_len"] = len(x)
    if self.cache_info["fs"] is None:
      self.cache_info["fs"] = int(fs)
    if self.cache_info["mel_shape"] is None:
      self.cache_info["mel_shape"] = list(mel.shape)
    if self.cache_info["mss_shape"] is None:
      self.cache_info["mss_shape"] = list(mss.shape)
    if self.cache_info["mel_size"] is None:
      self.cache_info["mel_size"] = int(mel_flat.size)
    if self.cache_info["mss_size"] is None:
      self.cache_info["mss_size"] = int(mss_flat.size)
    if self.cache_info["split_index"] is None:
      self.cache_info["split_index"] = int(mel_flat.size)
    if self.cache_info["feature_size_origin"] is None:
      self.cache_info["feature_size_origin"] = list(features.shape)

    assert self.cache_info["x_len"] == len(x)
    assert self.cache_info["fs"] == fs
    assert self.cache_info["mel_shape"] == list(mel.shape)
    assert self.cache_info["mss_shape"] == list(mss.shape)
    assert self.cache_info["mel_size"] == int(mel_flat.size)
    assert self.cache_info["mss_size"] == int(mss_flat.size)
    assert self.cache_info["split_index"] == int(mel_flat.size)
    assert self.cache_info["feature_size_origin"] == list(features.shape)
    return features
