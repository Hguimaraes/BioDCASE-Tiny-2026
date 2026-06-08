"""Datamodule variant that caches 2D modulation-spectrum features."""

from __future__ import annotations

import numpy as np

from datamodule import DatamoduleTinyMl
from experiments.features.modulation_spectrum import ModulationSpectrum2DFeatureHandler


class DatamoduleMSS2D(DatamoduleTinyMl):
  """TinyML datamodule using BioME-style 2D modulation features."""

  def at_caching_add_something_before_file_processing(self):
    self.cache_info.update({"x_len": None, "fs": None, "feature_size_origin": None})
    self.feature_handler = ModulationSpectrum2DFeatureHandler(
      **{**self.cfg["feature_extraction"], **self.cfg["feature_handler_add_kwargs"]}
    )

  def at_caching_add_something_after_file_processing(self):
    self.cache_info.update({
      "feature_extraction": self.cfg["feature_extraction"],
      "sample_rate": self.cfg["target_sample_rate"],
      "feature_type": "modulation_spectrum_2d",
      "feature_note": "BioME-style modulation PSD map without axis averaging",
    })

  def at_caching_extract_features_from_file(self, file):
    data = np.load(str(file))
    x, fs = data["x"], data["fs"]
    features = self.feature_handler.extract(x)

    if self.cache_info["x_len"] is None:
      self.cache_info["x_len"] = len(x)
    if self.cache_info["fs"] is None:
      self.cache_info["fs"] = int(fs)
    if self.cache_info["feature_size_origin"] is None:
      self.cache_info["feature_size_origin"] = list(features.shape)

    assert self.cache_info["x_len"] == len(x)
    assert self.cache_info["fs"] == fs
    assert self.cache_info["feature_size_origin"] == list(features.shape)
    return features.flatten()
