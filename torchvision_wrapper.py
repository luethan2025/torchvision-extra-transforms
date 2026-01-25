import importlib
import os
import sys

import torchvision

def wrap_torchvision():
  torchvision_transforms_extra_path = os.path.join("torchvision", "transforms", "extra")
  spec = importlib.util.spec_from_file_location(
    ".".join(["torchvision", "transforms", "extra"]),
    os.path.join(torchvision_transforms_extra_path, "__init__.py")
  )
  torchvision_transforms_extra_module = importlib.util.module_from_spec(spec)
  sys.modules["torchvision.transforms.extra"] = torchvision_transforms_extra_module
  spec.loader.exec_module(torchvision_transforms_extra_module)

  torchvision_datasets_extra_path = os.path.join("torchvision", "datasets", "extra")
  spec = importlib.util.spec_from_file_location(
    ".".join(["torchvision", "datasets", "extra"]),
    os.path.join(torchvision_datasets_extra_path, "__init__.py")
  )
  torchvision_datasets_extra_module = importlib.util.module_from_spec(spec)
  sys.modules["torchvision.datasets.extra"] = torchvision_datasets_extra_module
  spec.loader.exec_module(torchvision_datasets_extra_module)

wrap_torchvision()
