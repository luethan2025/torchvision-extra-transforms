## torchvision-extra-transforms

### Installation

```console
echo "import sys, importlib.util; \
  import torchvision, torchvision.transforms, torchvision.datasets; \
  torchvision.__path__.append('$(pwd)/torchvision'); \
  torchvision.transforms.__path__.append('$(pwd)/torchvision/transforms'); \
  torchvision.datasets.__path__.append('$(pwd)/torchvision/datasets'); \
  spec = importlib.util.spec_from_file_location('torchvision.extra', '$(pwd)/torchvision/extra/__init__.py'); \
  m = importlib.util.module_from_spec(spec); \
  sys.modules['torchvision.extra'] = m; \
  spec.loader.exec_module(m)" \
  > $(python -c "import site; print(site.getsitepackages()[0])")/torchvision-extra.pth
```
