from typing import Optional, Union

import torch
import torchvision.transforms.functional as F
import torchvision.utils

@torch.no_grad()
def make_grid(
  tensor: Union[torch.Tensor, list[torch.Tensor]],
  mask: Optional[Union[torch.Tensor, list[torch.Tensor]]]=None,
  nrow: int=8,
  padding: int=2,
  normalize: bool=False,
  value_range: Optional[tuple[int, int]]=None,
  scale_each: bool=False,
  pad_value: float=0.0,
  alpha: float=0.5,
) -> torch.Tensor:
  if isinstance(tensor, list):
    tensor = [F.to_tensor(t) if not isinstance(t, torch.Tensor) else t for t in tensor]
    tensor = torch.stack(tensor, dim=0)
  if mask is not None:
    if isinstance(mask, list):
      mask = [F.to_tensor(m) if not isinstance(m, torch.Tensor) else m for m in mask]
      mask = torch.stack(mask, dim=0)
    tensor = (1 - alpha) * tensor + alpha * mask
  return torchvision.utils.make_grid(
    tensor=tensor,
    nrow=nrow,
    padding=padding,
    normalize=normalize,
    value_range=value_range,
    scale_each=scale_each,
    pad_value=pad_value,
  )