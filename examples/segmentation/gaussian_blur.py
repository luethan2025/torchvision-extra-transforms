import matplotlib.pyplot as plt
import torch
import torchvision.transforms.extra as T
import torchvision.transforms.functional as F
from torchvision.extra import make_grid

from utils import load_pascal_voc2010_segmentation_example

def main():
  torch.manual_seed(0)

  transform = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
  img, mask = load_pascal_voc2010_segmentation_example()

  imgs = []
  masks = []
  for _ in range(4):
    img, mask = transform(img, mask)
    imgs.append(F.to_tensor(img))
    masks.append(mask)
  grid = make_grid(imgs, masks, nrow=4)
  plt.imshow(grid.permute(1, 2, 0))
  plt.axis("off")
  plt.show()

if __name__ == "__main__":
  main()