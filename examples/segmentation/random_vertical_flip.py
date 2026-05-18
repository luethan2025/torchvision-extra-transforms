import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.datasets.extra as datasets
import torchvision.transforms.extra as T
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.extra import make_grid

def main():
  torch.manual_seed(0)

  transform = T.RandomVerticalFlip(p=1.0)
  example_dataset = datasets.VOCSegmentation(
    root="./examples/example_datasets/data", image_set="train", transform=None,
  )
  img, mask = example_dataset[0]

  imgs = []
  masks = []
  for _ in range(4):
    img, mask = transform(img, mask)
    imgs.append(F.to_tensor(img))
    colored_mask = example_dataset.decode_segmentation_image(np.array(mask))
    masks.append(Image.fromarray(colored_mask))
  grid = make_grid(imgs, masks, nrow=4)
  plt.imshow(grid.permute(1, 2, 0))
  plt.axis("off")
  plt.show()

if __name__ == "__main__":
  main()
