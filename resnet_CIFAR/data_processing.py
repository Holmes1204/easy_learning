from PIL import Image
import torch
from torch.utils import data

class DogCat(data.Dataset):
    def __init__(self) -> None:
        super().__init__()