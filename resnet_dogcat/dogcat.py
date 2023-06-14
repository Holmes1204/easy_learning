import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from numpy import array
import os

transform_pic = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transforms_PIL = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224)])

LABEL = {0:'cats',1:'dogs'}
class DogCat(Dataset):
    def __init__(self,img_dir,transform=None,target_transform=None) -> None:
        self.img_dir = img_dir
        cat_img = os.listdir(self.img_dir+LABEL[0])
        cat_imgs = [os.path.join(self.img_dir+LABEL[0],img) for img in cat_img]
        cat_labels = [0 for img in cat_img]
        dog_img = os.listdir(self.img_dir+LABEL[1])
        dog_imgs = [os.path.join(self.img_dir+LABEL[1],img) for img in dog_img]
        dog_labels = [1 for img in dog_img]
        self.imgs = cat_imgs +dog_imgs 
        self.img_labels = cat_labels + dog_labels
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = Image.open(img_path)
        label = self.img_labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

training_data =DogCat(img_dir='./data/catdog_dataset/training_set/',transform=transform_pic)
test_data = DogCat(img_dir='./data/catdog_dataset/test_set/',transform=transform_pic)

def test_pic(path,transform=transform_pic,transfrom_PIL=transforms_PIL):
    image = Image.open(path)
    image_raw = transfrom_PIL(image)
    image_p = transform(image)
    return image_raw,image_p


if __name__ == "__main__":
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)    
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0]
    label = train_labels[0]
    img_PIL = F.to_pil_image(img)
    plt.axis("off")
    plt.imshow(array(img_PIL))
    plt.show()
    print(f"Label: {label}  "+LABEL[int(label)])

    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(LABEL[int(label)])
    #     plt.imshow(array(F.to_pil_image(img)))
    #     plt.axis("off")
    # plt.show()