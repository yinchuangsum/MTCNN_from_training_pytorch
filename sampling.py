from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch


class FaceDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, "positive.txt")).read().splitlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).read().splitlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).read().splitlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        cond = torch.Tensor([int(strs[5])])
        offset = torch.Tensor([float(strs[1]),float(strs[2]),float(strs[3]),float(strs[4])])
        alli = torch.Tensor([float(strs[6]),float(strs[7]),float(strs[8]),float(strs[9]),float(strs[10])
                                ,float(strs[11]),float(strs[12]),float(strs[13]),float(strs[14]),float(strs[15])])
        img_path = os.path.join(self.path,strs[0])
        img_data = transforms.Compose([transforms.ToTensor()])(np.array(Image.open(img_path))) #normalized and CHW



        return img_data,cond,offset,alli
