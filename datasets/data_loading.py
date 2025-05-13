import logging
import os.path
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import scipy.io as scio


class mydataset(Dataset):
    def __init__(self, mul:int, images_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mul=mul


        self.ids = [splitext(file)[0] for file in listdir(images_dir) if file.endswith('.npy')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)




    # 返回函数的静态方法,该方法不强制要求传递参数,可以不实例化
    # 经过归一化和resize后的图片数组
    @staticmethod
    def preprocess(mul,pil_img, scale):
        # w, h = pil_img.size
        #
        # newW, newH = int(scale * w), int(scale * h)
        # # newW=64
        # # newH=64
        # new=max(newW,newH)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # # NEAREST是最邻近插值，BICUBIC是双三次插值
        # pil_img=pil_img.crop(((w-scale)/2,(h-scale)/2,(w-scale)/2+scale,(h-scale)/2+scale))
        # pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)

        # pil_img = img_pad(pil_img,new)

        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 3:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((3, 0, 1,2))

        img_ndarray= (img_ndarray) / (np.max(img_ndarray))




        return img_ndarray

    # 加载指定的文件
    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            # return Image.fromarray(np.load(filename))
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename).convert('L')


    # 返回特定索引下的图片和蒙版对
    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        # .glob函数返回复合要求的文件名的列表
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = self.load(img_file[0])
        img = np.ones([700,700,700])


        img = self.preprocess(self.mul,img, self.scale)
        # x=img.copy()
        # import torchvision.transforms as transforms
        # transf = transforms.ToTensor()
        # x = transf(x)


        return torch.as_tensor(img.copy()).float().contiguous()



