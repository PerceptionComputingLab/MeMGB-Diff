import os
import numpy as np
from PIL import Image


def img_pad(pil_file):
    w, h = pil_file.size
    fixed_size = 200  # 输出正方形图片的尺寸

    if h >= w:
        factor = h / float(fixed_size)
        new_w = int(w / factor)
        if new_w % 2 != 0:
            new_w -= 1
        pil_file = pil_file.resize((new_w, fixed_size))
        pad_w = int((fixed_size - new_w) / 2)
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((0, 0), (pad_w, pad_w)), 'constant')
    else:
        factor = w / float(fixed_size)
        new_h = int(h / factor)
        if new_h % 2 != 0:
            new_h -= 1
        pil_file = pil_file.resize((fixed_size, new_h))
        pad_h = int((fixed_size - new_h) / 2)
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((pad_h, pad_h), (0, 0)), 'constant')

    output_file = Image.fromarray(array_file)
    return output_file


if __name__ == "__main__":
    dir_image = '..\data'  # 图片所在文件夹
    dir_output = '..\data_pad'  # 输出结果文件夹
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    i = 0
    list_image = os.listdir(dir_image)
    for file in list_image:
        path_image = os.path.join(dir_image, file)
        path_output = os.path.join(dir_output, file)
        pil_image = Image.open(path_image).convert('L')
        output_image = img_pad(pil_image)
        output_image.save(path_output)
        i += 1
        print('The num of processed images:', i)
        print(output_image.size)