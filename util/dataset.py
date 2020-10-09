import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, rain_data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            rain_image_name = os.path.join(rain_data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            rain_image_name = os.path.join(rain_data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, rain_image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, rain_data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, rain_data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, rain_image_path, label_path = self.data_list[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('uint8')  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        rain = cv2.imread(rain_image_path, cv2.IMREAD_COLOR)
        if rain is None:
            print(rain_image_path)
        rain_image = cv2.imread(rain_image_path, cv2.IMREAD_COLOR).astype('uint8')  # BGR 3 channel ndarray wiht shape H * W * 3
        rain_image = cv2.cvtColor(rain_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        rain_image = np.float32(rain_image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        if image.shape[0] != rain_image.shape[0] or image.shape[1] != rain_image.shape[1]:
            raise (RuntimeError("Image & Rain Image shape mismatch: " + image_path + " " + label_path + "\n"))

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, rain_image, label = self.transform(image, rain_image, label)
        return image, rain_image, label
