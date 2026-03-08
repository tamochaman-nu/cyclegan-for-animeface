import os
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    return images[:min(max_dataset_size, len(images))]

def get_transform(opt, grayscale=False):
    transform_list = []
    
    # Handle Grayscale conversions
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
        
    # Preprocess (Resize/Crop)
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, Image.BICUBIC)))

    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
        
    # Handle specific Test-Time preprocessing
    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=Image.BICUBIC)))

    if not opt.preprocess == 'none' and opt.phase == 'train':
        transform_list.append(transforms.RandomHorizontalFlip())

    # ToTensor + Normalize
    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_size, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_size):
        return img
    w = target_size
    h = int(target_size * oh / ow)
    return img.resize((w, h), method)

class UnalignedDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    """
    def __init__(self, opt):
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        # A returns sequentially, B returns randomly (in train mode to unsync)
        A_path = self.A_paths[index % self.A_size]
        B_index = random.randint(0, self.B_size - 1) if not self.opt.serial_batches else index % self.B_size
        B_path = self.B_paths[B_index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
