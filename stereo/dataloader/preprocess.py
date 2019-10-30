#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torchvision.transforms as transforms
import os


imagenet_normalize = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

def rand():
    return torch.rand(1).item()


# ------------------- transform ------------------- #
def augment_color(augment=False):
    normalize = normalize_default()
    if augment:
        t_list = list_transform_color_default() + [normalize] 
    else:
       t_list = [normalize]
    return transforms.Compose(t_list)


def get_transform(augment=True):

    t_list = [transforms.ToTensor()]
    if augment:
        t_list += list_transform_color_default()
    t_list.append(normalize_default())

    return transforms.Compose(t_list)


def list_transform_color_default():

    t_list = [
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, gamma=0.4),
        Lighting(0.4),
        PointNoise(0.3), 
        LinearNoise(0.3), 
        OccNoise(0.3), 
        ColorSwap(channel=3, p=0.3), 
        Grayscale(p=0.3), 
    ]
    order = torch.randperm(len(t_list))
    list = [t_list[i] for i in order]

    return list


# ------------------- normalize ------------------- #
normalize_sample = False
def normalize_default():
    if normalize_sample:
        return Normalize_Sample(32)
    else:
        return transforms.Normalize(**imagenet_normalize)
    
    
class Normalize_Sample(object):
    """Normalize image from the mean and std of samples
    >>> transform = Normalize_Sample(32)
    >>> img = torch.rand(3, 64, 64)
    >>> transform(img).shape
    torch.Size([3, 64, 64])
    """

    def __init__(self, num_sample=32):
        self.num_sample = num_sample

    def __call__(self, img):
        h_stride = max(1, img.size(-2)//self.num_sample)
        w_stride = max(1, img.size(-1)//self.num_sample)
        img_sample = img[..., ::h_stride, ::w_stride].contiguous().view(img.size(0), -1)
        mean = img_sample.mean(dim=-1)
        std = img_sample.std(dim=-1).clamp(0.1)

        return transforms.Normalize(mean, std)(img)


# ----------------- color augment ----------------- #
def grayscale(img):
    assert 3 == img.size(-3)
    gs = img.narrow(-3, 0, 1)*0.299 + \
         img.narrow(-3, 1, 1)*0.587 + \
         img.narrow(-3, 2, 1)*0.114
    return gs


class Grayscale(object):
    """Get Grayscale of image
    >>> transform = Grayscale()
    >>> img = torch.rand(3, 64, 64)
    >>> transform(img).shape
    torch.Size([3, 64, 64])
    """

    def __init__(self, p=0.3):
        
        self.gray = (rand() < p)

    def __call__(self, img):
        if not self.gray:
            return img
        else:
            return grayscale(img).expand_as(img)


class ColorSwap(object):
    """Color augment with brightness, contrast and saturation
    >>> transform = ColorSwap(channel=3, p=0.3)
    >>> img = torch.rand(3, 64, 64)
    >>> transform(img).shape
    torch.Size([3, 64, 64])
    """

    def __init__(self, channel=3, p=0.3):
        self.channel = channel
        self.order = None
        if(rand() <= p):
            self.order = list(torch.randperm(self.channel))
            if(sorted(self.order) == self.order):
                self.order = None


    def __call__(self, img):
        
        if(self.order is None):
            return img
        
        assert img.size(-3) == self.channel
        return img[..., self.order, :, :]


class ColorJitter(object):
    """Color augment with brightness, contrast and saturation
    >>> transform = ColorJitter(0.4, 0.4, 0.4, 0.4)
    >>> img = torch.rand(3, 64, 64)
    >>> transform(img).shape
    torch.Size([3, 64, 64])
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, gamma=0.4):
        
        self.alpha_brightness = rand()*brightness
        self.alpha_contrast = rand()*contrast
        self.alpha_saturation = rand()*saturation
        self.alpha_gamma = 1.0 + (rand()-0.5)*gamma
        self.eps = 1e-2
        
        self.need_gray = False
        self.transforms = []

        if abs(self.alpha_brightness) > self.eps:
            self.transforms.append(self._brightness)
        
        if abs(self.alpha_contrast) > self.eps:
            self.transforms.append(self._contrast)
            self.need_gray = True
        
        if abs(self.alpha_saturation) > self.eps:
            self.transforms.append(self._saturation)
            self.need_gray = True
        
        if abs(self.alpha_gamma - 1) > self.eps:
            self.transforms.append(self._gamma)
        
        self.order = torch.randperm(len(self.transforms))

    def _brightness(self, img):
        target = torch.zeros_like(img)
        k = (rand()*0.2 + 1)
        return img.lerp(target, k*self.alpha_brightness)


    def _contrast(self, img):
        target = self.gray.mean().expand_as(img)
        k = (rand()*0.2 + 0.9)
        return img.lerp(target, k*self.alpha_contrast)


    def _saturation(self, img):
        target = self.gray.expand_as(img)
        k = (rand()*0.2 + 0.9)
        return img.lerp(target, k*self.alpha_saturation)


    def _gamma(self, img):
        '''
        >>> torch.Tensor([-1])**0.8
        tensor([nan])
        '''
        k = (rand()*0.2 + 0.9)
        return img.clamp(0)**(k*self.alpha_gamma)


    def __call__(self, img):
        
        if (1 > len(self.transforms)):
            return img
        
        if(self.need_gray):
            self.gray = grayscale(img)
        
        img = img.clone()
        for i in self.order:
            img = self.transforms[i](img)
        
        return img.clamp(0, 1)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)
    >>> transform = Lighting(0.1)
    >>> img = torch.rand(3, 64, 64)
    >>> transform(img).shape
    torch.Size([3, 64, 64])
    """

    def __init__(self, alphastd=0.1, eigval=imagenet_pca['eigval'], eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
        self.alpha = torch.zeros(3).normal_(0, self.alphastd)
        self.rgb = self.eigvec.mul(self.alpha.view(1, 3).expand(3, 3))\
                   .mul(self.eigval.view(1, 3).expand(3, 3))\
                   .sum(1).view(3, 1, 1)

    def __call__(self, img):
        
        if (1e-2 <= self.alphastd):
            return img
        k = (rand()*0.2 + 0.9)
        img = img + (k*self.rgb).type_as(img).expand_as(img)
        
        return img.clamp(0, 1)


# ---------- PointNoise and LinearNoise ----------- #
class PointNoise(object):
    """Add gaussioan noise and point noise
    >>> transform = PointNoise(0.1)
    >>> img = torch.rand(3, 64, 64)
    >>> transform(img).shape
    torch.Size([3, 64, 64])
    """

    def __init__(self, p=0.3):

        self.flag_add_noise = rand()<p
        self.p1 = rand()*0.1
        self.p2 = rand()*0.02

        self.transforms = [self.gaussion_noise, self.point_noise]
        self.order = torch.randperm(len(self.transforms))


    def gaussion_noise(self, img):

        noise = torch.randn(img.shape).type_as(img)*self.p1
        return noise + img


    def point_noise(self, img):

        row, col = img.shape[-2:]
        count = int(self.p2*row*col)
        x = torch.randint(0, row, [count])
        y = torch.randint(0, col, [count])
        half_n = count//2
        img[:, x[:half_n], y[:half_n]] = 1
        img[:, x[half_n:], y[half_n:]] = 0
        
        return img


    def __call__(self, img):
        
        if(not self.flag_add_noise):
            return img
        
        img = img.clone()
        for i in self.order:
            img = self.transforms[i](img)
        
        return img.clamp(0, 1)


class LinearNoise(object):
    """Add noise with linear shape
    >>> transform = LinearNoise(0.1)
    >>> img = torch.rand(3, 64, 64)
    >>> transform(img).shape
    torch.Size([3, 64, 64])
    """

    def __init__(self, p=0.3):

        self.p = p
        self.flag_last = False


    def _noise_gen(self, img):
        
        h, w = img.shape[-2:]
        px = torch.arange(w).type_as(img).expand_as(img[..., 0, :, :])
        py = torch.arange(h).type_as(img)[:, None].expand_as(img[..., 0, :, :])
        cx, cy = rand()*w, rand()*h
        a = (rand()-0.5)*2
        b = (1 - a*a)**0.5
        dx, dy = px-cx, py-cy
        d1 = torch.abs(a*dx + b*dy)
        d2 = torch.abs(-b*dx + a*dy)
        width_d1 = rand()*15 + 5
        width_d2 = rand()*h/4 + h*3//4
        noise = (1 - (d1/width_d1).clamp(0, 1))*(1 - (d2/width_d2).clamp(0, 1))
        mask = torch.rand(noise.shape).type_as(noise) > noise
        noise[mask] = 0
        
        return noise


    def __call__(self, img):
        
        if(self.flag_last or rand() > self.p):
            self.flag_last = False
            return img
       
        self.flag_last = True
        return (self._noise_gen(img) + img).clamp(0, 1)


class OccNoise(object):
    """Add occlusion noise
    >>> transform = OccNoise(0.1)
    >>> img = torch.rand(3, 64, 64)
    >>> transform(img).shape
    torch.Size([3, 64, 64])
    """

    def __init__(self, p=0.3):

        self.p = p
        self.flag_last = False


    def __call__(self, img):
        
        if(self.flag_last or rand() > self.p):
            self.flag_last = False
            return img

        self.flag_last = True
        img = img.clone()
        h, w = img.shape[-2:]
        sx, sy = int(rand()*(w-16)), int(rand()*(h-16))
        width_x = int(rand()*12) + 4
        width_y = 32 - width_x
        img[..., sy:sy+width_y, sx:sx+width_x] = 0
        
        return img


# -------------- visualize augment ---------------- #
def read_img_tensor(imgpath):

    from PIL import Image
    import torchvision.transforms as transforms
    return transforms.ToTensor()(Image.open(imgpath))[None]
    

def ndirpath(dirpath=None, n=0):
    if(os.path.isdir(str(dirpath))):
        return ndirpath(os.path.dirname(dirpath), n-1) if(n>0) else dirpath
    else:
        return ndirpath(os.path.dirname(__file__), n)


def visualize_transforms():
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    fun_ToImage = lambda x: x.cpu().data.numpy().transpose(1, 2, 0)
    
    pathL = os.path.join(ndirpath(None, 1), 'images/10L.png')
    pathR = os.path.join(ndirpath(None, 1), 'images/10R.png')
    imgL = read_img_tensor(pathL).cuda()
    imgR = read_img_tensor(pathR).cuda()
    filters = [Grayscale(1), ColorSwap(3, 1), ColorJitter(0.4, 0.4, 0.4, 0.4), Lighting(0.5), 
                PointNoise(1), LinearNoise(1), OccNoise(1), ]
    filters += [transforms.Compose(list_transform_color_default()) for i in range(10)]
    for filter in filters:
        imgL_t = filter(imgL[0])[None]
        imgR_t = filter(imgR[0])[None]
        # visualize result
        imgs = torch.cat([imgL, imgR, imgL_t, imgR_t], dim=0)
        imgs = make_grid(imgs, nrow=2, padding=8, normalize=False)
        plt.subplot(111); plt.imshow(fun_ToImage(imgs)); plt.title(filter.__class__)
        plt.show()
    

if __name__ == '__main__':

    import doctest
    doctest.testmod()
    
    visualize_transforms()
    
