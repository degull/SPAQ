import numpy as np
from torchvision import transforms
from PIL import Image
import torch

class Image_load(object):
    def __init__(self, size, stride, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.stride = stride
        self.interpolation = interpolation

    def __call__(self, img):
        image = self.adaptive_resize(img)
        return self.generate_patches(image, input_size=self.stride)

    def adaptive_resize(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size
        if h < self.size or w < self.size:
            img = transforms.ToTensor()(img)
            print('if img.size=', img.size(), type(img))
            return img
        else:
            img = transforms.ToTensor()(transforms.Resize(self.size, self.interpolation)(img))
            print('else img.size=', img.size(), type(img))
            return img
        
    def to_numpy(self, image):
        # PIL 이미지를 NumPy 배열로 변환
        p = np.array(image)  # 수정된 부분: image.numpy() -> np.array(image)
        if len(p.shape) == 2:  # 만약 grayscale 이미지라면, 채널 차원 추가
            p = np.expand_dims(p, axis=-1)
        return p.transpose((1, 2, 0))  # (H, W, C)로 차원 변환

    def generate_patches(self, image, input_size, type=np.float32):
        img = self.to_numpy(image)
        img_shape = img.shape
        img = img.astype(dtype=type)
        if len(img_shape) == 2:
            H, W = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray([img, ] * 3, dtype=img.dtype)

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        
        patches_numpy = [img[hId:hId + input_size, wId:wId + input_size, :]
                         for hId in hIdx
                         for wId in wIdx]
        
        patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        return patches_tensor.squeeze(0)
