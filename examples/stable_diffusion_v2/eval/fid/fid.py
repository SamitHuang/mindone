import numpy as np
from PIL import Image
from mindspore import dataset as ds
from mindspore.dataset import vision
from mindspore.ops import adaptive_avg_pool2d
from scipy import linalg
from tqdm import tqdm

from .inception_v3 import inception_v3_fid 

class FrechetInceptionDistance():
    def __init__(self,):
        
        # TODO: set context
        self.model = inception_v3_fid()
        self.model.set_train(False)

    def update(images, real=True)
        '''
        Compute features

        images: ms.Tensor
        '''
    
    def get_activations(self, images):
        '''
        Extract inception v3 features from images
        '''
        pass

    def compute(self, gt_images, gen_images):
        pass
    
    def reset(self,): 
        pass


class ImagePathDatasetGenerator:
    '''
    Simple data loader for image files
    '''
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


if __name__ == '__main__':
    # 

