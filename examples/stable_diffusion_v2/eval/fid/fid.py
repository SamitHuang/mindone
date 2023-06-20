import numpy as np
from PIL import Image
from mindspore import dataset as ds
from mindspore.dataset import vision
from mindspore.ops import adaptive_avg_pool2d
from scipy import linalg
from tqdm import tqdm

from inception_v3 import inception_v3_fid

class ImagePathDatasetGenerator:
    '''
    Simple data loader for image files
    '''
    def __init__(self, img_paths, transforms=None):
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path = self.img_paths[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def get_activations(img_paths, model, batch_size=50, dims=2048, num_workers=1):
    '''
    Extract inception v3 features from images

    Args:
        img_list
    '''
    # vision ToTensor will norm the input to [0, 1]
    dataset_generator = ImagePathDatasetGenerator(img_paths, transforms=vision.ToTensor())
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False, num_parallel_workers=num_workers)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

    pred_arr = np.empty((len(img_paths), dims))

    start_idx = 0
    ds_iter = dataset.create_dict_iterator()
    for batch in tqdm(ds_iter):
        # TODO: simplify, just output the feature of last black and squeeze.
        pred = model(batch["data"])[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


class FrechetInceptionDistance():
    def __init__(self, ckpt_path=None):

        # TODO: set context
        self.model = inception_v3_fid(ckpt_path=ckpt_path)
        self.model.set_train(False)

    def calculate_activation_stat(self, act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def compute(self, gen_images, gt_images):
        """
        gen_images: list of generated image file paths
        gt_images: list of GT image file paths
        """
        gen_images = sorted(gen_images)
        gt_images = sorted(gt_images)

        gen_feats = get_activations(gen_images, self.model)
        gen_mu, gen_sigma = self.calculate_activation_stat(gen_feats)

        gt_feats = get_activations(gt_images, self.model)
        gt_mu, gt_sigma = self.calculate_activation_stat(gt_feats)

        fid_value = self.calculate_frechet_distance(gen_mu, gen_sigma, gt_mu, gt_sigma)

        return fid_value

    def update(images, real=True):
        pass

    def reset(self,):
        pass


if __name__ == '__main__':
    #
    gen_imgs = ['/Users/Samit/Data/datasets/ic15/det/test/ch4_test_images/img_1.jpg',
                '/Users/Samit/Data/datasets/ic15/det/test/ch4_test_images/img_2.jpg']
    gt_imgs = ['/Users/Samit/Data/datasets/ic15/det/test/ch4_test_images/img_10.jpg',
               '/Users/Samit/Data/datasets/ic15/det/test/ch4_test_images/img_11.jpg',
               ]

    fid_scorer = FrechetInceptionDistance(ckpt_path='./inception_v3_fid.ckpt')
    score = fid_scorer.compute(gen_imgs, gt_imgs)
    print('ms FID: ', score)


    # torch:
    #from torchmetrics.image.fid import FrechetInceptionDistance
    from PIL import Image
    import os

    real_images = [np.array(Image.open(path).convert("RGB")) for path in gt_imgs]
    fake_images = [np.array(Image.open(path).convert("RGB")) for path in gen_imgs]

    import torch
    import torchmetrics as tm
    from torchvision.transforms import functional as F
    #from torchmetrics.image.fid import FrechetInceptionDistance

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        #return F.center_crop(image, (256, 256))
        return image

    real_images = torch.cat([preprocess_image(image) for image in real_images])
    print(real_images.shape)
    fake_images = torch.cat([preprocess_image(image) for image in fake_images])
    print(fake_images.shape)

    # torch
    fid = tm.image.fid.FrechetInceptionDistance(normalize=True)
    #fid = FrechetInceptionDistance(normalize=True)


    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    print(f"pt FID: {float(fid.compute())}")
