from abc import abstractmethod

import mindspore_lite as mslite
import numpy as np
from tqdm import tqdm

from .model_base import ModelBase


class VCLite(ModelBase):
    def __init__(
        self,
        data_prepare,
        scheduler_preprocess,
        predict_noise,
        noisy_sample,
        vae_decoder,
        device_target="ascend",
        device_id=0,
        num_inference_steps=50,
        task=None,
    ):
        super(VCLite, self).__init__(device_target, device_id)
        # load sub-graphs
        self.data_prepare = self._init_model(data_prepare)
        self.scheduler_preprocess = self._init_model(scheduler_preprocess)
        self.predict_noise = self._init_model(predict_noise)
        self.noisy_sample = self._init_model(noisy_sample)
        self.vae_decoder = self._init_model(vae_decoder)

        n_infer_steps = mslite.Tensor()
        n_infer_steps.shape = []
        n_infer_steps.dtype = mslite.DataType.INT32
        n_infer_steps.set_data_from_numpy(np.array(num_inference_steps, np.int32))
        self.num_inference_steps = n_infer_steps

        # get input
        self.data_prepare_input = self.data_prepare.get_inputs()
        self.predict_noise_input = self.predict_noise.get_inputs()
        self.noisy_sample_input = self.noisy_sample.get_inputs()
        self.vae_decoder_input = self.vae_decoder.get_inputs()

        self.task = task

    @abstractmethod
    def data_prepare_predict(self, inputs):
        pass

    def __call__(self, inputs):
        '''
        Args:
            inputs: dict of data
        Returns:
            frames: (bs f C H W) = (bs f 3 H W)
        '''
        predict_outputs = self.data_prepare_predict(inputs)
        # 1. DataPrepare graph outputs: text_emb, style_emb, single_image_tr, motion_vectors_tr, noise (latents)
        if self.task=='motion_style_transfer':
            text_emb, style_emb, single_image_tr, motion_vectors_tr, latents = predict_outputs

            # 2. PredictNoise graph inputs: latents, ts, text_emb, style_emb, single_image, motion_vectors, guidance_scale):
            scale = self.predict_noise_input[-1]
            scale.set_data_from_numpy(np.array(inputs["scale"]))
            iterator = tqdm(inputs["timesteps"], desc="DDIM Sampling", total=len(inputs["timesteps"]))
            for i, t in enumerate(iterator):
                # predict the noise residual
                ts = self.predict_noise_input[1]
                ts.set_data_from_numpy(np.array(t).astype(np.int32))
                latents = self.scheduler_preprocess.predict([latents, ts])[0]

                noise_pred = self.predict_noise.predict([latents, ts, text_emb, style_emb, single_image_tr, motion_vectors_tr, scale])[0]
                
                # 3. NoisySample grah inputs: noise_pred, ts, latents, num_inference_steps
                latents = self.noisy_sample.predict([noise_pred, ts, latents, self.num_inference_steps])[0]

        else:
            raise ValueError("data_prepare_predict error")
        # 4. VAEDecoder graph inputs 
        frames = self.vae_decoder.predict([latents])[0]
        frames = frames.get_data_to_numpy()
        return frames


class VCLiteMotionStyleTransfer(VCLite):
    def __init__(
        self,
        data_prepare,
        scheduler_preprocess,
        predict_noise,
        noisy_sample,
        vae_decoder,
        device_target="ascend",
        device_id=0,
        num_inference_steps=50,
    ):
        super(VCLiteMotionStyleTransfer, self).__init__(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            device_target,
            device_id,
            num_inference_steps,
            task='motion_style_transfer',
        )

    def data_prepare_predict(self, inputs):
        # DataPrepare graph inputs:  prompt_data, negative_prompt_data, noise, style_image, single_image, motion_vectors):
        self.data_prepare_input[0].set_data_from_numpy(inputs["prompt_data"])
        self.data_prepare_input[1].set_data_from_numpy(inputs["negative_prompt_data"])
        self.data_prepare_input[2].set_data_from_numpy(inputs["noise"])
        self.data_prepare_input[3].set_data_from_numpy(inputs["style_image"])
        self.data_prepare_input[4].set_data_from_numpy(inputs["single_image"])
        self.data_prepare_input[5].set_data_from_numpy(inputs["motion_vectors"])
        #self.data_prepare_input[6].set_data_from_numpy(inputs["fps"])

        return self.data_prepare.predict(self.data_prepare_input)

