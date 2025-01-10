import os, sys
sys.path.append('vdvae')
import torch
import numpy as np
from image_utils import *
from model_utils import *
from PIL import Image
import torchvision.transforms as T


# Main Class    
class VDVAE():
    def __init__(self,
                 device="cuda",
                 cache_dir="../cache",
                 ):
        H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': f'{cache_dir}/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
        class dotdict(dict):
            """dot.notation access to dictionary attributes"""
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__
        H = dotdict(H)

        self.device = device
        H, self.preprocess_fn = set_up_data(H, device=self.device)
        self.ema_vae = load_vaes(H, device=self.device)
        
    def sample_from_hier_latents(self, latents):
        layers_num=len(latents)
        sample_latents = []
        for i in range(layers_num):
            sample_latents.append(latents[i].clone().detach().float().to(self.device))
        return sample_latents

    # Transfor latents from flattened representation to hierarchical
    def latent_transformation(self, latents):
        shapes=torch.load("vdvae/vdvae_shapes.pt")
        layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14])
        transformed_latents = []
        for i in range(31):
            t_lat = latents[:,layer_dims[:i].sum():layer_dims[:i+1].sum()]
            #std_norm_test_latent = (t_lat - np.mean(t_lat,axis=0)) / np.std(t_lat,axis=0)
            #renorm_test_latent = std_norm_test_latent * np.std(kamitani_latents[i][num_test:].reshape(num_train,-1),axis=0) + np.mean(kamitani_latents[i][num_test:].reshape(num_train,-1),axis=0)
            c,h,w=shapes[i]
            transformed_latents.append(t_lat.reshape(len(latents),c,h,w))
        return transformed_latents
    
    def embed_latent(self, image):
        img = T.functional.resize(image,(64,64))
        img = torch.tensor(np.array(img)).float()[None,:,:,:]
        
        latents = []
        data_input, _ = self.preprocess_fn(img)
        

        if(len(data_input.shape)>4):
            data_input = data_input[0]

        if(data_input.shape[-1] != 3):
            data_input = data_input.permute(0,2,3,1)

        # print(data_input.shape,'hell')
        with torch.no_grad():
            activations = self.ema_vae.encoder.forward(data_input)
            _, stats = self.ema_vae.decoder.forward(activations, get_latents=True)
            #recons = ema_vae.decoder.out_net.sample(px_z)
            batch_latent = []
            for j in range(31):
                batch_latent.append(stats[j]['z'].cpu().numpy().reshape(len(data_input),-1))
            latents.append(np.hstack(batch_latent))
        latents = np.concatenate(latents)
        latents = torch.from_numpy(latents)
        return latents

    def reconstruct(self, latents):
        input_latent = self.latent_transformation(latents)
        samp = self.sample_from_hier_latents(input_latent)
        px_z = self.ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
        sample_from_latent = self.ema_vae.decoder.out_net.sample(px_z)
        im = sample_from_latent
        im = Image.fromarray(im[0])
        im = im.resize((768,768),resample=Image.Resampling.LANCZOS)
        return im