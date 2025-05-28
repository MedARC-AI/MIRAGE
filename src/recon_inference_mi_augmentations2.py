#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import json
import pickle
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds
import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# sys.path.append('generative_models/')
# import sgm
from sc_reconstructor import SC_Reconstructor
from vdvae import VDVAE
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
# from models import *
device = "cuda"
print("device:",device)


# In[ ]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # model_name = "final_subj01_pretrained_40sess_24bs"
    model_name = "subj01_40sess_hypatia_ridge_sc2"
    print("model_name:", model_name)

    # other variables can be specified in the following string:
    jupyter_args = f"--data_path=../dataset \
                    --cache_dir=../cache \
                    --model_name={model_name} --subj=1 \
                    --mode vision \
                    --dual_guidance"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--mode",type=str,default="vision",choices=["vision","imagery","shared1000"],
)
parser.add_argument(
    "--gen_rep",type=int,default=10,
)
parser.add_argument(
    "--dual_guidance",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--normalize_preds",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--save_raw",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--raw_path",type=str,
)
parser.add_argument(
    "--strength",type=float,default=0.70,
)
parser.add_argument(
    "--textstrength",type=float,default=0.5,
)
parser.add_argument(
    "--filter_contrast",action=argparse.BooleanOptionalAction, default=True,
    help="Filter the low level output to be more intense and smoothed",
)
parser.add_argument(
    "--filter_sharpness",action=argparse.BooleanOptionalAction, default=True,
    help="Filter the low level output to be more intense and smoothed",
)
parser.add_argument(
    "--num_images_per_sample",type=int, default=16,
    help="Number of images to generate and select between for final recon",
)
parser.add_argument(
    "--retrieval",action=argparse.BooleanOptionalAction,default=True,
    help="Use the decoded captions for dual guidance",
)
parser.add_argument(
    "--prompt_recon",action=argparse.BooleanOptionalAction, default=True,
    help="Use for prompt generation",
)
parser.add_argument(
    "--caption_type",type=str,default='medium',choices=['coco','short', 'medium', 'schmedium'],
)
parser.add_argument(
    "--compile_models",action=argparse.BooleanOptionalAction, default=True,
    help="Use for speeding up stable cascade",
)
parser.add_argument(
    "--num_trial_reps",type=int, default=16,
    help="Number of trial repetitions to average test betas across",
)
parser.add_argument(
    "--gt",action=argparse.BooleanOptionalAction, default=False,
    help="enable ground truth clip",
)
parser.add_argument(
    "--gtc",action=argparse.BooleanOptionalAction, default=False,
    help="enable ground truth clip",
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()
print(f"args: {args}")
# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)


if seed > 0 and gen_rep == 1:
    # seed all random functions, but only if doing 1 rep
    utils.seed_everything(seed)
    

outdir = os.path.abspath(f'../train_logs/{model_name}')

# make output directory
os.makedirs("evals",exist_ok=True)
os.makedirs(f"evals/{model_name}",exist_ok=True)


# # Load data

# In[ ]:


if mode == "synthetic":
    voxels, all_images = utils.load_nsd_synthetic(subject=subj, average=False, nest=True)
elif subj > 8:
    _, _, voxels, all_images = utils.load_imageryrf(subject=subj-8, mode=mode, stimtype="object", average=False, nest=True, split=True)
    
elif mode == "shared1000":
    x_train, valid_nsd_ids_train, x_test, test_nsd_ids = utils.load_nsd(subject=subj, data_path=data_path)
    voxels = torch.mean(x_test, dim=1, keepdim=True)
    print(f"Loaded subj {subj} test betas! {voxels.shape}")
    f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
    images = f['images']

    all_images = torch.zeros((len(test_nsd_ids), 3, 224, 224))
    for i, idx in enumerate(test_nsd_ids):
        all_images[i] =  torch.from_numpy(images[idx])
    del images, f
    print(f"Filtered down to only the {len(test_nsd_ids)} test images for subject {subj}!")
else:
    voxels, all_images = utils.load_nsd_mental_imagery(subject=subj, 
                                                       mode=mode, 
                                                       stimtype="all", 
                                                       average=True, 
                                                       nest=False,
                                                       num_reps=num_trial_reps)
print(voxels.shape)


# # Load pretrained models

# # Load Stable Cascade

# In[ ]:


reconstructor = SC_Reconstructor(compile_models=compile_models, device=device)
if blurry_recon:
    vdvae = VDVAE(device=device, cache_dir=cache_dir)

image_embedding_variant = "stable_cascade"
clip_emb_dim = 768
clip_seq_dim = 1

retrieval_embedding_variant = "stable_cascade_hidden"
retrieval_emb_dim = 1024
retrieval_seq_dim = 257

text_embedding_variant = "stable_cascade"
clip_text_seq_dim=77
clip_text_emb_dim=1280

latent_embedding_variant = "vdvae"
latent_emb_dim = 91168

prompt_embedding_variant = "git"
git_seq_dim = 257
git_emb_dim = 1024

if caption_type != "coco":
    text_embedding_variant += f"_{caption_type}"


# ### Compute ground truth embeddings for training data (for feature normalization)

# In[ ]:


# If this is erroring, feature extraction failed in Train.ipynb
if normalize_preds:
    file_path = f"{data_path}/preprocessed_data/subject{subj}/{image_embedding_variant}_image_embeddings_train.pt"
    clip_image_train = torch.load(file_path)
        
    if dual_guidance:
        file_path_txt = f"{data_path}/preprocessed_data/subject{subj}/{text_embedding_variant}_text_embeddings_train.pt"
        clip_text_train = torch.load(file_path_txt)
        
    if blurry_recon:
        file_path = f"{data_path}/preprocessed_data/subject{subj}/{latent_embedding_variant}_latent_embeddings_train.pt"
        vae_image_train = torch.load(file_path)
    else:
        strength = 1.0
        
    if prompt_recon:
        file_path_prompt = f"{data_path}/preprocessed_data/subject{subj}/{prompt_embedding_variant}_prompt_embeddings_train.pt"
        git_text_train = torch.load(file_path_prompt) 
           
    if retrieval:
        file_path = f"{data_path}/preprocessed_data/subject{subj}/{retrieval_embedding_variant}_retrieval_embeddings_train.pt"
        retrieval_image_train = torch.load(file_path)
    else:
        num_images_per_sample = 1


# # Predicting latent vectors for reconstruction  

# In[ ]:


if gt:
    file_path = f"{data_path}/preprocessed_data/gt/{image_embedding_variant}_image_embeddings.pt"
    os.makedirs(f"{data_path}/preprocessed_data/gt", exist_ok=True)
    emb_batch_size = 1
    if not os.path.exists(file_path):
        # Generate CLIP Image embeddings
        print("Generating Image embeddings!")
        pred_clip_image = torch.zeros((len(all_images), clip_seq_dim, clip_emb_dim)).to("cpu")
        for i in tqdm(range(len(all_images) // emb_batch_size), desc="Encoding clip images..."):
            batch_list = []
            for img in all_images[i * emb_batch_size:i * emb_batch_size + emb_batch_size]:
                batch_list.append(transforms.ToPILImage()(img))
            pred_clip_image[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = reconstructor.embed_image(batch_list).to("cpu")

        torch.save(pred_clip_image, file_path)
    else:
        pred_clip_image = torch.load(file_path)

            
    if dual_guidance:
        emb_batch_size = 1
        gt_captions = np.load(f"{data_path}/preprocessed_data/captions_18.npy")
        file_path_txt = f"{data_path}/preprocessed_data/gt/{text_embedding_variant}_text_embeddings.pt"
        os.makedirs(f"{data_path}/preprocessed_data/gt", exist_ok=True)
        if not os.path.exists(file_path_txt):
            # Generate CLIP Text embeddings
            print("Generating Text embeddings!")
            pred_clip_text = torch.zeros((len(gt_captions), clip_text_seq_dim, clip_text_emb_dim)).to("cpu")
            for i in tqdm(range(len(gt_captions) // emb_batch_size), desc="Encoding captions..."):
                batch_captions = gt_captions[i * emb_batch_size:i * emb_batch_size + emb_batch_size].tolist()
                pred_clip_text[i * emb_batch_size:i * emb_batch_size + emb_batch_size] =  reconstructor.embed_text(batch_captions).to("cpu")
            torch.save(pred_clip_text, file_path_txt)
        else:
            pred_clip_text = torch.load(file_path_txt)


    if blurry_recon:
        emb_batch_size = 1
        file_path = f"{data_path}/preprocessed_data/gt/{latent_embedding_variant}_latent_embeddings.pt"
        os.makedirs(f"{data_path}/preprocessed_data/gt", exist_ok=True)
        if not os.path.exists(file_path):
            print("Generating Latent Image embeddings!")
            pred_blurry_vae = torch.zeros((len(all_images), latent_emb_dim)).to("cpu")
            for i in tqdm(range(len(all_images)), desc="Encoding blurry images..."):
                img = transforms.ToPILImage()(all_images[i])
                pred_blurry_vae[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = vdvae.embed_latent(img).reshape(-1, latent_emb_dim).to("cpu")
            torch.save(pred_blurry_vae, file_path)
        else:
            pred_blurry_vae = torch.load(file_path)
        
    print(f"Loaded vectors for subj{subj}!")
else:
    pred_clip_image = torch.zeros((len(all_images), clip_seq_dim, clip_emb_dim)).to("cpu")
    with open(f'{outdir}/ridge_image_weights.pkl', 'rb') as f:
        image_datadict = pickle.load(f)
    model = Ridge(
        alpha=100000,
        max_iter=50000,
        random_state=42,
    )
    model.coef_ = image_datadict["coef"]
    model.intercept_ = image_datadict["intercept"]
    pred_clip_image = torch.from_numpy(model.predict(voxels[:,0]).reshape(-1, clip_seq_dim, clip_emb_dim))

    if dual_guidance:
        with open(f'{outdir}/ridge_text_weights.pkl', 'rb') as f:
            text_datadict = pickle.load(f)
        pred_clip_text = torch.zeros((len(all_images), clip_text_seq_dim, clip_text_emb_dim)).to("cpu")
        model = Ridge(
            alpha=100000,
            max_iter=50000,
            random_state=42,
        )
        model.coef_ = text_datadict["coef"]
        model.intercept_ = text_datadict["intercept"]
        pred_clip_text = torch.from_numpy(model.predict(voxels[:,0]).reshape(-1, clip_text_seq_dim, clip_text_emb_dim))

    if prompt_recon:
        with open(f'{outdir}/ridge_prompt_weights.pkl', 'rb') as f:
            prompt_datadict = pickle.load(f)
        pred_git_text = torch.zeros((len(all_images), git_seq_dim, git_emb_dim)).to("cpu")
        model = Ridge(
            alpha=100000,
            max_iter=50000,
            random_state=42,
        )
        model.coef_ = prompt_datadict["coef"]
        model.intercept_ = prompt_datadict["intercept"]
        pred_git_text = torch.from_numpy(model.predict(voxels[:,0]).reshape(-1, git_seq_dim, git_emb_dim))

    if blurry_recon:
        pred_blurry_vae = torch.zeros((len(all_images), latent_emb_dim)).to("cpu")
        with open(f'{outdir}/ridge_blurry_weights.pkl', 'rb') as f:
            blurry_datadict = pickle.load(f)
        model = Ridge(
            alpha=100000,
            max_iter=50000,
            random_state=42,
        )
        model.coef_ = blurry_datadict["coef"]
        model.intercept_ = blurry_datadict["intercept"]
        pred_blurry_vae = torch.from_numpy(model.predict(voxels[:,0]).reshape(-1, latent_emb_dim))    

    if retrieval:
        pred_retrieval = torch.zeros((len(all_images), retrieval_seq_dim, retrieval_emb_dim)).to("cpu")
        with open(f'{outdir}/ridge_retrieval_weights.pkl', 'rb') as f:
            retrieval_datadict = pickle.load(f)
        model = Ridge(
            alpha=100000,
            max_iter=50000,
            random_state=42,
        )
        voxels_norm = torch.nn.functional.normalize(voxels[:,0], p=2, dim=1)
        model.coef_ = retrieval_datadict["coef"]
        model.intercept_ = retrieval_datadict["intercept"]
        pred_retrieval = torch.from_numpy(model.predict(voxels_norm).reshape(-1, retrieval_seq_dim, retrieval_emb_dim))
        
        
    if normalize_preds:
        std_pred_clip_image = (pred_clip_image - torch.mean(pred_clip_image,axis=0)) / (torch.std(pred_clip_image,axis=0) + 1e-6)
        pred_clip_image = std_pred_clip_image * torch.std(clip_image_train,axis=0) + torch.mean(clip_image_train,axis=0)
        del clip_image_train
        if dual_guidance:
            std_pred_clip_text = (pred_clip_text - torch.mean(pred_clip_text,axis=0)) / (torch.std(pred_clip_text,axis=0) + 1e-6)
            pred_clip_text = std_pred_clip_text * torch.std(clip_text_train,axis=0) + torch.mean(clip_text_train,axis=0)
            del clip_text_train
        if blurry_recon:
            std_pred_blurry_vae = (pred_blurry_vae - torch.mean(pred_blurry_vae,axis=0)) / (torch.std(pred_blurry_vae,axis=0) + 1e-6)
            pred_blurry_vae = std_pred_blurry_vae * torch.std(vae_image_train,axis=0) + torch.mean(vae_image_train,axis=0)
            del vae_image_train
        if retrieval:
            std_pred_retrieval = (pred_retrieval - torch.mean(pred_retrieval,axis=0)) / (torch.std(pred_retrieval,axis=0) + 1e-6)
            pred_retrieval = std_pred_retrieval * torch.std(retrieval_image_train,axis=0) + torch.mean(retrieval_image_train,axis=0)
            # L2 Normalize for optimal cosine similarity
            pred_retrieval = torch.nn.functional.normalize(pred_retrieval, p=2, dim=2)
            del retrieval_image_train
        if prompt_recon:
            for sequence in range(git_seq_dim):
                std_pred_git_text = (pred_git_text[:, sequence] - torch.mean(pred_git_text[:, sequence],axis=0)) / (torch.std(pred_git_text[:, sequence],axis=0) + 1e-6)
                pred_git_text[:, sequence] = std_pred_git_text * torch.std(git_text_train[:, sequence],axis=0) + torch.mean(git_text_train[:, sequence],axis=0)
            del git_text_train


# In[ ]:


if gtc:
    emb_batch_size = 1
    gt_captions = np.load(f"{data_path}/preprocessed_data/captions_18.npy")
    file_path_txt = f"{data_path}/preprocessed_data/gt/{text_embedding_variant}_text_embeddings.pt"
    os.makedirs(f"{data_path}/preprocessed_data/gt", exist_ok=True)
    if not os.path.exists(file_path_txt):
        # Generate CLIP Text embeddings
        print("Generating Text embeddings!")
        pred_clip_text = torch.zeros((len(gt_captions), clip_text_seq_dim, clip_text_emb_dim)).to("cpu")
        for i in tqdm(range(len(gt_captions) // emb_batch_size), desc="Encoding captions..."):
            batch_captions = gt_captions[i * emb_batch_size:i * emb_batch_size + emb_batch_size].tolist()
            pred_clip_text[i * emb_batch_size:i * emb_batch_size + emb_batch_size] =  reconstructor.embed_text(batch_captions).to("cpu")
        torch.save(pred_clip_text, file_path_txt)
    else:
        pred_clip_text = torch.load(file_path_txt)


# In[ ]:


final_recons = None
final_blurryrecons = None
if save_raw:
    model_tag = f"{strength}-str_{textstrength}-mix"
    if gt:
        model_tag += "_gt"
    elif gtc:
        model_tag += "_gtc"
    raw_root = f"{raw_path}/{mode}/mirage_augmentations/{model_tag}/subject{subj}/"
    print("raw_root:", raw_root)
    os.makedirs(raw_root,exist_ok=True)
    torch.save(pred_clip_image, f"{raw_root}/{image_embedding_variant}_image_voxels.pt")
    if dual_guidance:
        torch.save(pred_clip_text, f"{raw_root}/{text_embedding_variant}_text_voxels.pt")
    if blurry_recon:
        torch.save(pred_blurry_vae, f"{raw_root}/{latent_embedding_variant}_latent_voxels.pt")
    if retrieval:
        torch.save(pred_retrieval, f"{raw_root}/{retrieval_embedding_variant}_retrieval_voxels.pt")


for idx in tqdm(range(0,voxels.shape[0]), desc="sample loop"):
    clip_voxels = pred_clip_image[idx]
    if dual_guidance:
        clip_text_voxels = pred_clip_text[idx]
    else:
        clip_text_voxels = None
    
    latent_voxels=None
    if blurry_recon:
        latent_voxels = pred_blurry_vae[idx].unsqueeze(0)
        blurred_image = vdvae.reconstruct(latents=latent_voxels)
        if filter_sharpness:
            # This helps make the output not blurry when using the VDVAE
            blurred_image = ImageEnhance.Sharpness(blurred_image).enhance(20)
        if filter_contrast:
            # This boosts the structural impact of the blurred_image
            blurred_image = ImageEnhance.Contrast(blurred_image).enhance(1.5)
        im = transforms.ToTensor()(blurred_image)
        if final_blurryrecons is None:
            final_blurryrecons = im.cpu()
        else:
            final_blurryrecons = torch.vstack((final_blurryrecons, im.cpu()))
                
    samples = reconstructor.reconstruct(image=blurred_image,
                                        c_i=clip_voxels,
                                        c_t=clip_text_voxels,
                                        n_samples=gen_rep,
                                        textstrength=textstrength,
                                        strength=strength)
    
    if save_raw:
        os.makedirs(f"{raw_root}/{idx}/", exist_ok=True)
        for rep in range(gen_rep):
            transforms.ToPILImage()(samples[rep]).save(f"{raw_root}/{idx}/{rep}.png")
        
        if rep == 0:
            transforms.ToPILImage()(all_images[idx]).save(f"{raw_root}/{idx}/ground_truth.png")
            if blurry_recon:
                transforms.ToPILImage()(transforms.ToTensor()(blurred_image).cpu()).save(f"{raw_root}/{idx}/low_level.png")
            torch.save(clip_voxels, f"{raw_root}/{idx}/clip_image_voxels.pt")
            if dual_guidance:
                torch.save(clip_text_voxels, f"{raw_root}/{idx}/clip_text_voxels.pt")
            if prompt_recon:
                with open(f"{raw_root}/{idx}/predicted_caption.txt", "w") as f:
                    f.write(all_predcaptions[idx])


# In[ ]:


if not utils.is_interactive():
    sys.exit(0)

