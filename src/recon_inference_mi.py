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


if prompt_recon:
    from transformers import AutoProcessor
    from modeling_git import GitForCausalLMClipEmb
    all_predcaptions = []
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    git_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    git_text_model.to(device) 
    git_text_model.eval().requires_grad_(False)

    for pred_text in pred_git_text:
        pred_embedding = pred_text.to(device).to(torch.float32).unsqueeze(0)
        generated_ids = git_text_model.generate(pixel_values=pred_embedding, max_length=20)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_predcaptions = np.hstack((all_predcaptions, generated_caption))
    torch.save(all_predcaptions,f"evals/{model_name}/{model_name}_all_predcaptions_{mode}.pt")


# In[ ]:


final_recons = None
final_blurryrecons = None
if save_raw:
    raw_root = f"{raw_path}/{mode}/{model_name}/subject{subj}/"
    print("raw_root:", raw_root)
    os.makedirs(raw_root,exist_ok=True)
    torch.save(pred_clip_image, f"{raw_root}/{image_embedding_variant}_image_voxels.pt")
    if dual_guidance:
        torch.save(pred_clip_text, f"{raw_root}/{text_embedding_variant}_text_voxels.pt")
    if blurry_recon:
        torch.save(pred_blurry_vae, f"{raw_root}/{latent_embedding_variant}_latent_voxels.pt")
    if retrieval:
        torch.save(pred_retrieval, f"{raw_root}/{retrieval_embedding_variant}_retrieval_voxels.pt")

if num_images_per_sample == 1:
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
            transforms.ToPILImage()(all_images[idx]).save(f"{raw_root}/{idx}/ground_truth.png")
            transforms.ToPILImage()(transforms.ToTensor()(blurred_image).cpu()).save(f"{raw_root}/{idx}/low_level.png")
            torch.save(clip_voxels, f"{raw_root}/{idx}/clip_image_voxels.pt")
            if dual_guidance:
                torch.save(clip_text_voxels, f"{raw_root}/{idx}/clip_text_voxels.pt")

        if final_recons is None:
            final_recons = samples.unsqueeze(0).cpu()
        else:
            final_recons = torch.cat((final_recons, samples.unsqueeze(0).cpu()), dim=0)
else:
    for rep in tqdm(range(gen_rep)):
        utils.seed_everything(seed = random.randint(0,10000000))
        # get all reconstructions    
        all_blurryrecons = None
        all_recons = None
        
        minibatch_size = 1
        plotting = False
        for idx in tqdm(range(0,voxels.shape[0]), desc="sample loop"):
            clip_voxels = pred_clip_image[idx]
            if dual_guidance:
                clip_text_voxels = pred_clip_text[idx]
            else:
                clip_text_voxels = None
                
            blurred_image=None
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
                if all_blurryrecons is None:
                    all_blurryrecons = im.cpu()
                else:
                    all_blurryrecons = torch.vstack((all_blurryrecons, im.cpu()))
                    
            if retrieval:
                retrieval_voxels = pred_retrieval[idx].unsqueeze(0)
            else:
                retrieval_voxels = clip_voxels
            samples_multi = []
            for _ in range(num_images_per_sample // 8):
                samples_multi.append(reconstructor.reconstruct(image=blurred_image,
                                                    c_i=clip_voxels,
                                                    c_t=clip_text_voxels,
                                                    n_samples=min(8, num_images_per_sample),
                                                    textstrength=textstrength,
                                                    strength=strength))
            samples_multi = torch.vstack(samples_multi)
            samples = utils.pick_best_recon(samples_multi, retrieval_voxels, reconstructor, hidden=retrieval).unsqueeze(0)
            if all_recons is None:
                all_recons = samples.cpu()
            else:
                all_recons = torch.vstack((all_recons, samples.cpu()))
            
            if save_raw:
                os.makedirs(f"{raw_root}/{idx}/", exist_ok=True)
                transforms.ToPILImage()(samples[0]).save(f"{raw_root}/{idx}/{rep}.png")
                
                if rep == 0:
                    os.makedirs(f"{raw_root}/{idx}/retrieval_images/", exist_ok=True)
                    for r_idx, image in enumerate(samples_multi):
                        transforms.ToPILImage()(image).save(f"{raw_root}/{idx}/retrieval_images/{r_idx}.png")
                    transforms.ToPILImage()(all_images[idx]).save(f"{raw_root}/{idx}/ground_truth.png")
                    if blurry_recon:
                        transforms.ToPILImage()(transforms.ToTensor()(blurred_image).cpu()).save(f"{raw_root}/{idx}/low_level.png")
                    torch.save(clip_voxels, f"{raw_root}/{idx}/clip_image_voxels.pt")
                    if dual_guidance:
                        torch.save(clip_text_voxels, f"{raw_root}/{idx}/clip_text_voxels.pt")
                    if prompt_recon:
                        with open(f"{raw_root}/{idx}/predicted_caption.txt", "w") as f:
                            f.write(all_predcaptions[idx])
            
        if final_recons is None:
            final_recons = all_recons.unsqueeze(1)
            if blurry_recon:
                final_blurryrecons = all_blurryrecons.unsqueeze(1)
        else:
            final_recons = torch.cat((final_recons, all_recons.unsqueeze(1)), dim=1)
            if blurry_recon:
                final_blurryrecons = torch.cat((final_blurryrecons, all_blurryrecons.unsqueeze(1)), dim=1)
        
if blurry_recon:
    torch.save(final_blurryrecons,f"evals/{model_name}/{model_name}_all_blurryrecons_{mode}.pt")
torch.save(final_recons,f"evals/{model_name}/{model_name}_all_recons_{mode}.pt")
print(f"saved {model_name} mi outputs!")


# In[ ]:


if not utils.is_interactive():
    sys.exit(0)

