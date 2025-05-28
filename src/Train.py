#!/usr/bin/env python
# coding: utf-8

# # Import packages & functions

# In[1]:


import os
import sys
import json
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
import gc
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
from sklearn.linear_model import Ridge
import pickle
# custom functions #
import utils
from sc_reconstructor import SC_Reconstructor
from vdvae import VDVAE


# # Configurations

# In[ ]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "subj01_40sess_hypatia_ridge_scsubj01_40sess_hypatia_ridge_sc_medium_captions"
    print("model_name:", model_name)
    
    # global_batch_size and batch_size should already be defined in the 2nd cell block
    jupyter_args = f"--data_path=../dataset/ \
                    --cache_dir=../cache/ \
                    --model_name={model_name} \
                    --batch_size=64 \
                    --no-multi_subject --subj=1 --num_sessions=40 \
                    --dual_guidance --prompt_recon --caption_type medium"

    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[3]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
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
    "--num_sessions", type=float, default=40,
    help="Number of training sessions to include",
)
parser.add_argument(
    "--prompt_recon",action=argparse.BooleanOptionalAction, default=True,
    help="Use for prompt generating",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
    help="whether to output blurry reconstructions",
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--weight_decay",type=int,default=100000,
)
parser.add_argument(
    "--max_iter",type=int,default=50000,
)
parser.add_argument(
    "--dual_guidance",action=argparse.BooleanOptionalAction,default=True,
    help="Use the decoded captions for dual guidance",
)
parser.add_argument(
    "--caption_type",type=str,default='medium',choices=['coco','short', 'medium', 'schmedium'],
)
parser.add_argument(
    "--retrieval",action=argparse.BooleanOptionalAction,default=True,
    help="Use the decoded captions for dual guidance",
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()
print(f"args: {args}")
# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'../train_logs/{model_name}')
os.makedirs(outdir,exist_ok=True)
device = "cuda"


# # Prep data, models, and dataloaders

# In[ ]:


x_train, valid_nsd_ids_train, x_test, test_nsd_ids = utils.load_nsd(subject=subj, num_sessions=num_sessions, data_path=data_path)
print(x_train.shape, valid_nsd_ids_train.shape)

print(f"Loaded subj {subj} betas!\n")


# ## Prepare git feature

# In[ ]:


if not os.path.exists(f'{data_path}/git_image_features.hdf5'):
    print("Creating Git Feature...")
    from PIL import Image
    import requests
    from transformers import AutoProcessor, GitVisionModel, AutoModelForCausalLM, GitModel
    from modeling_git import GitForCausalLMClipEmb
    # Load 73k NSD images
    f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
    beta_images = f['images'] 
    print("Loaded all 73k possible NSD images to cpu!", beta_images.shape)

    git_images = []
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    
    git_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    git_text_model.to(device)
    git_text_model.eval().requires_grad_(False)
    print("success load Git model")
    for i, image in enumerate(tqdm(beta_images)):
        pil_image = (image.transpose((1, 2, 0))*255).astype(np.uint8)
        inputs = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
        outputs = git_text_model.git.image_encoder(inputs).last_hidden_state
        # valid the captions
        if i <= 5:
            generated_ids = git_text_model.generate(pixel_values=outputs, max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            print(generated_caption)
        git_images.append(outputs.detach().cpu().numpy())


    with h5py.File(f'{data_path}/git_image_features.hdf5', 'w') as f:
        f.create_dataset('features', data=np.array(git_images))
    print("Finished!")
    del beta_images, git_images
else:
    print("git_image_features.hdf5 already exist!")


# In[ ]:


# Load 73k NSD images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)
# images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD images to cpu!", images.shape)

# Load 73k NSD captions
if caption_type == "schmedium":
    captions_small = np.load(f'{data_path}/preprocessed_data/short_length_captions.npy')
    captions_medium = np.load(f'{data_path}/preprocessed_data/mid_length_captions_73K.npy')
    # Create a mask to randomly select elements from both arrays
    mask = np.random.rand(len(captions_small)) > 0.5
    # Mix the arrays based on the mask
    captions = np.where(mask, captions_small, captions_medium)
else:
    if caption_type == "coco":
        caption_file = "annots_73k.npy"
    elif caption_type == "short":
        caption_file = "short_length_captions.npy"
    elif caption_type == "medium":
        caption_file = "mid_length_captions_73K.npy"
    else:
        raise ValueError("Invalid caption type")
    captions = np.load(f'{data_path}/preprocessed_data/{caption_file}')
print("Loaded all 73k NSD captions to cpu!", captions.shape)

train_images = torch.zeros((len(valid_nsd_ids_train), 3, 224, 224))
train_captions = np.zeros((len(valid_nsd_ids_train),), dtype=object)

# Load specific training data
for i, idx in enumerate(valid_nsd_ids_train):
    train_images[i] =  torch.from_numpy(images[idx])
    train_captions[i] = captions[idx]
    
print(f"Filtered down to only the {len(valid_nsd_ids_train)} training images for subject {subj}!")


# ## Load models

# ### Feature extractor model

# In[ ]:


clip_extractor = SC_Reconstructor(compile_models=False, embedder_only=True, device=device, cache_dir=cache_dir)
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


# # Creating block of CLIP embeddings

# In[ ]:


file_path = f"{data_path}/preprocessed_data/subject{subj}/{image_embedding_variant}_image_embeddings_train_{num_sessions}sess.pt"
emb_batch_size = 50
if not os.path.exists(file_path):
    # Generate CLIP Image embeddings
    print("Generating Image embeddings!")
    clip_image_train = torch.zeros((len(train_images), clip_seq_dim, clip_emb_dim)).to("cpu")
    for i in tqdm(range(len(train_images) // emb_batch_size), desc="Encoding clip images..."):
        batch_list = []
        for img in train_images[i * emb_batch_size:i * emb_batch_size + emb_batch_size]:
            batch_list.append(transforms.ToPILImage()(img))
        clip_image_train[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = clip_extractor.embed_image(batch_list).to("cpu")

    torch.save(clip_image_train, file_path)
else:
    clip_image_train = torch.load(file_path)

        
if dual_guidance:
    emb_batch_size = 50
    file_path_txt = f"{data_path}/preprocessed_data/subject{subj}/{text_embedding_variant}_text_embeddings_train_{num_sessions}sess.pt"
    if not os.path.exists(file_path_txt):
        # Generate CLIP Text embeddings
        print("Generating Text embeddings!")
        clip_text_train = torch.zeros((len(train_captions), clip_text_seq_dim, clip_text_emb_dim)).to("cpu")
        for i in tqdm(range(len(train_captions) // emb_batch_size), desc="Encoding captions..."):
            batch_captions = train_captions[i * emb_batch_size:i * emb_batch_size + emb_batch_size].tolist()
            clip_text_train[i * emb_batch_size:i * emb_batch_size + emb_batch_size] =  clip_extractor.embed_text(batch_captions).to("cpu")
        torch.save(clip_text_train, file_path_txt)
    else:
        clip_text_train = torch.load(file_path_txt)


if blurry_recon:
    emb_batch_size = 1
    file_path = f"{data_path}/preprocessed_data/subject{subj}/{latent_embedding_variant}_latent_embeddings_train_{num_sessions}sess.pt"
    if not os.path.exists(file_path):
        print("Generating Latent Image embeddings!")
        vae_image_train = torch.zeros((len(train_images), latent_emb_dim)).to("cpu")
        for i in tqdm(range(len(train_images)), desc="Encoding blurry images..."):
            img = transforms.ToPILImage()(train_images[i])
            vae_image_train[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = vdvae.embed_latent(img).reshape(-1, latent_emb_dim).to("cpu")
        torch.save(vae_image_train, file_path)
    else:
        vae_image_train = torch.load(file_path)
        
if retrieval:
    file_path = f"{data_path}/preprocessed_data/subject{subj}/{retrieval_embedding_variant}_retrieval_embeddings_train_{num_sessions}sess.pt"
    emb_batch_size = 50
    if not os.path.exists(file_path):
        # Generate CLIP Retrieval embeddings
        print("Generating Retrieval embeddings!")
        retrieval_image_train = torch.zeros((len(train_images), retrieval_seq_dim, retrieval_emb_dim)).to("cpu")
        for i in tqdm(range(len(train_images) // emb_batch_size), desc="Encoding images..."):
            batch_list = []
            for img in train_images[i * emb_batch_size:i * emb_batch_size + emb_batch_size]:
                batch_list.append(transforms.ToPILImage()(img))
            retrieval_image_train[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = clip_extractor.embed_image(batch_list, hidden=True).to("cpu")
        # Normalize for optimal cosine similarity
        retrieval_image_train = torch.nn.functional.normalize(retrieval_image_train, p=2, dim=2)
        torch.save(retrieval_image_train, file_path)
    else:
        retrieval_image_train = torch.load(file_path)
        
# Load 73k GiT NSD features
if prompt_recon:
    file_path_git = f"{data_path}/preprocessed_data/subject{subj}/{prompt_embedding_variant}_prompt_embeddings_train_{num_sessions}sess.pt"
    if not os.path.exists(file_path_git):
        with h5py.File(f'{data_path}/git_image_features.hdf5', 'r') as f:
            git_features = f['features'][:]
        train_git_images = torch.zeros((len(valid_nsd_ids_train), 257,1024))
        for i, idx in enumerate(valid_nsd_ids_train):
            train_git_images[i] = torch.from_numpy(git_features[idx])
        torch.save(train_git_images, file_path_git)
        del git_features
    else:
        train_git_images = torch.load(file_path_git)
        
print(f"Loaded vectors for subj{subj}!")


# # Train Ridge regression models

# In[ ]:


start = time.time()
ridge_weights = np.zeros((clip_seq_dim * clip_emb_dim, x_train.shape[-1])).astype(np.float32)
ridge_biases = np.zeros((clip_seq_dim * clip_emb_dim)).astype(np.float32)
print(f"Training Ridge Image model with alpha={weight_decay}")
model = Ridge(
    alpha=weight_decay,
    max_iter=max_iter,
    random_state=42,
)

model.fit(x_train, clip_image_train.reshape(len(clip_image_train), -1))
ridge_weights = model.coef_
ridge_biases = model.intercept_
datadict = {"coef" : ridge_weights, "intercept" : ridge_biases}
# Save the regression weights
with open(f'{outdir}/ridge_image_weights.pkl', 'wb') as f:
    pickle.dump(datadict, f)
del clip_image_train
del ridge_weights
del ridge_biases
del datadict
    
if dual_guidance:
    ridge_weights_txt = np.zeros((clip_text_seq_dim * clip_text_emb_dim, x_train.shape[-1])).astype(np.float32)
    ridge_biases_txt = np.zeros((clip_text_seq_dim * clip_text_emb_dim)).astype(np.float32)
    print(f"Training Ridge Text model with alpha={weight_decay}")
    model = Ridge(
        alpha=weight_decay,
        max_iter=max_iter,
        random_state=42,
    )

    model.fit(x_train, clip_text_train.reshape(len(clip_text_train), -1))
    ridge_weights_txt = model.coef_
    ridge_biases_txt = model.intercept_
    datadict = {"coef" : ridge_weights_txt, "intercept" : ridge_biases_txt}
    # Save the regression weights
    with open(f'{outdir}/ridge_text_weights.pkl', 'wb') as f:
        pickle.dump(datadict, f)
    
    del clip_text_train
    del ridge_weights_txt
    del ridge_biases_txt
    del datadict
    
if blurry_recon:
    ridge_weights_blurry = np.zeros((latent_emb_dim, x_train.shape[-1])).astype(np.float32)
    ridge_biases_blurry = np.zeros((latent_emb_dim,)).astype(np.float32)
    print(f"Training Ridge Blurry recon model with alpha={weight_decay}")
    model = Ridge(
        alpha=weight_decay,
        max_iter=max_iter,
        random_state=42,
    )
    model.fit(x_train, vae_image_train)
    ridge_weights_blurry = model.coef_
    ridge_biases_blurry = model.intercept_
    datadict = {"coef" : ridge_weights_blurry, "intercept" : ridge_biases_blurry}
    # Save the regression weights
    with open(f'{outdir}/ridge_blurry_weights.pkl', 'wb') as f:
        pickle.dump(datadict, f)
    
    del vae_image_train
    del ridge_weights_blurry
    del ridge_biases_blurry
    del datadict

if retrieval:
    ridge_weights = np.zeros((retrieval_seq_dim * retrieval_emb_dim, x_train.shape[-1])).astype(np.float32)
    ridge_biases = np.zeros((retrieval_seq_dim * retrieval_emb_dim)).astype(np.float32)
    print(f"Training Ridge Retrieval model with alpha={weight_decay}")
    model = Ridge(
        alpha=weight_decay,
        max_iter=max_iter,
        random_state=42,
    )
    x_train_norm = torch.nn.functional.normalize(x_train, p=2, dim=1)
    model.fit(x_train_norm, retrieval_image_train.reshape(len(retrieval_image_train), -1))
    ridge_weights = model.coef_
    ridge_biases = model.intercept_
    datadict = {"coef" : ridge_weights, "intercept" : ridge_biases}
    # Save the regression weights
    with open(f'{outdir}/ridge_retrieval_weights.pkl', 'wb') as f:
        pickle.dump(datadict, f)
    
    del retrieval_image_train
    del ridge_weights
    del ridge_biases
    del datadict
        
if prompt_recon:
    ridge_weights_prompt = np.zeros((git_seq_dim*git_emb_dim, x_train.shape[-1])).astype(np.float32)
    ridge_biases_prompt = np.zeros((git_seq_dim*git_emb_dim)).astype(np.float32)
    print(f"Training Ridge prompt recon model with alpha={weight_decay}")
    model = Ridge(
        alpha=weight_decay,
        max_iter=max_iter,
        random_state=42,
    )
    model.fit(x_train, train_git_images.reshape(len(train_git_images), -1))
    ridge_weights_prompt = model.coef_
    ridge_biases_prompt = model.intercept_
    datadict = {"coef" : ridge_weights_prompt, "intercept" : ridge_biases_prompt}
    # Save the regression weights
    with open(f'{outdir}/ridge_prompt_weights.pkl', 'wb') as f:
        pickle.dump(datadict, f)
        
    del train_git_images
    del ridge_weights_prompt
    del ridge_biases_prompt
    del datadict

print(f"Elapsed training time for {model_name}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

