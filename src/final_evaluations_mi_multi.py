#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import math
import time
import random
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import torch
import evaluate
import torch.nn as nn
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from einops.layers.torch import Rearrange
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
import clip
import scipy as sp
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import utils
from models import GNet8_Encoder
import contextlib


# In[2]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "p_trained_subj01_40sess_hypatia_new_vd_dual_proj"
    # model_name = "pretest_pretrained_subj01_40sess_hypatia_pg_sessions40"
    mode = "vision"
    # all_recons_path = f"evals/{model_name}/{model_name}_all_enhancedrecons_{mode}.pt"
    all_recons_path = f"evals/{model_name}/{model_name}_all_recons_{mode}.pt"
    subj = 1
    
    cache_dir = "/weka/proj-medarc/shared/cache"
    data_path = "/weka/proj-medarc/shared/mindeyev2_dataset"
    
    print("model_name:", model_name)

    jupyter_args = f"--model_name={model_name} --subj={subj} --data_path={data_path} --cache_dir={cache_dir} --all_recons_path={all_recons_path} --mode {mode} \
                    --criteria=all --imagery_data_path=/weka/proj-medarc/shared/umn-imagery"
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
    "--all_recons_path", type=str,
    help="Path to where all_recons.pt is stored",
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
    help="Evaluate on which subject?",
)
parser.add_argument(
    "--mode",type=str,default="vision",
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--criteria",type=str, default="all",
)
parser.add_argument(
    "--imagery_data_path",type=str, default=None
)
parser.add_argument(
    "--output_path",type=str, default="tables/"
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)


if not imagery_data_path:
    imagery_data_path = data_path
    
# seed all random functions
utils.seed_everything(seed)
device = "cuda"


# # Evals

# In[ ]:


if mode == "synthetic":
    all_images = torch.zeros((284, 3, 714, 1360))
    all_images[:220] = torch.load(f"{imagery_data_path}/nsddata_stimuli/stimuli/nsdsynthetic/nsd_synthetic_stim_part1.pt")
    #The last 64 stimuli are slightly different for each subject, so we load these separately for each subject
    all_images[220:] = torch.load(f"{imagery_data_path}/nsddata_stimuli/stimuli/nsdsynthetic/nsd_synthetic_stim_part2_sub{subj}.pt")
    voxels, stimulus = utils.load_nsd_synthetic(subject=subj, average=False, nest=True, data_root=imagery_data_path)

elif mode == "shared1000":
    x_train, valid_nsd_ids_train, x_test, test_nsd_ids = utils.load_nsd(subject=subj, data_path=data_path)
    voxels = torch.mean(x_test, dim=1, keepdim=True)
    print(f"Loaded subj {subj} test betas! {voxels.shape}")
    f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
    images = f['images']
    # for the shared1000 git caption generation
    # from transformers import AutoProcessor
    # from modeling_git import GitForCausalLMClipEmb
    # processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    # git_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    # git_text_model.to(device) 
    # git_text_model.eval().requires_grad_(False)
    # all_git_generated_captions = []

    all_images = torch.zeros((len(test_nsd_ids), 3, 224, 224))
    for i, idx in enumerate(test_nsd_ids):
        all_images[i] =  torch.from_numpy(images[idx])
        # for the git captions
    #     pil_image = (images[idx].transpose((1, 2, 0))*255).astype(np.uint8)
    #     inputs = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
    #     outputs = git_text_model.git.image_encoder(inputs).last_hidden_state
    #     generated_ids = git_text_model.generate(pixel_values=outputs, max_length=50)
    #     decoded_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    #     all_git_generated_captions = np.hstack((all_git_generated_captions, decoded_caption))
    # torch.save(all_git_generated_captions,f"evals/{model_name}/shared1000_imgcaptions.pt")
    del images, f
    print(f"Filtered down to only the {len(test_nsd_ids)} test images for subject {subj}!")
else:
    all_images = torch.load(f"{imagery_data_path}/nsddata_stimuli/stimuli/imagery_stimuli_18.pt")
    voxels, _ = utils.load_nsd_mental_imagery(subject=subj, mode=mode, stimtype="all", average=True, nest=False, data_root=imagery_data_path)
    voxels = voxels[:12]
num_voxels = voxels.shape[-1]
num_test = voxels.shape[0]


# In[ ]:


print("all_recons_path:", all_recons_path)
print("all_recons_path:", all_recons_path)

# Determine the target image dimension
target_dim = 512
final_recons = torch.load(all_recons_path)
gen_rep = final_recons.shape[1]
# Resize the images if necessary
if final_recons.shape[-1] != target_dim:
    resize_transform = transforms.Resize((target_dim, target_dim))
    final_recons_resized = torch.zeros((num_test, gen_rep, 3, target_dim, target_dim))
    for sample in range(num_test):
        for rep in range(gen_rep):
            final_recons_resized[sample, rep] = resize_transform(final_recons[sample, rep])
    final_recons = final_recons_resized
final_recons = final_recons.to(torch.float32).to("cpu")
    

print("final_recons.shape:", final_recons.shape)

# Residual submodule
try:
    all_clipvoxels_mult = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels_{mode}.pt").reshape((18, 257, 768))
    print("all_clipvoxels_mult.shape:", all_clipvoxels_mult.shape)
    clip_enabled = True
except:
    clip_enabled = False
# Low-level submodule
if blurry_recon:
    all_blurryrecons_mult = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons_{mode}.pt")

# model name
model_name_plus_suffix = f"{model_name}_all_recons_{mode}"
print(model_name_plus_suffix)
print(all_images.shape, final_recons.shape)


# In[7]:


# ground truths, if using NSD-Imagery, we load only the first 12 because the last 6 are conceptual stimuli, for which there was no "ground truth image" to calculate statistics against
if mode == "imagery" or mode == "vision":
    all_images = all_images[:12]
    final_recons = final_recons[:12]
    if clip_enabled:
        all_clipvoxels = all_clipvoxels[:12]
    if blurry_recon:
        all_blurryrecons = all_blurryrecons[:12]


# ## 2-way identification

# In[12]:


@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=False, feature_name=None):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()
    
    if feature_name is not None:
        feature_path = f"/home/naxos2-raid25/kneel027/home/kneel027/nsd_imagery_journal_paper/features/"
        print(f"Feature: {feature_name}")

        if not os.path.exists(f"{feature_path}{feature_name}_gt.npy"):
            np.save(f"{feature_path}{feature_name}_gt.npy", reals)
        np.save(f"{feature_path}{model_name_plus_suffix}_{repetition}_{feature_name}.npy", preds)
        

    # Compute correlation matrix
    # Each row: features of an image
    # Transpose to have variables as columns
    reals_T = reals.T
    preds_T = preds.T
    r = np.corrcoef(reals_T, preds_T, rowvar=False)
    
    # Extract correlations between reals and preds
    N = len(all_images)
    r = r[:N, N:]  # Shape (N, N)
    
    # Get congruent correlations (diagonal elements)
    congruents = np.diag(r)
    
    # For each reconstructed image, compare its correlation with the correct original image
    # vs. other original images
    success_counts = []
    total_comparisons = N - 1  # Exclude self-comparison
    
    for i in range(N):
        # Correlations of reconstructed image i with all original images
        correlations = r[:, i]
        # Correlation with the correct original image
        congruent = congruents[i]
        # Count how many times the correlation with other images is less than the congruent correlation
        successes = np.sum(correlations < congruent) - 1  # Subtract 1 to exclude the self-comparison
        success_rate = successes / total_comparisons
        success_counts.append(success_rate)
    
    if return_avg:
        # Return the average success rate
        return np.mean(success_counts)
    else:
        # Return the list of success rates per reconstructed image
        return success_counts


# ## PixCorr

# In[13]:


preprocess_pixcorr = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
])

def get_pix_corr(all_images, all_recons, return_avg=False):

    
    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess_pixcorr(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess_pixcorr(all_recons).view(len(all_recons), -1).cpu()
    
    correlations = []
    for i in range(len(all_images)):
        correlations.append(np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1])
    if return_avg:
        return np.mean(correlations)
    else:
        return correlations
    
preprocess_ssim = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR), 
])


# ## SSIM

# In[14]:


preprocess_ssim = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR), 
])

def get_ssim(all_images, all_recons, return_avg=False):

    
    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess_ssim(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess_ssim(all_recons).permute((0,2,3,1)).cpu())
    
    ssim_score=[]
    for im,rec in zip(img_gray,recon_gray):
        ssim_score.append(structural_similarity(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
    if return_avg:
        return np.mean(ssim_score)
    else:
        return ssim_score


# ## AlexNet

# In[15]:


alex_weights = AlexNet_Weights.IMAGENET1K_V1

alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.1', 'features.4', 'features.7', 'features.9', 'features.11', 'classifier.2', 'classifier.5']).to(device)
alex_model.eval().requires_grad_(False)
preprocess_alexnet = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
def get_alexnet(all_images, all_recons, return_avg=False):
    #AlexNet(1)
    alexnet1 = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess_alexnet, 'features.1', return_avg=return_avg, feature_name='alexnet1')
    
    #AlexNet(2)
    alexnet2 = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess_alexnet, 'features.4', return_avg=return_avg, feature_name='alexnet2')
    #AlexNet(3)
    alexnet3 = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess_alexnet, 'features.7', return_avg=return_avg, feature_name='alexnet3')
    #AlexNet(4)
    alexnet4 = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess_alexnet, 'features.9', return_avg=return_avg, feature_name='alexnet4')
    #AlexNet(5)
    alexnet5 = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess_alexnet, 'features.11', return_avg=return_avg, feature_name='alexnet5')
    #AlexNet(6)
    alexnet6 = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess_alexnet, 'classifier.2', return_avg=return_avg, feature_name='alexnet6')
    #AlexNet(7)
    alexnet7 = two_way_identification(all_recons.to(device).float(), all_images, 
                                                            alex_model, preprocess_alexnet, 'classifier.5', return_avg=return_avg, feature_name='alexnet7')
    return alexnet1, alexnet2, alexnet3, alexnet4, alexnet5, alexnet6, alexnet7


# ## InceptionV3

# In[16]:


weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                        return_nodes=['avgpool']).to(device)
inception_model.eval().requires_grad_(False)
preprocess_inception = transforms.Compose([
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
def get_inceptionv3(all_images, all_recons, return_avg=False):
    
    inception = two_way_identification(all_recons.float(), all_images.float(),
                                            inception_model, preprocess_inception, 'avgpool', return_avg=return_avg, feature_name='inceptionV3')
            
    return inception


# ## CLIP

# In[17]:


import clip
clip_model, preprocess = clip.load("ViT-L/14", device=device)
preprocess_clip = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
])

def get_clip(all_images, all_recons, return_avg=False):
    clip_2way = two_way_identification(all_recons, all_images,
                                            clip_model.encode_image, preprocess_clip, None, return_avg=return_avg, feature_name="clip") # final layer
    return clip_2way

def get_clip_cosine(final_embeds, gt_embeds):
    # Get the cosine similarity between the clip embeddings
    # of the final recons and the ground truth images
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = [float(value) for value in cos(final_embeds, gt_embeds)]
    return cos_sim


# ## Efficient Net

# In[18]:


weights = EfficientNet_B1_Weights.DEFAULT
eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                    return_nodes=['avgpool'])
eff_model.eval().requires_grad_(False)
preprocess_efficientnet = transforms.Compose([
    transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
def get_efficientnet(all_images, all_recons, return_avg=False):
    # see weights.transforms()

    
    gt = eff_model(preprocess_efficientnet(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess_efficientnet(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()
    
    effnet = [sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]
    if return_avg:
        return np.mean(effnet)
    else:
        return effnet


# ## SwAV

# In[19]:


swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

swav_model = create_feature_extractor(swav_model, 
                                    return_nodes=['avgpool'])
swav_model.eval().requires_grad_(False)
preprocess_swav = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
def get_swav(all_images, all_recons, return_avg=False):
    gt = swav_model(preprocess_swav(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess_swav(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()
    
    swav = [sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]
    if return_avg:
        return np.mean(swav)
    else:
        return swav


# # Brain Correlation

# In[21]:


# Load brain region masks
try:
    brain_region_masks = {}
    with h5py.File(f"{cache_dir}/brain_region_masks.hdf5", "r") as file:
        # Iterate over each subject
        for subject in file.keys():
            subject_group = file[subject]
            # Load the masks data for each subject
            subject_masks = {"nsd_general" : subject_group["nsd_general"][:],
                             "V1" : subject_group["V1"][:], 
                             "V2" : subject_group["V2"][:], 
                             "V3" : subject_group["V3"][:], 
                             "V4" : subject_group["V4"][:],
                             "higher_vis" : subject_group["higher_vis"][:],
                             "EVC" : subject_group["V1"][:] | subject_group["V2"][:] | subject_group["V3"][:] | subject_group["V4"][:]}
            
            brain_region_masks[subject] = subject_masks
            
    subject_masks = brain_region_masks[f"subj0{subj}"]
except: 
    brain_region_masks = {}
    with h5py.File(f"{data_path}/brain_region_masks.hdf5", "r") as file:
        # Iterate over each subject
        for subject in file.keys():
            subject_group = file[subject]
            # Load the masks data for each subject
            subject_masks = {"nsd_general" : subject_group["nsd_general"][:],
                             "V1" : subject_group["V1"][:], 
                             "V2" : subject_group["V2"][:], 
                             "V3" : subject_group["V3"][:], 
                             "V4" : subject_group["V4"][:],
                             "higher_vis" : subject_group["higher_vis"][:],
                             "EVC" : subject_group["V1"][:] | subject_group["V2"][:] | subject_group["V3"][:] | subject_group["V4"][:]}
            brain_region_masks[subject] = subject_masks
    subject_masks = brain_region_masks[f"subj0{subj}"]


# ### Calculate Brain Correlation scores for each brain area

# In[22]:


from torchmetrics import PearsonCorrCoef

try:
    GNet = GNet8_Encoder(device=device,subject=subj,model_path=f"{cache_dir}/gnet_multisubject.pt")
except:
    GNet = GNet8_Encoder(device=device,subject=subj,model_path=f"{data_path}/gnet_multisubject.pt")
    

def get_brain_correlation(subject_masks, all_recons, return_avg=False):

    # Prepare image list for input to GNet
    recon_list = []
    for i in range(all_recons.shape[0]):
        img = all_recons[i].detach()
        img = transforms.ToPILImage()(img)
        recon_list.append(img)
        
    PeC = PearsonCorrCoef(num_outputs=len(recon_list))
    beta_primes = GNet.predict(recon_list)
    
    region_brain_correlations = {}
    for region, mask in subject_masks.items():
        score = PeC(voxels[:,0,mask].moveaxis(0,1), beta_primes[:,mask].moveaxis(0,1))
        score = score.tolist()
        if return_avg:
            region_brain_correlations[region] = float(torch.mean(score))
        else:
            region_brain_correlations[region] = score
    return region_brain_correlations


# In[23]:


metrics_data = {
            "sample": [],
            "repetition": [],
            "PixCorr": [],
            "SSIM": [],
            "AlexNet(1)": [],
            "AlexNet(2)": [],
            "AlexNet(3)": [],
            "AlexNet(4)": [],
            "AlexNet(5)": [],
            "AlexNet(6)": [],
            "AlexNet(7)": [],
            "InceptionV3": [],
            "CLIP": [],
            "EffNet-B": [],
            "SwAV": [],
            "Brain Corr. nsd_general": [],
            "Brain Corr. higher_vis": [],
            "Brain Corr. EVC": [],
        }

# Iterate over each sample and compute metrics with tqdm and suppressed output
for repetition in tqdm(range(final_recons.shape[1]), desc="Processing samples", file=sys.stdout):
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        rep_recons = final_recons[:, repetition]

        pixcorr = get_pix_corr(all_images, rep_recons)
        ssim = get_ssim(all_images, rep_recons)
        alexnet1, alexnet2, alexnet3, alexnet4, alexnet5, alexnet6, alexnet7 = get_alexnet(all_images, rep_recons)
        inception = get_inceptionv3(all_images, rep_recons)
        clip = get_clip(all_images, rep_recons)
        effnet = get_efficientnet(all_images, rep_recons)
        swav = get_swav(all_images, rep_recons)
        region_brain_correlations = get_brain_correlation(subject_masks, rep_recons)

        # Append each result to its corresponding list, and store the image index
        
        metrics_data["sample"].extend(list(range(final_recons.shape[0])))
        metrics_data["repetition"].extend([repetition for _ in range(final_recons.shape[0])])
        metrics_data["PixCorr"].extend(pixcorr)
        metrics_data["SSIM"].extend(ssim)
        metrics_data["AlexNet(1)"].extend(alexnet1)
        metrics_data["AlexNet(2)"].extend(alexnet2)
        metrics_data["AlexNet(3)"].extend(alexnet3)
        metrics_data["AlexNet(4)"].extend(alexnet4)
        metrics_data["AlexNet(5)"].extend(alexnet5)
        metrics_data["AlexNet(6)"].extend(alexnet6)
        metrics_data["AlexNet(7)"].extend(alexnet7)
        metrics_data["InceptionV3"].extend(inception)
        metrics_data["CLIP"].extend(clip)
        metrics_data["EffNet-B"].extend(effnet)
        metrics_data["SwAV"].extend(swav)
        metrics_data["Brain Corr. nsd_general"].extend(region_brain_correlations["nsd_general"])
        metrics_data["Brain Corr. higher_vis"].extend(region_brain_correlations["higher_vis"])
        metrics_data["Brain Corr. EVC"].extend(region_brain_correlations["EVC"])

# Check that all lists have the same length before creating DataFrame
lengths = [len(values) for values in metrics_data.values()]
if len(set(lengths)) != 1:
    print("Error: Not all metric lists have the same length")
    for metric, values in metrics_data.items():
        print(f"{metric}: {len(values)} items")
else:
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(metrics_data)

    # Save the table to a CSV file
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f'{output_path}{model_name_plus_suffix}.csv', sep='\t')



# In[ ]:


# Print averaged metrics for current table, in order of eval google sheet.
# Remove the 'Unnamed: 0', 'sample', and 'repetition' columns
df = df.drop(columns=['sample', 'repetition'])
# Calculate the average value for each metric column
average_metrics = df.mean()
# Print out the average values in order
for metric, value in average_metrics.items():
    print(f"{value:.5f}")


# ### 

# # Captions Evaluation

# ## Load data

# In[ ]:


if mode == "vision" or mode == "imagery":
    all_git_generated_captions = torch.load(f"{data_path}/preprocessed_data/git_all_imgcaptions_18.pt")
elif mode == "shared1000":
    pass
else:
    assert False, "Not suitable mode for Captions Evaluation!"
    
    
all_predcaptions = torch.load(f"evals/{model_name}/git_all_predcaptions_{mode}.pt")
all_captions = np.load(f"{data_path}/preprocessed_data/captions_18.npy")


# ## Meteor

# In[ ]:


meteor = evaluate.load('meteor')
meteor_img_ref=meteor.compute(predictions=all_git_generated_captions,references=all_captions)
meteor_brain_ref=meteor.compute(predictions=all_predcaptions,references=all_captions)
meteor_brain_img=meteor.compute(predictions=all_predcaptions,references=all_git_generated_captions)


relative_brain_image_meteor=meteor_brain_img["meteor"]/meteor_img_ref["meteor"]

print(f"[GROUND] METEOR GIT from images vs captions: {meteor_img_ref['meteor']}")
print(f"[ABSOLUTE] METEOR GIT from brain vs captions: {meteor_brain_ref['meteor']}")
print(f"[ABSOLUTE] METEOR GIT from brain vs images: {meteor_brain_img['meteor']}")
print(f"[RELATIVE] METEOR  {relative_brain_image_meteor}")


# # Rouge

# In[ ]:


rouge = evaluate.load('rouge')
rouge_img_ref=rouge.compute(predictions=all_git_generated_captions,references=all_captions)
rouge_brain_ref=rouge.compute(predictions=all_predcaptions,references=all_captions)
rouge_brain_img=rouge.compute(predictions=all_predcaptions,references=all_git_generated_captions)


relative_brain_image_rouge1 = rouge_brain_img['rouge1']/rouge_img_ref['rouge1']
relative_brain_image_rougeL = rouge_brain_img['rougeL']/rouge_img_ref['rougeL']

print(f"[GROUND] ROUGE-1 GIT from images vs captions: {rouge_img_ref['rouge1']}")
print(f"[ABSOLUTE] ROUGE-1 GIT from brain vs captions: {rouge_brain_ref['rouge1']}")
print(f"[ABSOLUTE] ROUGE-1 GIT from brain vs images: {rouge_brain_img['rouge1']}")
print(f"[RELATIVE] ROUGE-1  {relative_brain_image_rouge1}")

print(f"[GROUND] ROUGE-L GIT from images vs captions: {rouge_img_ref['rougeL']}")
print(f"[ABSOLUTE] ROUGE-L GIT from brain vs captions: {rouge_brain_ref['rougeL']}")
print(f"[ABSOLUTE] ROUGE-L GIT from brain vs images: {rouge_brain_img['rougeL']}")
print(f"[RELATIVE] ROUGE-L  {relative_brain_image_rougeL}")


# ## Sentence transformer

# In[ ]:


sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


with torch.no_grad():
    embedding_brain= sentence_model.encode(all_predcaptions, convert_to_tensor=True)
    embedding_captions = sentence_model.encode(all_captions, convert_to_tensor=True)
    embedding_images = sentence_model.encode(all_git_generated_captions, convert_to_tensor=True)

    ss_sim_brain_img=util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
    ss_sim_brain_cap=util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
    ss_sim_img_cap=util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()

    relative_brain_image_ss=ss_sim_brain_img.diag().mean()/ss_sim_img_cap.diag().mean()

print(f"[GROUND] Sentence Transformer Similarity GIT from images vs captions: {ss_sim_img_cap.diag().mean()}")
print(f"[ABSOLUTE] Sentence Transformer Similarity GIT from brain vs captions: {ss_sim_brain_cap.diag().mean()}")
print(f"[ABSOLUTE] Sentence Transformer Similarity GIT from brain vs images: {ss_sim_brain_img.diag().mean()}")
print(f"[RELATIVE] Sentence Transformer Similarity   {relative_brain_image_ss.mean()}")


# ## CLIP-B

# In[ ]:


model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer =  AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

with torch.no_grad():
    input_ids=tokenizer(list(all_predcaptions),return_tensors="pt",padding=True)
    embedding_brain= model_clip.get_text_features(**input_ids)

    input_ids=tokenizer(list(all_captions),return_tensors="pt",padding=True)
    embedding_captions= model_clip.get_text_features(**input_ids)

    input_ids=tokenizer(all_git_generated_captions.tolist(),return_tensors="pt",padding=True)
    embedding_images= model_clip.get_text_features(**input_ids)

clip_B_sim_brain_img=util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
clip_B_sim_brain_cap=util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
clip_B_sim_img_cap=util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()

relative_brain_image_clip_B=clip_B_sim_brain_img.diag().mean()/clip_B_sim_img_cap.diag().mean()

print(f"[GROUND] CLIP Similarity GIT from images vs captions: {clip_B_sim_img_cap.diag().mean()}")
print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs captions: {clip_B_sim_brain_cap.diag().mean()}")
print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs images: {clip_B_sim_brain_img.diag().mean()}")
print(f"[RELATIVE] CLIP Similarity   {relative_brain_image_clip_B.mean()}")


# ## CLIP-L

# In[ ]:


model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer =  AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

with torch.no_grad():
    input_ids=tokenizer(list(all_predcaptions),return_tensors="pt",padding=True)
    embedding_brain= model_clip.get_text_features(**input_ids)

    input_ids=tokenizer(list(all_captions),return_tensors="pt",padding=True)
    embedding_captions= model_clip.get_text_features(**input_ids)

    input_ids=tokenizer(all_git_generated_captions.tolist(),return_tensors="pt",padding=True)
    embedding_images= model_clip.get_text_features(**input_ids)

clip_L_sim_brain_img=util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
clip_L_sim_brain_cap=util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
clip_L_sim_img_cap=util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()

relative_brain_image_clip_L=clip_L_sim_brain_img.diag().mean()/clip_L_sim_img_cap.diag().mean()

print(f"[GROUND] CLIP Similarity GIT from images vs captions: {clip_L_sim_img_cap.diag().mean()}")
print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs captions: {clip_L_sim_brain_cap.diag().mean()}")
print(f"[ABSOLUTE] CLIP Similarity GIT from brain vs images: {clip_L_sim_brain_img.diag().mean()}")
print(f"[RELATIVE] CLIP Similarity   {relative_brain_image_clip_L.mean()}")


# ## Save to .csv files

# In[ ]:


caption_metrics={ 
            "Rouge1_img_ref":rouge_img_ref['rouge1'],
            "Rouge1_brain_ref":rouge_brain_ref['rouge1'],
            "Rouge1_brain_img":rouge_brain_img['rouge1'],
            "Rouge1_relative":relative_brain_image_rouge1.mean(),
            "RougeL_img_ref":rouge_img_ref['rougeL'],
            "RougeL_brain_ref":rouge_brain_ref['rougeL'],
            "RougeL_brain_img":rouge_brain_img['rougeL'],
            "RougeL_relative":relative_brain_image_rougeL.mean(),
            "Meteor_img_ref":meteor_img_ref['meteor'],
            "Meteor_brain_ref":meteor_brain_ref['meteor'],
            "Meteor_brain_img":meteor_brain_img['meteor'],
            "Meteor_relative":relative_brain_image_meteor,
            "Sentence_img_ref":ss_sim_img_cap.diag().mean().item(),
            "Sentence_brain_ref":ss_sim_brain_cap.diag().mean().item(),
            "Sentence_brain_img":ss_sim_brain_img.diag().mean().item(),
            "Sentence_relative":relative_brain_image_ss.mean().item(),
            "CLIP-B_img_ref":clip_B_sim_img_cap.diag().mean().item(),
            "CLIP-B_brain_ref":clip_B_sim_brain_cap.diag().mean().item(),
            "CLIP-B_brain_img":clip_B_sim_brain_img.diag().mean().item(),
            "CLIP-B_relative":relative_brain_image_clip_B.mean().item(),
            "CLIP-L_img_ref":clip_L_sim_img_cap.diag().mean().item(),
            "CLIP-L_brain_ref":clip_L_sim_brain_cap.diag().mean().item(),
            "CLIP-L_brain_img":clip_L_sim_brain_img.diag().mean().item(),
            "CLIP-L_relative":relative_brain_image_clip_L.mean().item(),
            }

os.makedirs(output_path,exist_ok=True)
df=pd.DataFrame.from_dict(caption_metrics,orient='index',columns=["Value"])
df.to_csv(f'{output_path}{model_name_plus_suffix}_caption_metrics.csv', sep='\t', index=False)

