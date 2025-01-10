# MindEye Imagery

This branch (ME2_backbone_VD) uses the MindEye2 backbone paired with Reese's new portable Versatile Diffusion installation to produce reconstructions.

This new VD installation appears to perform much worse when paired with the ME2 backbone (although it does great for ridge regression), we have no idea why this is, but if you want to use the ME2 backbone with the ME1 implementation of versatile diffusion (which works better with MLP backbones), see the other branch titled ME2_backbone_VD_old.

This branch requires no special environment, mostly just ME2 but with Reese's VD package installed from source:
`pip install git+https://github.com/reesekneeland/Versatile-Diffusion.git`

You will need to place the versatile diffusion model files from the following hugginface repo in your 'cache_dir':
https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth

## Installation

1. Git clone this repository:

```
git clone https://github.com/MedARC-AI/MindEye_Imagery.git
```

2. Download necessary project files from the two hugginface repositories and place them in the same folder as your git clone.
    - https://huggingface.co/datasets/pscotti/mindeyev2
    - https://huggingface.co/datasets/reesekneeland/mindeye_imagery
    
Warning: **This will download over 300 GB of data!** You may want to only download some parts of the huggingface dataset (e.g., not all the pretrained models contained in "train_logs", only one of the preparations of brain activity—whole brain or not—whole brain betas are only necessary for SNR thresholding.)

```
cd MindEyeV2
git clone https://huggingface.co/datasets/pscotti/mindeyev2 .
git clone https://huggingface.co/datasets/reesekneeland/mindeye_imagery .
```

or for specifically downloading only parts of the dataset (will need to edit depending on what you want to download):
```
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", allow_patterns="*.tar",
    local_dir= "your_local_dir", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="coco_images_224_float16.hdf5", repo_type="dataset")
```

3. Run ```. src/setup.sh``` to install a new "mei-env" virtual environment. Make sure the virtual environment is activated with "source mei-env/bin/activate".

## Usage

- ```src/Train.ipynb``` trains models (both single-subject and multi-subject). Check the argparser arguments to specify how you want to train the model (e.g., ```--num_sessions=1``` to train with 1-hour of data).
    - Final models used in the paper were trained on an 8xA100 80GB node and will OOM on weaker compute. You can train the model on weaker compute with minimal performance impact by changing certain model arguments: We recommend lowering hidden_dim to 1024 (or even 512), removing the low-level submodule (``--no-blurry_recon``), and lowering the batch size.
    - To train a single-subject model, set ```--no-multi_subject``` and ```--subj=#``` where # is the subject from NSD you wish to train
    - To train a multi-subject model (i.e., pretraining), set ```--multi_subject``` and ```--subj=#``` where # is the one subject out of 8 NSD subjects to **not** include in the pretraining.
    - To fine-tune from a multi-subject model, set ```--no-multi_subject``` and ```--multisubject_ckpt=path_to_your_pretrained_ckpt_folder```
- ```src/recon_inference_mi.ipynb``` will run inference on the NSD Imagery dataset using a pretrained model, outputting tensors of reconstructions/predicted captions/etc.
- ```src/final_evaluations_multi_mi.ipynb``` will visualize reconstructions the best and median output from ```src/recon_inference_mi``` and compute quantitative metrics.
- See .slurm files for example scripts for running the .ipynb notebooks as batch jobs submitted to Slurm job scheduling.
