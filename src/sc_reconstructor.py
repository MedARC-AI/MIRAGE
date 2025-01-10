import os
import sys
from PIL import Image
import PIL
import numpy as np
import torch
import yaml
from tqdm import tqdm
from torchvision import transforms
sys.path.append("StableCascade/")
from inference.utils import *
from train import WurstCoreC, WurstCoreB


class SC_Reconstructor(object):
    def __init__(self, device="cuda:0", cache_dir="../cache/", embedder_only=False, compile_models=False):
        print(f"Stable Cascade Reconstructor: Loading model...")
        self.device = device
        self.cache_dir = cache_dir
        self.dtype = torch.bfloat16
        # SETUP STAGE C
        config_c = {
                "model_version": "3.6B",
                "dtype": "bfloat16",
                "effnet_checkpoint_path": f"{cache_dir}/effnet_encoder.safetensors",
                "previewer_checkpoint_path": f"{cache_dir}/previewer.safetensors",
                "generator_checkpoint_path": f"{cache_dir}/stage_c_bf16.safetensors",
            }

        self.core = WurstCoreC(config_dict=config_c, device=device, training=False)
        # SETUP MODELS & DATA
        self.extras = self.core.setup_extras_pre()
        self.models = self.core.setup_models(self.extras)
        self.models.generator.eval().requires_grad_(False)
        if compile_models:
            self.models = WurstCoreC.Models(
            **{**self.models.to_dict(), 'generator': torch.compile(self.models.generator, mode="reduce-overhead", fullgraph=True)}
            )
        print("STAGE C READY")
        
        if not embedder_only:
            # SETUP STAGE B
            config_b = {
                "model_version": "3B",
                "dtype": "bfloat16",
                "batch_size": 4,
                "image_size": 1024,
                "grad_accum_steps": 1,
                "effnet_checkpoint_path": f"{cache_dir}/effnet_encoder.safetensors",
                "stage_a_checkpoint_path": f"{cache_dir}/stage_a.safetensors",
                "generator_checkpoint_path": f"{cache_dir}/stage_b_bf16.safetensors"
            }
                
            self.core_b = WurstCoreB(config_dict=config_b, device=device, training=False)
        
            self.extras_b = self.core_b.setup_extras_pre()
            self.models_b = self.core_b.setup_models(self.extras_b, skip_clip=True)
            self.models_b = WurstCoreB.Models(
            **{**self.models_b.to_dict(), 'tokenizer': self.models.tokenizer, 'text_model': self.models.text_model}
            )
            self.models_b.generator.bfloat16().eval().requires_grad_(False)
            if compile_models:
                self.models_b = WurstCoreB.Models(
                **{**self.models_b.to_dict(), 'generator': torch.compile(self.models_b.generator, mode="reduce-overhead", fullgraph=True)}
                )
            print("STAGE B READY")
    
    def embed_image(self, images, hidden=False):
        if isinstance(images, PIL.Image.Image):
            images = resize_image(images).to(self.device)
        elif isinstance(images, list):
            images = torch.stack([resize_image(image).to(self.device) for image in images])
        if images.dim() == 3:
            images = images.unsqueeze(0)
        preprocessed_images = self.extras.clip_preprocess(images)
        outputs = self.models.image_model(preprocessed_images)
        if hidden:
            return outputs.last_hidden_state
        else:
            return outputs.image_embeds.unsqueeze(1)
    
    def embed_text(self, text):
        clip_tokens_unpooled = self.models.tokenizer(text, truncation=True, padding="max_length",
                                                    max_length=self.models.tokenizer.model_max_length,
                                                    return_tensors="pt").to(self.device)
        text_encoder_output = self.models.text_model(**clip_tokens_unpooled, output_hidden_states=True)
        return text_encoder_output.hidden_states[-1]
    
    def embed_latent(self, images):
        if isinstance(images, PIL.Image.Image):
            images = resize_image(images).to(self.device)
        elif isinstance(images, list):
            images = torch.stack([resize_image(image).to(self.device) for image in images])
        if images.dim() == 3:
            images = images.unsqueeze(0)
        latent_batch = {'images': images}
        effnet_latents = self.core.encode_latents(latent_batch, self.models, self.extras)
        return effnet_latents
        
    def reconstruct(self,
                    image=None, 
                    latent=None,
                    c_i=None, 
                    c_t=None, 
                    n_samples=1, 
                    textstrength=0.5, 
                    strength=1.0, 
                    seed=None,
                    num_steps_c = 20,
                    cfg_c=4,
                    cfg_b=1.1,
                    shift_c=2,
                    shift_b=1,
                    num_steps_b=10,
                    uncond_multiplier=0,
                    guidance_ratio=1):
        height, width = 1024, 1024
        stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=n_samples)
        self.extras.sampling_configs['cfg'] = cfg_c
        self.extras.sampling_configs['shift'] = shift_c
        self.extras_b.sampling_configs['cfg'] = cfg_b
        self.extras_b.sampling_configs['shift'] = shift_b
        self.extras_b.sampling_configs['timesteps'] = num_steps_b
        self.extras_b.sampling_configs['t_start'] = 1.0
        if strength < 1.0: # Prepare partially noised latents
            if latent is not None:
                effnet_latents = latent.reshape((-1, 16, 24, 24)).expand(n_samples, -1, -1, -1).to(self.device, self.dtype)
            elif image is not None:
                effnet_latents = self.embed_latent(image).expand(n_samples, -1, -1, -1).to(self.device, self.dtype)
            else:
                raise ValueError("Image must be provided for strength < 1.0")
            t = torch.ones(effnet_latents.size(0), device=self.device) * strength
            noised = self.extras.gdf.diffuse(effnet_latents, t=t)[0]
            self.extras.sampling_configs['timesteps'] = int(num_steps_c * strength)
            self.extras.sampling_configs['t_start'] = strength
            self.extras.sampling_configs['x_init'] = noised
        else:
            self.extras.sampling_configs['timesteps'] = num_steps_c
            self.extras.sampling_configs['t_start'] = 1.0
            
        if int(strength * num_steps_c) > 0:
            
            # Prep CLIP guidance, we are only guiding the first stage
            conditions = {"clip_text" : c_t.reshape((-1, 77, 1280)).expand(n_samples, -1, -1), 
                        "clip_text_pooled" : c_t.mean(dim=1).to(self.device, self.dtype), # Placeholder, will replace with uncond guidance
                        "clip_img" : c_i.reshape((-1,1,768)).expand(n_samples, -1, -1)}
            #Unconditional guidance
            batch = {'captions': [""] * n_samples}
            conditions_b = self.core_b.get_conditions(batch, self.models_b, self.extras_b, is_eval=True, is_unconditional=False)
            unconditions = self.core.get_conditions(batch, self.models, self.extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
            # These don't matter for guidance
            conditions["clip_text_pooled"] = unconditions["clip_text_pooled"].to(self.device, self.dtype)
            unconditions_b = self.core_b.get_conditions(batch, self.models_b, self.extras_b, is_eval=True, is_unconditional=True)
            
            # Mix the guidance according to text strength, and stabilize with unconditional guidance for small values
            imgstrength = (1-textstrength) 
            textstrength = textstrength
            conditions['clip_img'] = conditions['clip_img'].to(self.device, self.dtype) * imgstrength + unconditions['clip_img'].to(self.device, self.dtype) * textstrength **5
            conditions['clip_text'] = conditions['clip_text'].to(self.device, self.dtype) * textstrength + unconditions['clip_text'].to(self.device, self.dtype) * imgstrength **5
            conditions['clip_img'] = conditions['clip_img'] + unconditions['clip_img'] * uncond_multiplier
            conditions['clip_text'] = conditions['clip_text'] + unconditions['clip_text'] * uncond_multiplier
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Sampling Stage C
                sampling_c = self.extras.gdf.sample(
                    self.models.generator, conditions, stage_c_latent_shape,
                    unconditions, device=self.device, **self.extras.sampling_configs,
                )
                for (sampled_c, _, _) in tqdm(sampling_c, total=self.extras.sampling_configs['timesteps']):
                    sampled_c = sampled_c
                conditions_b['effnet'] = sampled_c
                unconditions_b['effnet'] = torch.zeros_like(sampled_c)
        else:
            batch = {'captions': [""] * n_samples}
            conditions_b = self.core_b.get_conditions(batch, self.models_b, self.extras_b, is_eval=True, is_unconditional=False)
            conditions_b['effnet'] = effnet_latents
            unconditions_b = self.core_b.get_conditions(batch, self.models_b, self.extras_b, is_eval=True, is_unconditional=True)
            unconditions_b['effnet'] = torch.zeros_like(effnet_latents)
            
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Sampling Stage B
            sampling_b = self.extras_b.gdf.sample(
                self.models_b.generator, conditions_b, stage_b_latent_shape,
                unconditions_b, device=self.device, **self.extras_b.sampling_configs
            )
            for (sampled_b, _, _) in tqdm(sampling_b, total=self.extras_b.sampling_configs['timesteps']):
                sampled_b = sampled_b
            sampled = self.models_b.stage_a.decode(sampled_b).float()
        return sampled.clamp(0, 1)