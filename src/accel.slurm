#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=p5
#SBATCH --job-name=pg_mi_1c
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --qos=top

cd /weka/proj-fmri/ckadirt/imagery_clean/MindEye_Imagery/src
source /admin/home-ckadirt/mei-env/bin/activate

jupyter nbconvert TrainSC2.ipynb --to python
# jupyter nbconvert recon_inference_mi.ipynb --to python
# jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
# jupyter nbconvert plots_across_subjects.ipynb --to python
# jupyter nbconvert plots_across_methods.ipynb --to python




export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=36 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

subj=1 
model_name="subj01_stable_cascade_test_no_prior_mse_ME1_"
data_path="/weka/proj-medarc/shared/mindeyev2_dataset"
cache_dir="/weka/proj-medarc/shared/cache"
echo model_name=${model_name}
python TrainSC2.py \
    --data_path=/weka/proj-medarc/shared/mindeyev2_dataset \
    --cache_dir=/weka/proj-medarc/shared/cache \
    --model_name=${model_name} \
    --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --num_sessions=40 \
    --hidden_dim=1024 --clip_scale=1. \
    --blurry_recon --blur_scale=.5 --no-use_prior --resume_from_ckpt \
    --seq_past=0 --seq_future=0 \
    --prior_scale=30 --wandb_log \
    --n_blocks=4 --max_lr=9e-5 --mixup_pct=.33 --num_epochs=150 --no-use_image_aug \
    --ckpt_interval=5 --ckpt_saving --dual_guidance --caption_type medium \


    # --data_path=${data_path} \
    # --cache_dir=${cache_dir} \
    # --model_name=${pretrain_model_name} \
    # --multi_subject --subj=${subj} \
    # --batch_size=${BATCH_SIZE} \
    # --max_lr=3e-5 \
    # --mixup_pct=.33 \
    # --num_epochs=150 \
    # --use_prior \
    # --prior_scale=30 \
    # --clip_scale=1 \
    # --blur_scale=.5 \
    # --no-use_image_aug \
    # --n_blocks=4 \
    # --hidden_dim=1024 \
    # --num_sessions=40 \
    # --ckpt_interval=5 \
    # --ckpt_saving \
    # --wandb_log \
    # --dual_guidance

# # singlesubject finetuning

# model_name="p_trained_subj0${subj}_40sess_hypatia_new_vd_dual_proj"
# echo model_name=${model_name}
# python Train.py --data_path=${data_path} \
#     --cache_dir=${cache_dir} \
#     --model_name=${model_name} \
#     --no-multi_subject \
#     --subj=${subj} \
#     --batch_size=${BATCH_SIZE} \
#     --max_lr=3e-5 \
#     --mixup_pct=.33 \
#     --num_epochs=150 \
#     --use_prior \
#     --prior_scale=30 \
#     --clip_scale=1 \
#     --blur_scale=.5 \
#     --no-use_image_aug \
#     --n_blocks=4 \
#     --hidden_dim=1024 \
#     --num_sessions=40 \
#     --ckpt_interval=5 \
#     --ckpt_saving \
#     --wandb_log \
#     --multisubject_ckpt=../train_logs/${pretrain_model_name} \
#     --dual_guidance 


# for mode in "imagery" "vision"; do
#     python recon_inference_mi.py \
#         --model_name $model_name \
#         --subj $subj \
#         --mode $mode \
#         --cache_dir=${cache_dir} \
#         --data_path ../dataset \
#         --hidden_dim 1024 \
#         --n_blocks 4 \
#         --dual_guidance

#     python final_evaluations_mi_multi.py \
#             --model_name $model_name \
#             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#             --subj $subj \
#             --mode $mode \
#             --data_path=${data_path} \
#             --cache_dir=${cache_dir} \
#             --imagery_data_path=/weka/proj-medarc/shared/umn-imagery \

#     python plots_across_subjects.py \
#             --model_name="${model_name}" \
#             --mode="${mode}" \
#             --data_path=${data_path} \
#             --cache_dir=${cache_dir} \
#             --criteria all \
#             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#             --subjs=$subj \
#             --imagery_data_path=/weka/proj-medarc/shared/umn-imagery \

#     done

# python plots_across_methods.py \
# --methods "mindeye1_subj01, \
# braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# pretrained_subj01_40sess_hypatia_vd2, \
# pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
# subj01_40sess_hypatia_turbo_ridge_flat3,
# pretrained_subj01_40sess_hypatia_new_vd_dual_proj" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_scatter_new_vd_main