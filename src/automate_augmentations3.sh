jupyter nbconvert recon_inference_mi_augmentations2.ipynb --to python
export CUDA_VISIBLE_DEVICES="2"

for subj in 1; do
    model_name="subj0${subj}_40sess_hypatia_mirage"
    
    for strength in 1.0; do
        for mode in "imagery"; do

            python recon_inference_mi_augmentations2.py \
                --model_name $model_name \
                --subj $subj \
                --mode $mode \
                --cache_dir ../cache \
                --data_path ../dataset \
                --save_raw \
                --raw_path /export/raid1/home/kneel027/nsd_imagery_journal_paper/reconstructions/ \
                --no-retrieval \
                --no-prompt_recon \
                --textstrength 1 \
                --strength $strength \
                --no-compile_models \
                --gt

            done
        done
    done

for subj in 1 2 5 7; do
    model_name="subj0${subj}_40sess_hypatia_mirage"
    
    for strength in 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
        for mode in "vision"; do #imagery

            python recon_inference_mi_augmentations2.py \
                --model_name $model_name \
                --subj $subj \
                --mode $mode \
                --cache_dir ../cache \
                --data_path ../dataset \
                --save_raw \
                --raw_path /export/raid1/home/kneel027/nsd_imagery_journal_paper/reconstructions/ \
                --no-retrieval \
                --no-prompt_recon \
                --textstrength 1 \
                --strength $strength \
                --no-compile_models 

            done
        done
    done