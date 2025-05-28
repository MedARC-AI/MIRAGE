jupyter nbconvert final_evaluations_mi_multi.ipynb --to python

export CUDA_VISIBLE_DEVICES="0"
# for subj in 1 2 5 7; do
#     for mode in "imagery" "vision"; do
#         model_name="mindbridge_subj0${subj}"
#         echo "[INFO] Now processing: ${model_name}"
#         python final_evaluations_mi_multi.py \
#                 --model_name $model_name \
#                 --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#                 --subj $subj \
#                 --mode $mode \
#                 --data_path ../dataset \
#                 --cache_dir ../cache
#     done
# done

for subj in 1 2 5 7; do
    for mode in "imagery" "vision"; do
        model_name="brainram_subj0${subj}"
        echo "[INFO] Now processing: ${model_name}"
        python final_evaluations_mi_multi.py \
                --model_name $model_name \
                --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                --subj $subj \
                --mode $mode \
                --data_path ../dataset \
                --cache_dir ../cache
    done
done

for subj in 1 2 5 7; do
    for mode in "imagery" "vision"; do
        model_name="neuropictor_subj0${subj}"
        echo "[INFO] Now processing: ${model_name}"
        python final_evaluations_mi_multi.py \
                --model_name $model_name \
                --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                --subj $subj \
                --mode $mode \
                --data_path ../dataset \
                --cache_dir ../cache
    done
done