jupyter nbconvert final_evaluations_mi_multi.ipynb --to python

export CUDA_VISIBLE_DEVICES="3"
# for subj in 1 2 5 7; do
#     for strength in 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
#         for mode in "imagery" "vision"; do
#             model_name="braindiffuser_jp_ts1_subj0${subj}_${strength}"
#             echo "[INFO] Now processing: ${model_name}"
#             python final_evaluations_mi_multi.py \
#                     --model_name $model_name \
#                     --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#                     --subj $subj \
#                     --mode $mode \
#                     --data_path ../dataset \
#                     --cache_dir ../cache

#         done
#     done
# done
for subj in 1 2 5 7; do
    for method in "braindiffuser" "mindeye"; do
        for objective in "clip" "brain_corr"; do
            for topn in {1..64}; do
                model_name="${method}_top${topn}_${objective}_subj0${subj}"
                echo "[INFO] Now processing: ${model_name}"
                python final_evaluations_mi_multi.py \
                        --model_name $model_name \
                        --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons.pt \
                        --subj $subj \
                        --mode "shared1000" \
                        --data_path ../dataset \
                        --cache_dir ../cache
            done
        done
    done
done