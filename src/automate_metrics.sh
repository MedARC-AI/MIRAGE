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
#                     --cache_dir ../cache \
#                     --output_path /home/naxos2-raid25/kneel027/home/kneel027/nsd_imagery_journal_paper/dataframes/

#         done
#     done
# done
# Shared1000 BOI Recons
# for subj in 1 2 5 7; do
#     for method in "brain-optimized-inference_2.3"; do #"brain-optimized-inference_2.3"
#         for iteration in 0 1 2 3 4 5; do
#             model_name="${method}_iter${iteration}_subj0${subj}"
#             echo "[INFO] Now processing: ${model_name}"
#             python final_evaluations_mi_multi.py \
#                     --model_name $model_name \
#                     --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons.pt \
#                     --subj $subj \
#                     --mode "shared1000" \
#                     --data_path ../dataset \
#                     --cache_dir ../cache \
#                     --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
#         done
#         model_name="${method}_basemodel_subj0${subj}"
#         echo "[INFO] Now processing: ${model_name}"
#         python final_evaluations_mi_multi.py \
#                 --model_name $model_name \
#                 --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons.pt \
#                 --subj $subj \
#                 --mode "shared1000" \
#                 --data_path ../dataset \
#                 --cache_dir ../cache \
#                 --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
#         model_name="${method}_searchrecon_subj0${subj}"
#         echo "[INFO] Now processing: ${model_name}"
#         python final_evaluations_mi_multi.py \
#                 --model_name $model_name \
#                 --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons.pt \
#                 --subj $subj \
#                 --mode "shared1000" \
#                 --data_path ../dataset \
#                 --cache_dir ../cache \
#                 --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
#     done
# done
# subj=1
# mode="imagery"

# for strength in 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
#     model_name="braindiffuser_jp_ts1_gt_${strength}"
#     echo "[INFO] Now processing: ${model_name}"
#     python final_evaluations_mi_multi.py \
#             --model_name $model_name \
#             --all_recons_path evals/${model_name}/${model_name}_all_recons_gt.pt \
#             --subj $subj \
#             --mode $mode \
#             --data_path ../dataset \
#             --cache_dir ../cache \
#             --output_path /home/naxos2-raid25/kneel027/home/kneel027/nsd_imagery_journal_paper/dataframes/

# done

# Shared1000 TopN recons
# for subj in 5 7; do
#     for method in "mirage"; do #"mindeye1" "braindiffuser"
#         for objective in "clip" "brain_corr"; do
#             for topn in {1..64}; do
#                 model_name="${method}_top${topn}_${objective}_subj0${subj}"
#                 echo "[INFO] Now processing: ${model_name}"
#                 python final_evaluations_mi_multi.py \
#                         --model_name $model_name \
#                         --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons.pt \
#                         --subj $subj \
#                         --mode "shared1000" \
#                         --data_path ../dataset \
#                         --cache_dir ../cache \
#                         --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
#             done
#         done
#     done
# done

# NSD-Imagery BOI recons
# for mode in "vision" "imagery"; do
#     for subj in 1 2 5 7; do
#         for method in "brain-optimized-inference_2.1" "brain-optimized-inference_2.3" "brain-optimized-inference_2.4"; do #
#             for iteration in 0 1 2 3 4 5; do
#                 model_name="${method}_iter${iteration}_subj0${subj}"
#                 echo "[INFO] Now processing: ${model_name}"
#                 python final_evaluations_mi_multi.py \
#                         --model_name $model_name \
#                         --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons_${mode}.pt \
#                         --subj $subj \
#                         --mode $mode \
#                         --data_path ../dataset \
#                         --cache_dir ../cache \
#                         --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
#             done
#             model_name="${method}_basemodel_subj0${subj}"
#             echo "[INFO] Now processing: ${model_name}"
#             python final_evaluations_mi_multi.py \
#                     --model_name $model_name \
#                     --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons_${mode}.pt \
#                     --subj $subj \
#                     --mode $mode \
#                     --data_path ../dataset \
#                     --cache_dir ../cache \
#                     --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
#             model_name="${method}_searchrecon_subj0${subj}"
#             echo "[INFO] Now processing: ${model_name}"
#             python final_evaluations_mi_multi.py \
#                     --model_name $model_name \
#                     --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons_${mode}.pt \
#                     --subj $subj \
#                     --mode $mode \
#                     --data_path ../dataset \
#                     --cache_dir ../cache \
#                     --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
#         done
#     done
# done
# # NSD-Imagery TopN recons
# for mode in "vision" "imagery"; do
#     for subj in 1 2 5 7; do
#         for method in "mindeye1" "braindiffuser" "mirage"; do
#             for objective in "clip" "brain_corr"; do
#                 for topn in {1..64}; do
#                     model_name="${method}_top${topn}_${objective}_subj0${subj}"
#                     echo "[INFO] Now processing: ${model_name}"
#                     python final_evaluations_mi_multi.py \
#                             --model_name $model_name \
#                             --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons_${mode}.pt \
#                             --subj $subj \
#                             --mode $mode \
#                             --data_path ../dataset \
#                             --cache_dir ../cache \
#                             --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
#                 done
#             done
#         done
#     done
# done

#NSD-Imagery COI Recons
for mode in "vision"; do #"imagery"
    for subj in 1 2 5 7; do
        for method in "brain-optimized-inference_2.1" "brain-optimized-inference_2.3" "brain-optimized-inference_2.4"; do #
            for iteration in 0 1 2 3 4 5; do
                model_name="${method}_iter${iteration}_subj0${subj}"
                echo "[INFO] Now processing: ${model_name}"
                python final_evaluations_mi_multi.py \
                        --model_name $model_name \
                        --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons_${mode}.pt \
                        --subj $subj \
                        --mode $mode \
                        --data_path ../dataset \
                        --cache_dir ../cache \
                        --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
            done
            model_name="${method}_basemodel_subj0${subj}"
            echo "[INFO] Now processing: ${model_name}"
            python final_evaluations_mi_multi.py \
                    --model_name $model_name \
                    --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons_${mode}.pt \
                    --subj $subj \
                    --mode $mode \
                    --data_path ../dataset \
                    --cache_dir ../cache \
                    --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
            model_name="${method}_searchrecon_subj0${subj}"
            echo "[INFO] Now processing: ${model_name}"
            python final_evaluations_mi_multi.py \
                    --model_name $model_name \
                    --all_recons_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/evals/${model_name}_all_recons_${mode}.pt \
                    --subj $subj \
                    --mode $mode \
                    --data_path ../dataset \
                    --cache_dir ../cache \
                    --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
        done
    done
done