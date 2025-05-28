jupyter nbconvert final_evaluations_mi_multi.ipynb --to python

export CUDA_VISIBLE_DEVICES="3"

# Shared1000 TopN recons
for subj in 5 7; do
    for method in "mirage"; do #"mindeye1" "braindiffuser"
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
                        --cache_dir ../cache \
                        --output_path /home/naxos2-raid25/kneel027/home/kneel027/Second-Sight/output/boi_paper/dataframes/
            done
        done
    done
done
