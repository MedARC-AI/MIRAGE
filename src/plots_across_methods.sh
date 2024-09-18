jupyter nbconvert plots_across_methods.ipynb --to python

python plots_across_methods.py \
    --methods "mindeye1_subj01,final_subj01_pretrained_40sess_24bs,pretrained_subj01_40sess_hypatia_vd2,pretrained_subj01_40sess_hypatia_vd_dual_proj" \
    --data_path ../dataset \
    --output_path ../figs/