jupyter nbconvert plots_across_methods.ipynb --to python

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     pretrained_subj01_40sess_hypatia_vd2, \
#     subj01_40sess_hypatia_turbo_ridge_flat, \
#     pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
#     pretrained_subj01_40sess_hypatia_vd_dual_proj_wd_1, \
#     subj01_40sess_hypatia_turbo_ridge_flat_vd_clip, \
#     subj01_40sess_hypatia_turbo_ridge_flat_vd_clip_new_vd"  \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --output_file methods_scatter_reduced

#     python plots_across_methods.py \
# --methods "mindeye1_subj01, \
# braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# pretrained_subj01_40sess_hypatia_vd2, \
# pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
# subj01_40sess_hypatia_turbo_ridge_flat,
# subj01_40sess_hypatia_turbo_ridge_seq,
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_scatter_reduced.png




# python plots_across_methods.py \
# --methods "mindeye1_subj01, \
# braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# pretrained_subj01_40sess_hypatia_vd2, \
# pretrained_subj01_40sess_hypatia_vd_snr_0_5, \
# pretrained_subj01_40sess_hypatia_vd_multisubject_snr_0_5, \
# pretrained_subj01_40sess_hypatia_vd_snr_0_55, \
# pretrained_subj01_40sess_hypatia_vd_multisubject_snr_0_6, \
# pretrained_subj01_40sess_hypatia_vd_snr_0_65, \
# pretrained_subj01_40sess_hypatia_vd_snr_0_75" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_snr

# python plots_across_methods.py \
# --methods "mindeye1_subj01, \
# braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# pretrained_subj01_40sess_hypatia_vd2, \
# pretrained_subj01_40sess_hypatia_vd_dual_proj, \
# pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
# pretrained_subj01_40sess_hypatia_ip_adapter2, \
# pretrained_subj01_40sess_hypatia_ip_adapter_plus" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_snr


# python plots_across_methods.py \
# --methods "braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# subj01_40sess_hypatia_turbo_ridge_alpha_0.01, \
# subj01_40sess_hypatia_turbo_ridge_alpha_0.1, \
# subj01_40sess_hypatia_turbo_ridge_alpha_1, \
# subj01_40sess_hypatia_turbo_ridge_alpha_10, \
# subj01_40sess_hypatia_turbo_ridge_alpha_100, \
# subj01_40sess_hypatia_turbo_ridge_alpha_1000, \
# subj01_40sess_hypatia_turbo_ridge_alpha_10000, \
# subj01_40sess_hypatia_turbo_ridge_alpha_60000, \
# subj01_40sess_hypatia_turbo_ridge_alpha_100000" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_ridge_alpha

python plots_across_methods.py \
--methods "braindiffuser_subj01, \
final_subj01_pretrained_40sess_24bs, \
subj01_40sess_hypatia_ridge_rank_order_rois_1, \
subj01_40sess_hypatia_ridge_rank_order_rois_2, \
subj01_40sess_hypatia_ridge_rank_order_rois_3, \
subj01_40sess_hypatia_ridge_rank_order_rois_4, \
subj01_40sess_hypatia_ridge_rank_order_rois_5, \
subj01_40sess_hypatia_ridge_rank_order_rois_6, \
subj01_40sess_hypatia_ridge_rank_order_rois_7, \
subj01_40sess_hypatia_ridge_rank_order_rois_8, \
subj01_40sess_hypatia_ridge_rank_order_rois_9, \
subj01_40sess_hypatia_ridge_rank_order_rois_10, \
subj01_40sess_hypatia_ridge_rank_order_rois_11, \
subj01_40sess_hypatia_ridge_rank_order_rois_12, \
subj01_40sess_hypatia_ridge_rank_order_rois_13, \
subj01_40sess_hypatia_ridge_rank_order_rois_14, \
subj01_40sess_hypatia_ridge_rank_order_rois_15, \
subj01_40sess_hypatia_ridge_rank_order_rois_16, \
subj01_40sess_hypatia_ridge_rank_order_rois_17, \
subj01_40sess_hypatia_ridge_rank_order_rois_18, \
subj01_40sess_hypatia_ridge_rank_order_rois_19, \
subj01_40sess_hypatia_ridge_rank_order_rois_20, \
subj01_40sess_hypatia_ridge_rank_order_rois_21, \
subj01_40sess_hypatia_ridge_rank_order_rois_22, \
subj01_40sess_hypatia_ridge_rank_order_rois_23, \
subj01_40sess_hypatia_ridge_rank_order_rois_24, \
subj01_40sess_hypatia_ridge_rank_order_rois_25" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file methods_comparison_roi_threshold