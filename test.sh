## FastRx's result
model_name=FastRx
log_dir=lr1e-4_ddi0.06_dim64_alpha0.95_thres0.5_visitlg2_Adam_wd1e-6_Cyclic1e-4_FastRx_emb_dim_ff128_seed1203_triangular1_3
resume_path=Epoch_72_TARGET_0.06_JA_0.5422_DDI_0.06674.model

python FastRx.py --cuda 0 --log_dir $log_dir \
        --model_name $model_name --Test --resume_path $resume_path

# ## Ablation Study

# # FastRx without Diagnosis
# model_name=FastRx_wo_Diag
# log_dir=lr1e-4_ddi0.06_dim64_alpha0.95_thres0.5_visitlg2_Adam_wd1e-6_Reduce4e-5_FastRx_wo_Diag_emb_dim_ff128_seed1203
# resume_path=Epoch_77_TARGET_0.06_JA_0.5102_DDI_0.06711.model

# python FastRx.py --cuda 0 --log_dir $log_dir \
#         --model_name $model_name --Test --resume_path $resume_path

# # FastRx without Procedure
# model_name=FastRx_wo_Proc
# log_dir=lr1e-4_ddi0.06_dim64_alpha0.95_thres0.5_visitlg2_Adam_wd1e-6_Reduce4e-5_FastRx_wo_Proc_emb_dim_ff128_seed1203
# resume_path=Epoch_85_TARGET_0.06_JA_0.5221_DDI_0.06868.model

# python FastRx.py --cuda 0 --log_dir $log_dir \
#         --model_name $model_name --Test --resume_path $resume_path

# # FastRx without GCN
# model_name=FastRx_wo_GCN
# log_dir=lr1e-4_ddi0.06_dim64_alpha0.95_thres0.5_visitlg2_Adam_wd1e-6_Reduce4e-5_FastRx_wo_GCN_emb_dim_ff128_seed1203
# resume_path=Epoch_85_TARGET_0.06_JA_0.5394_DDI_0.06939.model

# python FastRx.py --cuda 0 --log_dir $log_dir \
#         --model_name $model_name --Test --resume_path $resume_path

# # FastRx without 1D-CNN
# model_name=FastRx_wo_CNN1D
# log_dir=lr1e-4_ddi0.06_dim64_alpha0.95_thres0.5_visitlg2_Adam_wd1e-6_Reduce4e-5_FastRx_wo_CNN1D_emb_dim_ff128_seed1203
# resume_path=Epoch_61_TARGET_0.06_JA_0.542_DDI_0.06931.model

# python FastRx.py --cuda 0 --log_dir $log_dir \
#         --model_name $model_name --Test --resume_path $resume_path
