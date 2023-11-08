# For training
seed=1203
lr=1e-4
optim=Adam
sched=Cyclic
wd=1e-6
emb_dim_ff=128
emb_dim=64
model_name=FastRx
log_dir=lr$lr\_ddi0.06_dim$emb_dim\_alpha0.95_thres0.5_visitlg2_$optim\_wd$wd\_$sched\1e-4_$model_name\_emb_dim_ff$emb_dim_ff\_seed$seed\_triangular1_3
train_log=saved/$model_name/$log_dir/train.log

rm -f $train_log && python FastRx.py --dim $emb_dim --lr $lr --cuda 0 --log_dir $log_dir --model_name $model_name --optim $optim --sched $sched --weight_decay $wd | tee -a $train_log