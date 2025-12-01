export CUDA_VISIBLE_DEVICES=0
nohup python ../scripts/train_stage5_joint_training.py --config ../configs/training_config_full_model.yaml > /root/autodl-tmp/vln_training_outputs/training_stage5.log 2>&1 &
