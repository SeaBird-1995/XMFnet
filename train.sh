

set -x
# python train_pl.py --cfg cfg/official.yaml --exp_name watercraft_resume --checkpoint experiments/lightning_logs/version_6/checkpoints/last.ckpt
# python train_pl.py --cfg cfg/official.yaml --exp_name four_gpus --checkpoint experiments/lightning_logs/version_8/checkpoints/last.ckpt


############### The Model Distillation ###############
# python train_pl.py --cfg cfg/model_distill.yaml --exp_name model_distill_high_kd_loss

############### The 3D only model ###############
python train_pl.py --cfg cfg/model_3d_only.yaml --exp_name model_3d_only
