source .venv/bin/activate
source .venv_pt7/bin/activate

# CUDA_VISIBLE_DEVICES=7 python src/ppo.py \
#   --n-iter            300       \
#   --num-envs          64        \
#   --alpha             3e-4      \
#   --gamma             0.99      \
#   --entropy-coef      0.01      \
#   --hidden-size       1024       \
#   --seed              42        \
#   --save-every-n      10 \
#   --exp-dir           runs/ppo_exp_005_kl_0.7 \
#   --batch-size 512 \
#   --clip-eps 0.1 \
#   --max-grad-norm 1.0 \
#   --beta-kl 0.7

# CUDA_VISIBLE_DEVICES=7 python src/ppo.py \
#   --n-iter            300       \
#   --num-envs          64        \
#   --alpha             3e-4      \
#   --gamma             0.99      \
#   --entropy-coef      0.01      \
#   --hidden-size       1024       \
#   --seed              42        \
#   --save-every-n      10 \
#   --exp-dir           runs/ppo_exp_005_kl_1.0 \
#   --batch-size 512 \
#   --clip-eps 0.1 \
#   --max-grad-norm 1.0 \
#   --beta-kl 1.0

for clip in 0.01 0.05 0.2 0.5 1.0; do
CUDA_VISIBLE_DEVICES=7 python src/ppo.py \
  --n-iter            300       \
  --num-envs          64        \
  --alpha             3e-4      \
  --gamma             0.99      \
  --entropy-coef      0.01      \
  --hidden-size       1024       \
  --seed              42        \
  --save-every-n      10 \
  --exp-dir           runs/ppo_exp_006_clip_${clip}_kl_0.5 \
  --batch-size 512 \
  --clip-eps ${clip} \
  --max-grad-norm 1.0 \
  --beta-kl 0.5
done