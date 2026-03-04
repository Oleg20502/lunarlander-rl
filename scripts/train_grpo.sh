source .venv/bin/activate
source .venv_pt7/bin/activate

# CUDA_VISIBLE_DEVICES=6 python src/grpo.py \
#   --n-iter            500       \
#   --n-groups          32        \
#   --group-size        16        \
#   --alpha             3e-4      \
#   --entropy-coef      0.01      \
#   --hidden-size       1024       \
#   --seed              42        \
#   --save-every-n      10 \
#   --exp-dir           runs/grpo_fixed_groups \
#   --batch-size 512 \
#   --clip-eps 0.1 \
#   --max-grad-norm 1.0

CUDA_VISIBLE_DEVICES=7 python src/grpo.py \
  --n-iter            300       \
  --n-groups          32        \
  --group-size        16        \
  --alpha             3e-4      \
  --entropy-coef      0.01      \
  --hidden-size       1024       \
  --seed              42        \
  --save-every-n      10 \
  --exp-dir           runs/grpo_fixed_groups_v2 \
  --batch-size 512 \
  --clip-eps 0.1 \
  --max-grad-norm 1.0
