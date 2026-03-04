source .venv/bin/activate
source .venv_pt7/bin/activate

CUDA_VISIBLE_DEVICES=6 python src/grpo.py \
  --n-iter            500       \
  --num-envs          64        \
  --alpha             3e-4      \
  --gamma             0.99      \
  --entropy-coef      0.01      \
  --hidden-size       1024       \
  --seed              42        \
  --save-every-n      10 \
  --exp-dir           runs/grpo_exp_002 \
  --batch-size 512 \
  --clip-eps 0.1 \
  --max-grad-norm 1.0
