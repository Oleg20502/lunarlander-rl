source .venv/bin/activate

# grpo_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/grpo_exp_001_500/

# python src/eval.py \
#   --checkpoint $grpo_ckpt \
#   --results-dir results/grpo \
#   --n-episodes  100

# ppo_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_001_500/checkpoints/ckpt_iter_0060_best_return_252.97.pt
# ppo_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_001_500/checkpoints/ckpt_iter_0040_best_return_231.80.pt
ppo_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_001_500/checkpoints/ckpt_iter_0070_best_return_252.97.pt
python src/eval.py \
  --checkpoint $ppo_ckpt \
  --results-dir results/ppo \
  --n-episodes  100