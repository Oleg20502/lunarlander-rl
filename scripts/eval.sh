source .venv/bin/activate

checkpoint="/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/reinforce_gae_nn/checkpoints/ckpt_iter_0050_best_return_47.03.pt"
python src/eval.py \
  --checkpoint $checkpoint \
  --results-dir results/reinforce_gae_nn          \
  --n-episodes  10                       \
  --deterministic
