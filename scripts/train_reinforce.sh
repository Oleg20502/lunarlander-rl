source .venv/bin/activate

baseline="gae_no_nn"

python src/reinforce.py \
  --n-iter            300       \
  --n-episodes        512       \
  --num-envs          64        \
  --alpha             1e-3      \
  --gamma             0.99      \
  --entropy-coef      0.01      \
  --baseline          $baseline    \
  --baseline-lr       1e-4     \
  --hidden-size       512       \
  --seed              42        \
  --exp-dir           runs/reinforce_$baseline \
  --save-every-n      50
