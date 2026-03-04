source .venv/bin/activate
source .venv_pt7/bin/activate

grpo_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/grpo_fixed_groups_v2/checkpoints/ckpt_iter_0100_best_return_165.67.pt

CUDA_VISIBLE_DEVICES=7 python src/eval.py \
  --checkpoint $grpo_ckpt \
  --results-dir results/grpo/grpo_fixed_groups_v2 \
  --n-episodes  100


# ppo_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_001_500/best_policy.pt

# CUDA_VISIBLE_DEVICES=6 python src/eval.py \
#   --checkpoint $ppo_ckpt \
#   --results-dir results/ppo \
#   --n-episodes  100


########### REINFORCE ###########

# reinforce_none_v3_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs_reinforce/reinforce_none_v3
# reinforce_nn_v3_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs_reinforce/reinforce_nn_v3
# reinforce_gae_no_nn_v2_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs_reinforce/reinforce_gae_no_nn_v2
# reinforce_gae_nn_v2_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs_reinforce/reinforce_gae_nn_v2

# for ckpt in $reinforce_none_v3_ckpt $reinforce_nn_v3_ckpt $reinforce_gae_no_nn_v2_ckpt $reinforce_gae_nn_v2_ckpt; do  
# CUDA_VISIBLE_DEVICES=6 python src/eval.py \
#     --checkpoint $ckpt/best_policy.pt \
#     --results-dir results/reinforce/$(basename $ckpt) \
#     --n-episodes  100
# done

########### PPO, different beta_kl ###########

# ppo_exp_005_kl_0_2_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_005_kl_0.2
# ppo_exp_005_kl_0_5_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_005_kl_0.5
# ppo_exp_004_kl_0_01_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_004_kl_0.01
# ppo_exp_003_kl_0_1_ckpt=/home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_003_kl_0.1

# for clip in 0.01 0.05 0.2 0.5 1.0; do
# CUDA_VISIBLE_DEVICES=7 python src/eval.py \
#     --checkpoint /home/jovyan/avarlamov/skoltech/lunarlander-rl/runs/ppo_exp_006_clip_${clip}_kl_0.5/best_policy.pt \
#     --results-dir results/ppo/ppo_exp_006_clip_${clip}_kl_0.5 \
#     --n-episodes  100
# done