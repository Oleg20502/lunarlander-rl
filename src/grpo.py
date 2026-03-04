import csv
import os
import numpy as np
from tqdm.auto import tqdm
import gymnasium as gym
import torch
import torch.optim as optim
import argparse

from reinforce import NeuralPolicy


def train_grpo(
    env_maker,
    n_iter=300,
    n_groups=32,
    group_size=16,
    n_epochs=10,
    batch_size=64,
    alpha=3e-4,
    clip_eps=0.2,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    adv_eps=1e-8,
    hidden_size=128,
    device='cpu',
    seed=None,
    exp_dir=None,
    save_every_n=50,
):
    """
    Vectorized GRPO: all n_groups × group_size episodes run in parallel.
    Per group, envs share the same initial state (seed). Returns are
    normalized within each group to produce the group-relative advantage.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if exp_dir is not None:
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        metrics_path = os.path.join(exp_dir, "metrics.csv")
        _CSV_FIELDS = [
            'iteration', 'mean_return', 'best_return',
            'policy_loss', 'entropy',
            'ep_return_std', 'ep_return_min', 'ep_return_max',
        ]
        with open(metrics_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    total_envs = n_groups * group_size
    envs = gym.vector.SyncVectorEnv([env_maker for _ in range(total_envs)], copy=False)
    n_features = envs.single_observation_space.shape[0]
    n_actions  = envs.single_action_space.n

    policy    = NeuralPolicy(n_features, n_actions, hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=alpha)

    mean_returns      = []
    best_return       = -np.inf
    best_policy_state = None

    pbar = tqdm(range(n_iter), desc="Train Iter...")

    for iteration in pbar:
        seeds = []
        for _ in range(n_groups):
            s = int(np.random.randint(0, 2**31))
            seeds.extend([s] * group_size)

        obs, _ = envs.reset(seed=seeds)
        obs = torch.FloatTensor(obs).to(device)

        active      = torch.ones(total_envs, dtype=torch.bool, device=device)
        env_returns = torch.zeros(total_envs, device=device)

        obs_buf  = []
        act_buf  = []
        logp_buf = []
        mask_buf = []

        while active.any():
            with torch.no_grad():
                dist    = policy.get_dist(obs)
                actions = dist.sample()
                logps   = dist.log_prob(actions)

            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            dones_np = terminated | truncated

            obs_buf.append(obs)
            act_buf.append(actions)
            logp_buf.append(logps)
            mask_buf.append(active.clone())

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones_t   = torch.tensor(dones_np, dtype=torch.bool, device=device)

            env_returns += rewards_t * active.float()
            active = active & ~dones_t

            obs = torch.FloatTensor(next_obs).to(device)

        obs_all  = torch.stack(obs_buf)    # [T, total_envs, n_features]
        act_all  = torch.stack(act_buf)    # [T, total_envs]
        logp_all = torch.stack(logp_buf)   # [T, total_envs]
        mask_all = torch.stack(mask_buf)   # [T, total_envs]

        ret_grouped = env_returns.view(n_groups, group_size)
        grp_mean = ret_grouped.mean(dim=1, keepdim=True)
        grp_std  = ret_grouped.std(dim=1, keepdim=True) + adv_eps
        adv_per_env = ((ret_grouped - grp_mean) / grp_std).view(-1)   # [total_envs]
        adv_all = adv_per_env.unsqueeze(0).expand_as(act_all)         # [T, total_envs]

        valid     = mask_all
        obs_flat  = obs_all[valid]    # [B, n_features]
        act_flat  = act_all[valid]    # [B]
        logp_flat = logp_all[valid]   # [B]
        adv_flat  = adv_all[valid]    # [B]

        B = obs_flat.shape[0]
        episode_returns = env_returns.cpu().numpy()

        total_policy_loss = 0.0
        total_entropy     = 0.0
        n_updates = 0

        indices = np.arange(B)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, B, batch_size):
                mb_idx = indices[start:start + batch_size]
                mb_obs      = obs_flat[mb_idx]
                mb_act      = act_flat[mb_idx]
                mb_old_logp = logp_flat[mb_idx]
                mb_adv      = adv_flat[mb_idx]

                dist     = policy.get_dist(mb_obs)
                new_logp = dist.log_prob(mb_act)
                entropy  = dist.entropy().mean()

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                loss = policy_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_entropy     += entropy.item()
                n_updates += 1

        avg_policy_loss = total_policy_loss / n_updates
        avg_entropy     = total_entropy     / n_updates

        mean_ret = float(episode_returns.mean())
        if mean_ret > best_return:
            best_return       = mean_ret
            best_policy_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        mean_returns.append(mean_ret)

        pbar.set_postfix({
            'mean_ret': f"{mean_ret:.1f}",
            'best':     f"{best_return:.1f}",
            'π_loss':   f"{avg_policy_loss:.3f}",
        })

        if exp_dir is not None:
            step_metrics = {
                'iteration':     iteration,
                'mean_return':   round(mean_ret, 4),
                'best_return':   round(best_return, 4),
                'policy_loss':   round(avg_policy_loss, 6),
                'entropy':       round(avg_entropy, 6),
                'ep_return_std': round(float(episode_returns.std()), 4),
                'ep_return_min': round(float(episode_returns.min()), 4),
                'ep_return_max': round(float(episode_returns.max()), 4),
            }
            with open(metrics_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(step_metrics)

            if (iteration + 1) % save_every_n == 0 and best_policy_state is not None:
                ckpt = {
                    'iteration': iteration,
                    'policy':    best_policy_state,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_iter_{iteration + 1:04d}_best_return_{best_return:.2f}.pt"))

    envs.close()

    if best_policy_state is not None:
        policy.load_state_dict(best_policy_state)
        if exp_dir is not None:
            torch.save(best_policy_state, os.path.join(exp_dir, "best_policy.pt"))

    return policy, mean_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter",            type=int,   default=300)
    parser.add_argument("--n-groups",          type=int,   default=32)
    parser.add_argument("--group-size",        type=int,   default=16)
    parser.add_argument("--n-epochs",          type=int,   default=10)
    parser.add_argument("--batch-size",        type=int,   default=64)
    parser.add_argument("--alpha",             type=float, default=3e-4)
    parser.add_argument("--clip-eps",          type=float, default=0.2)
    parser.add_argument("--entropy-coef",      type=float, default=0.01)
    parser.add_argument("--max-grad-norm",     type=float, default=0.5)
    parser.add_argument("--hidden-size",       type=int,   default=128)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--exp-dir",           type=str,   default="runs/grpo_exp_001")
    parser.add_argument("--save-every-n",      type=int,   default=50)
    parser.add_argument("--device",            type=str,   default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    def make_env():
        return gym.make(
            "LunarLander-v3",
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
            render_mode=None,
        )

    policy, returns = train_grpo(
        env_maker=make_env,
        n_iter=args.n_iter,
        n_groups=args.n_groups,
        group_size=args.group_size,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_size=args.hidden_size,
        device=device,
        seed=args.seed,
        exp_dir=args.exp_dir,
        save_every_n=args.save_every_n,
    )
