import csv
import os
import numpy as np
from tqdm.auto import tqdm
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence
import argparse

from reinforce import NeuralPolicy, ValueNetwork


def compute_gae_buffer(rewards, values, dones, last_values, gamma=0.99, gae_lambda=0.95):
    """
    Vectorized GAE over a rollout buffer respecting episode boundaries.

    rewards, values, dones : [T, num_envs] tensors
    last_values            : [num_envs]  — bootstrap values at the end of rollout
    Returns returns, advantages : [T, num_envs]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        next_val = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return returns, advantages


def train_ppo(
    env_maker,
    n_iter=300,
    n_steps=512,
    num_envs=8,
    n_epochs=10,
    batch_size=64,
    alpha=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    beta_kl=0.0,
    max_grad_norm=0.5,
    use_advantage_norm=True,
    adv_eps=1e-8,
    hidden_size=128,
    device='cpu',
    seed=None,
    exp_dir=None,
    save_every_n=50,
):
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
            'policy_loss', 'value_loss', 'entropy', 'kl',
            'ep_return_std', 'ep_return_min', 'ep_return_max',
        ]
        with open(metrics_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    envs = gym.vector.SyncVectorEnv([env_maker for _ in range(num_envs)], copy=False)
    n_features = envs.single_observation_space.shape[0]
    n_actions  = envs.single_action_space.n

    policy    = NeuralPolicy(n_features, n_actions, hidden_size).to(device)
    value_net = ValueNetwork(n_features, hidden_size).to(device)
    optimizer = optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=alpha)

    mean_returns      = []
    best_return       = -np.inf
    best_policy_state = None

    obs, _ = envs.reset(seed=seed)
    obs = torch.FloatTensor(obs).to(device)

    pbar = tqdm(range(n_iter), desc="Train Iter...")

    for iteration in pbar:
        obs_buf    = []
        act_buf    = []
        rew_buf    = []
        done_buf   = []
        logp_buf   = []
        val_buf    = []
        logits_buf = []
        episode_returns = []
        ep_reward_accum = np.zeros(num_envs)

        for _ in range(n_steps):
            with torch.no_grad():
                dist      = policy.get_dist(obs)
                actions   = dist.sample()
                log_probs = dist.log_prob(actions)
                values    = value_net(obs)

            next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
            dones = terminated | truncated

            obs_buf.append(obs.clone())
            act_buf.append(actions)
            rew_buf.append(torch.FloatTensor(rewards).to(device))
            done_buf.append(torch.FloatTensor(dones.astype(float)).to(device))
            logp_buf.append(log_probs)
            val_buf.append(values)
            logits_buf.append(dist.logits)

            ep_reward_accum += rewards
            for i in range(num_envs):
                if dones[i]:
                    episode_returns.append(float(ep_reward_accum[i]))
                    ep_reward_accum[i] = 0.0

            obs = torch.FloatTensor(next_obs).to(device)

        with torch.no_grad():
            last_values = value_net(obs)

        obs_t    = torch.stack(obs_buf)    # [T, num_envs, n_features]
        act_t    = torch.stack(act_buf)    # [T, num_envs]
        rew_t    = torch.stack(rew_buf)    # [T, num_envs]
        done_t   = torch.stack(done_buf)   # [T, num_envs]
        logp_t   = torch.stack(logp_buf)   # [T, num_envs]
        val_t    = torch.stack(val_buf)    # [T, num_envs]
        logits_t = torch.stack(logits_buf) # [T, num_envs, n_actions]

        returns_t, adv_t = compute_gae_buffer(rew_t, val_t, done_t, last_values, gamma, gae_lambda)

        # Flatten to [T * num_envs]
        B = n_steps * num_envs
        obs_flat     = obs_t.view(B, n_features)
        act_flat     = act_t.view(B)
        logp_flat    = logp_t.view(B)
        returns_flat = returns_t.view(B)
        adv_flat     = adv_t.view(B)
        logits_flat  = logits_t.view(B, -1)

        if use_advantage_norm:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + adv_eps)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_kl          = 0.0
        n_updates = 0

        indices = np.arange(B)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, B, batch_size):
                mb_idx = indices[start:start + batch_size]
                mb_obs        = obs_flat[mb_idx]
                mb_act        = act_flat[mb_idx]
                mb_old_logp   = logp_flat[mb_idx]
                mb_returns    = returns_flat[mb_idx]
                mb_adv        = adv_flat[mb_idx]
                mb_old_logits = logits_flat[mb_idx]

                dist     = policy.get_dist(mb_obs)
                new_logp = dist.log_prob(mb_act)
                entropy  = dist.entropy().mean()

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value_net(mb_obs), mb_returns)

                if beta_kl > 0.0:
                    old_dist = Categorical(logits=mb_old_logits)
                    kl = kl_divergence(old_dist, dist).mean()
                else:
                    kl = torch.tensor(0.0, device=device)

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + beta_kl * kl
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_net.parameters()),
                    max_grad_norm,
                )
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                total_kl          += kl.item()
                n_updates += 1

        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss  = total_value_loss  / n_updates
        avg_entropy     = total_entropy     / n_updates
        avg_kl          = total_kl          / n_updates

        mean_ret = float(np.mean(episode_returns)) if episode_returns else float('nan')
        if mean_ret > best_return:
            best_return       = mean_ret
            best_policy_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        mean_returns.append(mean_ret)

        pbar.set_postfix({
            'mean_ret': f"{mean_ret:.1f}",
            'best':     f"{best_return:.1f}",
            'π_loss':   f"{avg_policy_loss:.3f}",
            'V_loss':   f"{avg_value_loss:.3f}",
            'kl':       f"{avg_kl:.4f}",
        })

        if exp_dir is not None:
            step_metrics = {
                'iteration':     iteration,
                'mean_return':   round(mean_ret, 4),
                'best_return':   round(best_return, 4),
                'policy_loss':   round(avg_policy_loss, 6),
                'value_loss':    round(avg_value_loss, 6),
                'entropy':       round(avg_entropy, 6),
                'kl':            round(avg_kl, 6),
                'ep_return_std': round(float(np.std(episode_returns)), 4) if len(episode_returns) > 1 else 0.0,
                'ep_return_min': round(float(np.min(episode_returns)), 4) if episode_returns else float('nan'),
                'ep_return_max': round(float(np.max(episode_returns)), 4) if episode_returns else float('nan'),
            }
            with open(metrics_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(step_metrics)

            if (iteration + 1) % save_every_n == 0 and best_policy_state is not None:
                ckpt = {
                    'iteration':  iteration,
                    'policy':     best_policy_state,
                    'value_net':  {k: v.cpu() for k, v in value_net.state_dict().items()},
                    'optimizer':  optimizer.state_dict(),
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
    parser.add_argument("--n-steps",           type=int,   default=512)
    parser.add_argument("--num-envs",          type=int,   default=8)
    parser.add_argument("--n-epochs",          type=int,   default=10)
    parser.add_argument("--batch-size",        type=int,   default=64)
    parser.add_argument("--alpha",             type=float, default=3e-4)
    parser.add_argument("--gamma",             type=float, default=0.99)
    parser.add_argument("--gae-lambda",        type=float, default=0.95)
    parser.add_argument("--clip-eps",          type=float, default=0.2)
    parser.add_argument("--entropy-coef",      type=float, default=0.01)
    parser.add_argument("--value-coef",        type=float, default=0.5)
    parser.add_argument("--beta-kl",           type=float, default=0.0)
    parser.add_argument("--max-grad-norm",     type=float, default=0.5)
    parser.add_argument("--no-advantage-norm", action="store_true")
    parser.add_argument("--hidden-size",       type=int,   default=128)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--exp-dir",           type=str,   default="runs/ppo_exp_001")
    parser.add_argument("--save-every-n",      type=int,   default=50)
    parser.add_argument("--device",            type=str,   default=None)

    parser.add_argument("--enable_wind",        action="store_true")
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
            enable_wind=args.enable_wind,
            wind_power=15.0,
            turbulence_power=1.5,
            render_mode=None,
        )

    policy, returns = train_ppo(
        env_maker=make_env,
        n_iter=args.n_iter,
        n_steps=args.n_steps,
        num_envs=args.num_envs,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        beta_kl=args.beta_kl,
        max_grad_norm=args.max_grad_norm,
        use_advantage_norm=not args.no_advantage_norm,
        hidden_size=args.hidden_size,
        device=device,
        seed=args.seed,
        exp_dir=args.exp_dir,
        save_every_n=args.save_every_n,
    )
