"""
Aggregate curriculum results into one JSON summary.

Two sources are combined:

1. Training curves, read from each stage's CSV scalar logs (reward over time).

2. Per-defender role metrics, which the CSV logs CANNOT give us: BenchMARL
   writes one scalar per info key, averaged over all agents, so
   "which defender captured the attacker" is collapsed away. To recover it we
   rebuild each stage's final policy and run a fresh batch of deterministic
   evaluation episodes, reading the per-defender tensors out of the info dict.

Usage:
  python analyze_curriculum.py --out outputs/curriculum/summary.json
"""
import argparse
import json
from pathlib import Path

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

CURRICULUM_ROOT = Path("outputs/curriculum")


def read_curve(scalars_dir: Path, name: str):
    f = scalars_dir / name
    if not f.exists():
        return []
    out = []
    for line in f.read_text().strip().splitlines():
        if not line.strip():
            continue
        a, b = line.split(",")
        out.append([int(float(a)), float(b)])
    return out


def find_scalars_dir(stage_folder: Path):
    hits = list(stage_folder.glob("*/*/scalars"))
    return hits[0] if hits else None


def rebuild_and_load(task_name: str, share_params: bool, entropy: float, ckpt: str):
    """Rebuild a stage's experiment exactly as trained, then load its weights."""
    task = VmasTask[task_name].get_from_yaml()
    algo = MappoConfig.get_from_yaml()
    algo.entropy_coef = entropy

    ec = ExperimentConfig.get_from_yaml()
    ec.loggers = []
    ec.render = False
    ec.evaluation = True
    ec.create_json = False
    ec.checkpoint_interval = 0
    ec.max_n_frames = ec.on_policy_collected_frames_per_batch
    ec.share_policy_params = share_params
    ec.save_folder = str(Path("/tmp").resolve())

    exp = Experiment(task=task, algorithm_config=algo,
                     model_config=MlpConfig.get_from_yaml(),
                     critic_model_config=MlpConfig.get_from_yaml(),
                     seed=0, config=ec)

    state = torch.load(ckpt, map_location="cpu")
    for group in exp.group_map.keys():
        key = f"loss_{group}"
        if key in state:
            exp.losses[group].load_state_dict(state[key], strict=True)
    return exp


def role_metrics(exp: Experiment, episodes_note: str = ""):
    """Run one deterministic evaluation batch and pull per-defender stats."""
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        roll = exp.test_env.rollout(
            max_steps=exp.max_steps, policy=exp.policy,
            auto_cast_to_device=True, break_when_any_done=False)

    # IMPORTANT: read the info out of `next`, not the root. With auto-reset,
    # the root info at a terminal step has already been cleared by the reset,
    # so terminal flags (captured / reached_target) are invisible there and
    # every episode looks like it ended for no reason. `next` holds the true
    # terminal state.
    nxt = roll.get("next")
    info = nxt.get(("defender", "info"))
    done = nxt.get("done").squeeze(-1).bool()
    n_env = roll.batch_size[0]

    keys = set(info.keys())

    def ever(key):
        # (batch, time, n_agent_copies, n_defenders) -> (batch, n_defenders)
        return info[key][:, :, 0, :].any(dim=1)

    def any_flag(key):
        # target_defense_basic has no capture concept at all, so its info dict
        # genuinely lacks the capture keys - report zeros rather than crashing.
        if key not in keys:
            return torch.zeros(n_env, dtype=torch.bool)
        return info[key].bool().any(dim=1).squeeze(-1).any(dim=-1)

    # Episode-level outcomes, counted at the steps where `done` actually fires.
    def at_done(key):
        if key not in keys:
            return 0
        flag = info[key][:, :, 0, :].bool().any(dim=-1)
        return int((flag & done).sum())

    n_episodes = int(done.sum())
    n_captured = at_done("attackers_captured")
    n_reached = at_done("attackers_reached_target")

    sensed_any = any_flag("attackers_sensed")

    # Per-defender attribution, also taken at terminal steps.
    def per_def_at_done(key):
        if key not in keys:
            return [0] * len(info["defender_distance_traveled"][0, 0, 0])
        f = info[key][:, :, 0, :].bool()               # (batch, time, n_def)
        return (f & done.unsqueeze(-1)).sum(dim=(0, 1)).tolist()

    def_sensed = per_def_at_done("defender_sensed")
    def_captured = per_def_at_done("defender_captured")
    # peak cumulative distance within an episode (resets to 0 each episode)
    dist_final = info["defender_distance_traveled"][:, :, 0, :].max(dim=1).values

    defenders = [a for a in exp.test_env.base_env.world.agents if a.is_defender] \
        if hasattr(exp.test_env, "base_env") else []
    speeds = [float(a.max_speed) for a in defenders]
    sensing = [float(a.sensing_radius) for a in defenders]
    capture_r = [float(getattr(a, "capture_distance", float("nan"))) for a in defenders]

    return {
        "env_slots": int(n_env),
        "episodes": n_episodes,
        "episodes_captured": n_captured,
        "episodes_target_reached": n_reached,
        "capture_rate": (n_captured / n_episodes) if n_episodes else None,
        "env_slots_sensed": int(sensed_any.sum()),
        "per_defender_sensed_count": def_sensed,
        "per_defender_captured_count": def_captured,
        "per_defender_mean_peak_distance": dist_final.mean(dim=0).tolist(),
        "per_defender_max_speed": speeds,
        "per_defender_sensing_radius": sensing,
        "per_defender_capture_distance": capture_r,
        "note": episodes_note,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/curriculum/summary.json")
    ap.add_argument("--skip-eval", action="store_true",
                    help="only read training curves, don't roll out checkpoints")
    args = ap.parse_args()

    summary = {}
    for res_file in sorted(CURRICULUM_ROOT.glob("*/stage_result.json")):
        meta = json.loads(res_file.read_text())
        tag = meta["tag"]
        stage_folder = res_file.parent
        entry = {"meta": meta}

        sdir = find_scalars_dir(stage_folder)
        if sdir:
            entry["curves"] = {
                "train_reward": read_curve(sdir, "collection_reward_episode_reward_mean.csv"),
                "eval_reward": read_curve(sdir, "eval_reward_episode_reward_mean.csv"),
                "train_entropy": read_curve(sdir, "train_defender_entropy.csv"),
                "eval_episode_len": read_curve(sdir, "eval_reward_episode_len_mean.csv"),
                "captured_rate": read_curve(sdir, "collection_defender_info_attackers_captured.csv"),
                "sensed_rate": read_curve(sdir, "collection_defender_info_attackers_sensed.csv"),
            }

        ckpt = meta.get("final_checkpoint")
        if ckpt and Path(ckpt).exists() and not args.skip_eval:
            try:
                exp = rebuild_and_load(meta["task"], meta["share_params"],
                                       meta["entropy"], ckpt)
                entry["role_metrics"] = role_metrics(exp)
                print(f"  evaluated {tag}: "
                      f"{entry['role_metrics']['episodes_captured']}/"
                      f"{entry['role_metrics']['episodes']} captured")
            except Exception as e:
                entry["role_metrics_error"] = f"{type(e).__name__}: {e}"
                print(f"  eval FAILED {tag}: {e}")

        summary[tag] = entry
        print(f"collected {tag}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
