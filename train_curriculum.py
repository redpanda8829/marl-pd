"""
Curriculum training driver with partial weight transfer.

The five curriculum environments have different observation dimensions and
different defender counts, so a checkpoint from one stage cannot be restored
into the next with BenchMARL's normal `restore_file` (strict load_state_dict
fails on shape mismatch). This driver instead does a PARTIAL transfer:

  - tensors whose shape matches exactly            -> copied
  - tensors that differ only in the leading agent  -> broadcast / tiled / sliced
    dimension (shared <-> unshared params, or a       across the agent dim
    change in defender count)
  - tensors whose remaining dims differ (i.e. the  -> left at fresh init
    input layer, because obs dim changed)

So the learned hidden representation carries across stages while only the
input projection is relearned.

Usage:
  python train_curriculum.py --task TARGET_DEFENSE_SMART --tag stage2_smart \
      --parent outputs/curriculum/stage1_basic/.../checkpoint_1000000.pt \
      --frames 1000000 --entropy 0.01 --share-params true
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torchrl.record.loggers.csv import CSVLogger

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

VIDEO_FPS = 5


def _fit_agent_dim(parent_t: torch.Tensor, target: torch.Tensor):
    """
    Try to reconcile a parent tensor with the target shape when they differ
    only in the leading agent dimension.

    Handles:
      (out,in)    -> (A,out,in)   broadcast one shared net into every agent slot
      (A,out,in)  -> (out,in)     collapse per-agent nets into one (mean)
      (Ap,out,in) -> (Ac,out,in)  tile/slice to fit the new agent count

    Returns the fitted tensor, or None if the shapes are not reconcilable.
    """
    p_shape, t_shape = tuple(parent_t.shape), tuple(target.shape)

    # shared -> unshared: (out,in) into (A,out,in)
    if len(t_shape) == len(p_shape) + 1 and p_shape == t_shape[1:]:
        return parent_t.unsqueeze(0).expand(t_shape).clone()

    # unshared -> shared: (A,out,in) into (out,in); average the agent nets
    if len(p_shape) == len(t_shape) + 1 and t_shape == p_shape[1:]:
        return parent_t.mean(dim=0).clone()

    # agent count changed, trailing dims identical: (Ap,...) -> (Ac,...)
    if len(p_shape) == len(t_shape) and len(p_shape) >= 2 and p_shape[1:] == t_shape[1:]:
        ap, ac = p_shape[0], t_shape[0]
        if ac <= ap:
            return parent_t[:ac].clone()
        reps = [1] * len(p_shape)
        reps[0] = (ac + ap - 1) // ap
        return parent_t.repeat(*reps)[:ac].clone()

    return None


def partial_transfer(experiment: Experiment, parent_ckpt: str):
    """Copy what fits from a parent checkpoint into a freshly built experiment."""
    parent = torch.load(parent_ckpt, map_location="cpu")
    report = {}

    for group in experiment.group_map.keys():
        key = f"loss_{group}"
        if key not in parent:
            report[group] = {"status": "group not in parent checkpoint"}
            continue

        parent_sd = parent[key]
        loss_module = experiment.losses[group]
        current_sd = loss_module.state_dict()

        merged, copied, fitted, fresh, meta = {}, [], [], [], []
        for k, cur in current_sd.items():
            if k not in parent_sd:
                merged[k] = cur
                fresh.append(k)
                continue
            p = parent_sd[k]
            # TensorDictParams state_dicts carry non-tensor metadata entries
            # (__batch_size, __device); keep the fresh experiment's own values.
            if not isinstance(p, torch.Tensor) or not isinstance(cur, torch.Tensor):
                merged[k] = cur
                meta.append(k)
                continue
            if tuple(p.shape) == tuple(cur.shape):
                merged[k] = p
                copied.append(k)
                continue
            adapted = _fit_agent_dim(p, cur)
            if adapted is not None:
                merged[k] = adapted
                fitted.append(f"{k} {tuple(p.shape)}->{tuple(cur.shape)}")
            else:
                merged[k] = cur
                fresh.append(f"{k} {tuple(p.shape)}!={tuple(cur.shape)}")

        loss_module.load_state_dict(merged, strict=True)
        report[group] = {
            "copied": len(copied),
            "agent_dim_fitted": len(fitted),
            "reinitialized": len(fresh),
            "non_tensor_metadata_skipped": len(meta),
            "fitted_detail": fitted,
            "reinit_detail": fresh,
        }

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="VmasTask member name, e.g. TARGET_DEFENSE_SMART")
    ap.add_argument("--tag", required=True, help="short name for this stage's output folder")
    ap.add_argument("--parent", default=None, help="parent checkpoint .pt to partially transfer from")
    ap.add_argument("--frames", type=int, default=1_000_000)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--share-params", default="true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="outputs/curriculum")
    ap.add_argument("--batch", type=int, default=None, help="on_policy_collected_frames_per_batch")
    ap.add_argument("--envs", type=int, default=None, help="on_policy_n_envs_per_worker")
    ap.add_argument("--eval-interval", type=int, default=None)
    args = ap.parse_args()

    share_params = str(args.share_params).lower() in ("true", "1", "yes")

    save_folder = Path(args.out_root) / args.tag
    save_folder.mkdir(parents=True, exist_ok=True)

    task = VmasTask[args.task].get_from_yaml()

    algorithm_config = MappoConfig.get_from_yaml()
    algorithm_config.entropy_coef = args.entropy

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = args.frames
    experiment_config.loggers = ["csv"]
    experiment_config.render = True
    experiment_config.evaluation = True
    experiment_config.checkpoint_at_end = True
    experiment_config.keep_checkpoints_num = 3
    experiment_config.share_policy_params = share_params
    experiment_config.save_folder = str(save_folder.resolve())
    if args.batch is not None:
        experiment_config.on_policy_collected_frames_per_batch = args.batch
    if args.envs is not None:
        experiment_config.on_policy_n_envs_per_worker = args.envs
    if args.eval_interval is not None:
        experiment_config.evaluation_interval = args.eval_interval

    # BenchMARL requires checkpoint/evaluation intervals to be exact multiples
    # of the collected frames per batch, so derive them from the batch size
    # rather than hardcoding.
    batch = experiment_config.on_policy_collected_frames_per_batch
    experiment_config.checkpoint_interval = max(1, round(50_000 / batch)) * batch
    if args.eval_interval is None:
        experiment_config.evaluation_interval = max(1, round(100_000 / batch)) * batch
    else:
        experiment_config.evaluation_interval = max(1, round(args.eval_interval / batch)) * batch

    print(f"\n=== curriculum stage: {args.tag} ===")
    print(f"  task           : {args.task}")
    print(f"  frames         : {args.frames:,}")
    print(f"  entropy_coef   : {args.entropy}")
    print(f"  share_params   : {share_params}")
    print(f"  parent ckpt    : {args.parent or '(none - training from scratch)'}")

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=experiment_config,
    )

    if args.parent:
        report = partial_transfer(experiment, args.parent)
        print("\n  --- partial weight transfer ---")
        for group, r in report.items():
            if "status" in r:
                print(f"    {group}: {r['status']}")
                continue
            print(f"    {group}: {r['copied']} copied, "
                  f"{r['agent_dim_fitted']} agent-dim fitted, {r['reinitialized']} reinitialized")
            for d in r["fitted_detail"]:
                print(f"       fitted : {d}")
            for d in r["reinit_detail"]:
                print(f"       fresh  : {d}")
        with open(save_folder / "transfer_report.json", "w") as f:
            json.dump(report, f, indent=2)

    for logger in experiment.logger.loggers:
        if isinstance(logger, CSVLogger):
            logger.experiment.video_format = "mp4"
            logger.experiment.video_fps = VIDEO_FPS

    experiment.run()

    ckpt_dir = Path(experiment.folder_name) / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    final = ckpts[-1] if ckpts else None
    print(f"\n=== stage {args.tag} complete ===")
    print(f"  experiment folder: {experiment.folder_name}")
    print(f"  final checkpoint : {final}")
    with open(save_folder / "stage_result.json", "w") as f:
        json.dump({"tag": args.tag, "task": args.task, "frames": args.frames,
                   "entropy": args.entropy, "share_params": share_params,
                   "folder": str(experiment.folder_name),
                   "final_checkpoint": str(final) if final else None}, f, indent=2)


if __name__ == "__main__":
    main()
