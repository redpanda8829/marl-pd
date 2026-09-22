"""
Live-view a policy in an on-screen VMAS window while training runs in a
separate process.

Usage:
    python live_view_training.py <checkpoint_dir> [--poll-interval SECONDS]

<checkpoint_dir> is the "checkpoints" folder of an in-progress (or finished)
BenchMARL run, e.g.:
    outputs/2026-09-01/12-00-00/mappo_target_defense_smart_w_obs_mlp__.../checkpoints

Requires the training run to have been started with
experiment.checkpoint_interval > 0 (e.g. equal to
experiment.on_policy_collected_frames_per_batch, so a checkpoint is written
every training iteration) so new checkpoints keep appearing while this script
polls for them. Requires a display (X11/Wayland) on this machine.

Loads the latest available checkpoint's weights and replays one evaluation
episode live, on repeat, picking up newer weights whenever a newer checkpoint
file appears between episodes.
"""
import argparse
import glob
import os
import re
import time

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.hydra_config import reload_experiment_from_file

# Work around a pyglet 1.5.x / XWayland incompatibility where an event type
# XCheckWindowEvent doesn't expect raises `ctypes.ArgumentError` on the very
# first dispatch_events() call, before the window ever gets to draw+flip.
# Swallowing it here just skips input/window-event processing for that call;
# the actual rendering (the part we care about for a passive viewer) still runs.
try:
    from pyglet.window.xlib import XlibWindow

    _orig_dispatch_events = XlibWindow.dispatch_events

    def _safe_dispatch_events(self):
        try:
            _orig_dispatch_events(self)
        except Exception:
            pass

    XlibWindow.dispatch_events = _safe_dispatch_events
except ImportError:
    pass


def find_latest_checkpoint(checkpoint_dir):
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not files:
        return None

    def frame_num(path):
        match = re.search(r"checkpoint_(\d+)\.pt$", path)
        return int(match.group(1)) if match else -1

    return max(files, key=frame_num)


def render_callback(env, td):
    env.render(mode="human")


def run_live_episode(experiment):
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        experiment.test_env.rollout(
            max_steps=experiment.max_steps,
            policy=experiment.policy,
            callback=render_callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", help="Path to a run's checkpoints/ folder")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between checking for a new checkpoint",
    )
    args = parser.parse_args()

    experiment = None
    loaded_checkpoint = None

    print(f"Watching {args.checkpoint_dir} for checkpoints...")
    while True:
        latest = find_latest_checkpoint(args.checkpoint_dir)

        if latest is None:
            print("No checkpoint yet, waiting...")
            time.sleep(args.poll_interval)
            continue

        if latest != loaded_checkpoint:
            if experiment is None:
                print(f"Loading experiment from {latest}")
                experiment = reload_experiment_from_file(latest)
            else:
                print(f"Loading new weights from {os.path.basename(latest)}")
                state_dict = torch.load(
                    latest, map_location=experiment.config.restore_map_location
                )
                experiment.load_state_dict(state_dict)
            loaded_checkpoint = latest

        print(f"Rendering live rollout (weights from {os.path.basename(loaded_checkpoint)})...")
        run_live_episode(experiment)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
