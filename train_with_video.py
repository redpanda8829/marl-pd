"""
Run BenchMARL training with mp4 video export enabled during periodic evaluation.

BenchMARL's CSV logger only writes raw .pt video tensors by default, so this
wrapper patches it to write playable .mp4 files (like evaluate_checkpoint.py
does for post-hoc evaluation). CLI usage mirrors `python -m benchmarl.run`:

    python train_with_video.py task=vmas/target_defense_smart_w_obs algorithm=mappo \
        experiment.loggers=[csv] experiment.render=true \
        experiment.max_n_frames=2000000

Videos are written every experiment.evaluation_interval frames under
<log_dir>/videos/.
"""
import os

import benchmarl
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchrl.record.loggers.csv import CSVLogger

from benchmarl.hydra_config import load_experiment_from_hydra

VIDEO_FPS = 5
BENCHMARL_CONF_PATH = os.path.join(os.path.dirname(benchmarl.__file__), "conf")


@hydra.main(version_base=None, config_path=BENCHMARL_CONF_PATH, config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment = load_experiment_from_hydra(cfg, task_name=task_name)

    for logger in experiment.logger.loggers:
        if isinstance(logger, CSVLogger):
            logger.experiment.video_format = "mp4"
            logger.experiment.video_fps = VIDEO_FPS

    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
