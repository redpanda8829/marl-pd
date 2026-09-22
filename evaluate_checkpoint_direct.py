import sys
from torchrl.record.loggers.csv import CSVLogger
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

# Use when a checkpoint's .hydra/config.yaml is missing or corrupted
# (e.g. two concurrent runs writing to the same output folder).
# Reconstructs the experiment from each task's plain defaults instead.
#
# Usage: python evaluate_checkpoint_direct.py <task_name> <checkpoint_path>
# task_name is a VmasTask member name, e.g. TARGET_DEFENSE_PATROL_END2END or TARGET_DEFENSE_PATROL_MA

task_name = sys.argv[1]
checkpoint_path = sys.argv[2]

task = VmasTask[task_name].get_from_yaml()
algorithm_config = MappoConfig.get_from_yaml()
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()
experiment_config = ExperimentConfig.get_from_yaml()

experiment_config.restore_file = checkpoint_path
experiment_config.render = True
experiment_config.loggers = ["csv"]

experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)

for logger in experiment.logger.loggers:
    if isinstance(logger, CSVLogger):
        logger.experiment.video_format = "mp4"
        logger.experiment.video_fps = 5

experiment.evaluate()

for logger in experiment.logger.loggers:
    if isinstance(logger, CSVLogger):
        print(f"Video saved under: {logger.experiment.log_dir}/videos/")
