import sys
from torchrl.record.loggers.csv import CSVLogger
from benchmarl.hydra_config import reload_experiment_from_file

checkpoint_path = sys.argv[1]
experiment = reload_experiment_from_file(checkpoint_path)
experiment.config.render = True

for logger in experiment.logger.loggers:
    if isinstance(logger, CSVLogger):
        logger.experiment.video_format = "mp4"
        logger.experiment.video_fps = 5

experiment.evaluate()

for logger in experiment.logger.loggers:
    if isinstance(logger, CSVLogger):
        print(f"Video saved under: {logger.experiment.log_dir}/videos/")
