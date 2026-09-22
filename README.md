# BenchMARL Target Defense Environments

Multi-agent reinforcement learning environments for target defense scenarios using BenchMARL and VMAS.

## Overview

This repository contains custom VMAS environments for target defense scenarios where defenders must protect a target from attackers. The environments implement game-theoretic principles using Apollonius circles for optimal defense strategies.

## Environments

### 1. Target Defense Basic (`target_defense_basic.py`)
- Simple 1v1 or 3v1 defender-attacker scenario
- Defenders must detect attackers before they reach the target
- Uses sensing radius-based detection
- Apollonius optimization for reward calculation

### 2. Target Defense Smart (`target_defense_smart.py`)
- Enhanced version with smarter attacker behavior
- More sophisticated reward shaping
- Advanced spawning strategies

### 3. Target Defense Patrol (`target_defense_patrol.py`)
- Multi-agent patrol scenario
- Includes patrollers and pursuers with different roles
- Area-based spawning system

### 4. Target Defense Patrol End2End (`target_defense_patrol_End2End.py`)
- End-to-end learning version
- Integrated patrol and pursuit behaviors
- Heterogeneous agent speeds and capabilities

## Installation

Two scripts handle the whole setup: one installs the Python/conda stack, the
other deploys this repo's custom scenarios into it.

### 1. Clone this repository
```bash
git clone <this-repo-url> marl-pd
cd marl-pd
```

### 2. Install Miniconda and the `marl` conda environment
```bash
./setup_environment.sh          # CPU build of torch (default)
./setup_environment.sh cu121    # or pass a CUDA variant instead
```
This script is self-contained — it doesn't read anything from the repo. It
installs Miniconda if it isn't already present, creates the `marl` conda
environment (Python 3.10), and installs the pinned package versions this
project was developed against: torch, torchrl, tensordict, benchmarl, vmas,
cvxpy, hydra-core, pyglet, av (mp4 video export), matplotlib, and friends.

If this is the first time Miniconda was installed on this machine, open a new
terminal (or `source ~/miniconda3/etc/profile.d/conda.sh`) before the
`conda activate` step below.

### 3. Deploy the custom scenarios
```bash
conda activate marl
./deploy_scenarios.sh
```
A plain `pip install vmas benchmarl` gives you none of the environments in
this repo. VMAS loads scenarios by walking its own `vmas/scenarios/`
directory, and BenchMARL looks tasks up in a hardcoded enum. This script
copies `environments/*.py` and `yaml_configs/*.yaml` into the installed
packages, registers the `TARGET_DEFENSE_*` tasks in BenchMARL, and verifies
that every scenario loads.

Re-run `./deploy_scenarios.sh` any time you edit files under `environments/`
or `yaml_configs/` — the libraries execute their own copies under
site-packages, not the ones in this repo.

## Usage

### Basic Training Command

```bash
python -m benchmarl.run \
    task=vmas/target_defense_basic \
    algorithm=mappo \
    task.spawn_area_mode=true \
    task.spawn_area_width=0.1 \
    task.speed_ratio=0.3 \
    experiment.max_n_frames=1800000 \
    experiment.evaluation_interval=60000 \
    experiment.sampling_device=cuda \
    experiment.train_device=cuda \
    experiment.buffer_device=cuda \
    experiment.parallel_collection=true
```

### Environment-Specific Commands

#### Target Defense Basic (1v1)
```bash
python -m benchmarl.run \
    task=vmas/target_defense_basic \
    algorithm=mappo \
    task.num_defenders=1 \
    task.num_attackers=1 \
    task.speed_ratio=0.2 \
    task.sensing_radius=0.15 \
    experiment.max_n_frames=1000000
```

#### Target Defense Basic (3v1)
```bash
python -m benchmarl.run \
    task=vmas/target_defense_basic \
    algorithm=mappo \
    task.num_defenders=3 \
    task.num_attackers=1 \
    task.speed_ratio=0.3 \
    task.sensing_radius=0.15 \
    experiment.max_n_frames=1800000
```

#### Target Defense Smart
```bash
python -m benchmarl.run \
    task=vmas/target_defense_smart \
    algorithm=mappo \
    task.spawn_area_mode=true \
    task.spawn_area_width=0.1 \
    experiment.max_n_frames=2000000
```

#### Target Defense Patrol
```bash
python -m benchmarl.run \
    task=vmas/target_defense_patrol \
    algorithm=mappo \
    task.num_patrollers=1 \
    task.num_pursuers=2 \
    task.num_attackers=1 \
    task.patroller_sensing_radius=0.35 \
    experiment.max_n_frames=2000000
```

#### Target Defense Patrol End2End
```bash
python -m benchmarl.run \
    task=vmas/target_defense_patrol_end2end \
    algorithm=mappo \
    task.num_patrollers=1 \
    task.num_pursuers=2 \
    experiment.max_n_frames=2500000
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `task.num_defenders` | Number of defender agents | 3 |
| `task.num_attackers` | Number of attacker agents | 1 |
| `task.speed_ratio` | Attacker speed / Defender speed | 0.2 |
| `task.sensing_radius` | Detection radius for defenders | 0.15 |
| `task.spawn_area_mode` | Enable area-based spawning | false |
| `task.spawn_area_width` | Width of spawn area | 0.1 |
| `task.use_apollonius` | Use Apollonius optimization | true |
| `experiment.max_n_frames` | Total training frames | 1000000 |
| `experiment.evaluation_interval` | Evaluation frequency | 60000 |

## Apollonius Solver

The `apollonius_solver.py` module implements game-theoretic optimal defense using Apollonius circles:

```python
from apollonius_solver import solve_apollonius_optimization

result = solve_apollonius_optimization(
    attacker_pos=[0.5, 0.8],
    defender_positions=[[0.2, 0.2], [0.5, 0.1], [0.8, 0.2]],
    nu=5.0  # defender_speed / attacker_speed
)

print(f"Min Y-coordinate: {result['min_y_coordinate']}")
print(f"Defender payoff: {result['defender_payoff']}")
```

### Features:
- Computes Apollonius circle centers and radii
- Solves convex optimization problem for minimal attacker penetration
- Supports multiple defenders with heterogeneous speeds
- Numerical stability for small speed ratios

## Results

### Training Performance

See the `plots/` directory for visualization of training results:

- **3v1_comparison_20250927_122218.png**: Three defenders vs one attacker training curves
- **1v1_comparison_20250927_205404.png**: One defender vs one attacker comparison
- **evaluation_1000_episodes.png**: Evaluation results over 1000 episodes

### Key Findings

1. **3v1 Scenario**: Defenders achieve 85%+ detection rate after 1.8M frames
2. **1v1 Scenario**: Single defender achieves 60-70% detection rate
3. **Patrol Scenario**: Heterogeneous agents show emergent coordination

## Project Structure

```
benchmarl-target-defense/
├── environments/
│   ├── target_defense_basic.py
│   ├── target_defense_smart.py
│   ├── target_defense_patrol.py
│   └── target_defense_patrol_End2End.py
├── yaml_configs/
│   ├── target_defense_basic.yaml
│   ├── target_defense_smart.yaml
│   ├── target_defense_patrol.yaml
│   └── target_defense_patrol_End2End.yaml
├── plots/
│   ├── 3v1_comparison_20250927_122218.png
│   ├── 1v1_comparison_20250927_205404.png
│   └── evaluation_1000_episodes.png
├── apollonius_solver.py
└── README.md
```

## Advanced Usage

### Custom Training Script

```python
from benchmarl.experiment import Experiment
from benchmarl.algorithms import MappoConfig

# Create experiment
experiment = Experiment(
    task="vmas/target_defense_basic",
    algorithm=MappoConfig.get_from_yaml(),
    seed=0,
    config={
        "task": {
            "num_defenders": 3,
            "speed_ratio": 0.3,
            "sensing_radius": 0.15
        },
        "experiment": {
            "max_n_frames": 1800000,
            "evaluation_interval": 60000
        }
    }
)

# Run training
experiment.run()
```

### Evaluation Only

```bash
python -m benchmarl.run \
    task=vmas/target_defense_basic \
    algorithm=mappo \
    experiment.mode=evaluation \
    experiment.checkpoint_path=/path/to/checkpoint
```

## Citation

If you use these environments in your research, please cite:

```bibtex
@misc{benchmarl_target_defense,
  author = {Goutam Das},
  title = {BenchMARL Target Defense Environments},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/das-goutam/benchmarl-target-defense}
}
```

## References

- [BenchMARL](https://github.com/facebookresearch/BenchMARL): Multi-Agent Reinforcement Learning benchmark
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator): Vectorized Multi-Agent Simulator
- [TorchRL](https://github.com/pytorch/rl): PyTorch reinforcement learning library

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: [your-email@example.com]
- GitHub: [@das-goutam](https://github.com/das-goutam)

## Acknowledgments

This work builds upon the BenchMARL and VMAS frameworks and implements game-theoretic principles for multi-agent defense scenarios.
# marl-pd
