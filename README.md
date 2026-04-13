# JPack:

JPack is a **Jittor-based reconstruction project** for Online 3D Bin Packing (Online 3D-BPP). Built on top of our team's previously released **RoboBPP** benchmark, this repository ports several representative Torch-based methods to **Jittor** and organizes them into a reusable codebase for training, testing, and evaluation.

The repository currently contains four core subprojects:

- `AR2L`: Adjustable Robust Reinforcement Learning
- `PCT`: Packing Configuration Tree
- `PCT-full`: an extended PCT framework with additional packing models and evaluation utilities
- `TAPNet++`: an end-to-end framework from perception to packing decision

Benefiting from Jittor's meta-operator system and kernel fusion capabilities, JPack provides a lightweight and efficient reconstruction for industrial online packing research. According to our project note, the reconstructed models achieve an average **10%** improvement in forward inference speed, making them better suited for real-time decision-making in dynamic production environments.

## Overview

Online 3D bin packing requires an algorithm to decide the placement position and pose of each incoming item **without knowing the future item sequence**. In real industrial scenarios, the problem is not only about maximizing space utilization, but also about balancing:

- stacking stability
- physical executability
- robotic operation safety

Many previous studies rely on idealized geometric assumptions. In contrast, **RoboBPP** pushes the problem toward a more realistic benchmark setting with physical constraints and robotic execution constraints. JPack further extends this effort by providing **Jittor reconstructions** of representative learning-based methods in the RoboBPP benchmark, supporting:

- unified algorithm comparison and reproducible experiments
- training and testing on real industrial data
- progressive evaluation from pure geometric packing to physics simulation and execution-aware testing

## Benchmark Setting

### Real-World Industrial Data

JPack follows RoboBPP and targets three representative industrial data scenarios:

- `Repetitive Dataset`: assembly-line style scenarios with highly repetitive item patterns
- `Diverse Dataset`: warehouse and e-commerce style scenarios with highly heterogeneous item sizes
- `Wood Board Dataset`: long-board and panel scenarios that require stronger stability and pose control

In the current codebase and example commands, common dataset paths include:

- `data/time_series/pg.xlsx`
- `data/occupancy/deli.xlsx`
- `data/flat_long/opai.txt`

### Progressive Evaluation

JPack inherits the staged evaluation design of RoboBPP and covers three levels of testing:

- `Math Pack`: upper-bound performance under ideal geometric assumptions
- `Physics Pack`: stability evaluation with gravity, friction, and other physical constraints
- `Execution Pack`: evaluation with additional robotic grasping and execution constraints

This evaluation protocol is closer to real industrial deployment and highlights the gap between geometric feasibility and physical executability.

## Repository Structure

```text
JPack/
├── AR2L/        # Jittor reconstruction of adjustable robust RL
├── PCT/         # Jittor reconstruction of Packing Configuration Tree
├── PCT-full/    # Extended PCT framework with more model variants and evaluation scripts
├── TAPNet++/    # Jittor implementation of TAPNet++
├── Jpack.tex    # Project note / communication draft
└── README.md
```

The four subdirectories serve the following purposes:

- `AR2L`: robust online packing policy learning under challenging or adversarial item sequences
- `PCT`: hierarchical packing state representation and reinforcement learning based on Packing Configuration Trees
- `PCT-full`: a broader experimental framework with more model architectures and benchmark-oriented evaluation entry points
- `TAPNet++`: joint packing decision optimization with candidate space representation and policy learning

## Installation

We recommend using **Python 3.8**. Since the four subprojects do not share exactly the same dependency set, it is recommended to create separate environments for different methods when necessary, or install dependencies on demand.

```bash
conda create -n jpack python=3.8
conda activate jpack
pip install jittor==1.3.10.0
```

Then install the dependencies required by the target subproject:

- `AR2L/requirements.txt`
- `PCT/requirements.txt`
- `PCT-full/requirements.txt`
- `TAPNet++/requirements.txt`

Notes:

- `AR2L`, `PCT`, and `PCT-full` mainly depend on the `gym` 0.x ecosystem
- `TAPNet++` uses `gymnasium` and `tianshou`
- `PCT-full` additionally depends on physics-related packages such as `pybullet` and `trimesh`

## Quick Start

Run the following commands inside the corresponding subdirectory. The example arguments are adapted from the existing `demo.sh` files or experiment scripts. Please replace the dataset path with the valid JPack / RoboBPP data files on your machine.

### 1. PCT

```bash
cd PCT
python main.py --setting 1 --internal-node-holder 80 --leaf-node-holder 50 --load-dataset --dataset-path data/time_series/pg.xlsx --custom time_series --container-size 134 125 100
```

### 2. AR2L

```bash
cd AR2L
python main.py --num-box 80 --num-next-box 20 --num-candidate-action 120 --alpha 1 --dataset-path data/time_series/pg.xlsx --custom time_series
```

### 3. PCT-full

```bash
cd PCT-full
python main.py --setting 2 --custom discrete_real_opai_setting_2 --preview 1 --select 1 --internal_node_holder 150 --leaf_node_holder 150 --env_version 3 --device 0 --item_size_set discrete --container_size 250,120,100 --training_without_evaluate --distribution real_opai
```

`PCT-full` also provides:

- batch experiment examples: `run_ijrr.py`
- offline evaluation entry: `evaluation.py`
- multiple model variants: `PCT`, `CDRL`, `PackE`, `Attend2Pack`, `RCQL`, and `Rainbow`

### 4. TAPNet++

```bash
cd TAPNet++
python tap_train.py --box-num 30 --box-range 10 80 --container-size 250 120 100 --train 1 --test-num 1 --model tnpp --prec-type none --fact-type box --data-type rand --ems-type ems-id-stair --stable-rule none --rotate-axes z --hidden-dim 128 --world-type ideal --container-type single --pack-type last --stable-predict 0 --note 4corner --corner-num 1 --max-epoch 300 --reward-type H --dataset-path data/flat_long/opai.txt --custom flat_long
```

For testing, change `--train 1` to `--train 0`.

## Included Methods

This repository reconstructs and organizes several representative methods used in the RoboBPP benchmark:

- **PCT**: represents packing states as Packing Configuration Trees and reduces continuous placement difficulty through candidate-node selection
- **TAPNet++**: jointly optimizes item order, orientation, and placement position in a unified decision step
- **AR2L**: explicitly models the trade-off between average-case performance and worst-case robustness
- **PCT-full**: extends the unified framework with more model architectures and benchmark-oriented evaluation utilities

## Use Cases

JPack is suitable for:

- reproducing Online 3D Bin Packing reinforcement learning methods with Jittor
- benchmarking algorithms under a RoboBPP-style evaluation protocol
- studying the transition from ideal geometric packing to physics-based and execution-aware evaluation
- embodied intelligence research for industrial logistics, warehousing, and board-manufacturing scenarios

## Citation

If you use JPack in your research, please cite both the RoboBPP benchmark and the corresponding algorithm papers:

```bibtex
@inproceedings{zhao2021learning,
  title={Learning Efficient Online 3D Bin Packing on Packing Configuration Trees},
  author={Zhao, Hang and Yu, Yang and Xu, Kai},
  booktitle={International Conference on Learning Representations},
  year={2021},
}

@article{xu2023neural,
  title={Neural Packing: From Visual Sensing to Reinforcement Learning},
  author={Xu, Juzhan and Gong, Minglun and Zhang, Hao and Huang, Hui and Hu, Ruizhen},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={6},
  pages={1--11},
  year={2023},
  publisher={ACM New York, NY, USA}
}

@article{pan2023adjustable,
  title={Adjustable robust reinforcement learning for online 3d bin packing},
  author={Pan, Yuxin and Chen, Yize and Lin, Fangzhen},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={51926--51954},
  year={2023}
}

@article{wang2025robobpp,
  title={RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation},
  author={Wang, Zhoufeng and Zhao, Hang and Xu, Juzhan and Zhang, Shishun and Xiong, Zeyu and Hu, Ruizhen and Zhu, Chenyang and Zeng, Zecui and Xu, Kai},
  journal={arXiv preprint arXiv:2512.04415},
  year={2025}
}
```

## Acknowledgements

This project is built on top of our team's previously proposed **RoboBPP** benchmark and reconstructs the core implementations of several representative online packing methods. It is contributed by members of a joint team from:

- Institute of AI For Industries, Chinese Academy of Sciences
- Shenzhen University
- National University of Defense Technology
- Wuhan University
- China Post Technology
- JD Technology
- SF Technology

We would also like to thank:

- the authors of `RoboBPP`, `PCT`, `AR2L`, `TAPNet++`, and related works
- the `Jittor` framework for supporting efficient reconstruction and deployment
