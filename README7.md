# Single-Arm ACT on UR Robot Demonstrations

Adapted the original ALOHA ACT (bimanual) training pipeline to support **single-arm UR robot demonstrations** collected in the real world. Replaced hardcoded bimanual assumptions with configurable dimensions (`state_dim=32`, `action_dim=7`), enabling ACT training on custom UR data.

Built a **data conversion pipeline** to transform per-episode `trajectory.pkl` files (multi-view RGB + state/action trajectories) into ACT-compatible HDF5 episodes. Added custom task configuration support for UR single-arm training and verified end-to-end training startup with ACT.

Also explored **validation sampling strategies** (random vs deterministic timestep selection) and showed that deterministic validation improves metric stability and experiment comparability while preserving random sampling for training diversity.

## Key Contributions
- ALOHA bimanual → UR single-arm ACT adaptation
- PKL → HDF5 dataset pipeline for ACT dataloader
- Custom UR task training support (`sim_tomato_soup_can`)
- Validation random vs deterministic sampling comparison