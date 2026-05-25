# Franka 迁移差异记录

仓库：`/home/geist/wbcd_workspace/lerobot/dual_arm_data_collection/lerobot_dual_arm_teleop`  
当前分支：`develop/dual_arm`  
待迁移分支：`dagger_test_v1-dual_franka`  
共同祖先：`27022aafdea2817e87d39f66109c7347e5bb3dab`  
工作区状态：干净，本次没有改动代码。

## 分支结论

`dagger_test_v1-dual_franka` 只有 3 个独有提交：

- `06fbc9b feat(franka): align the dual_franka to nero_compatible`
- `4b56bf9 feat(teleop): add mirror_teleop`
- `7b46e55 test(act): act on franka worked`

不要直接 merge 或 cherry-pick 整个分支。当前 `develop/dual_arm` 已经有新的 `policy_config_utils.py`、policy yaml、DAgger sampling、dataset tools、run_record/run_train CLI 等逻辑；旧 Franka 分支整体合入会回退这些内容。

重要约定：

- 不修改 `run_record.py` 和 `run_train.py` 的默认配置路径。
- Franka 备份配置文件保留，但使用当前分支命名风格：
  - `scripts/config/record_cfg_franka.yaml`
  - `scripts/config/train_cfg_franka.yaml`
  - `scripts/config/dagger_rounds_cfg_franka.yaml`
- 后续运行 Franka 配置时通过 `--config scripts/config/record_cfg_franka.yaml` 或 `--config scripts/config/train_cfg_franka.yaml` 显式指定。

## 需要迁移的内容

### 1. Robot 注册

文件：`robots/__init__.py`

需要把 `franka_dual_arm` 注册进 `ROBOT_CONFIG_REGISTRY`：

- import `FrankaDualArmConfig`
- import `FrankaDualArm`
- 增加 `"franka_dual_arm": (FrankaDualArmConfig, FrankaDualArm)`
- 更新 `__all__`

### 2. dual_franka package 导出修正

文件：`robots/dual_franka/__init__.py`

当前代码引用不存在的 `franka_interface_server`。Franka 分支改为延迟导入：

- `dual_franka_robotiq_rpc_server.DualFrankaRobotiqRpcApi`
- 如果 ROS2/server 依赖不可用，抛出更明确的 ImportError
- 移除 `FrankaDualArmServer` 的静态 `__all__`

### 3. Franka schema 兼容

文件：

- `robots/dual_franka/config_franka.py`
- `robots/dual_franka/franka_dual_arm.py`

需要迁移：

- 增加 `schema_mode`
  - `nero_compatible`
  - `franka_native`
- `nero_compatible` 下 action schema 对齐 Nero：
  - `left_delta_ee_pose.*`
  - `right_delta_ee_pose.*`
  - `left_gripper_cmd`
  - `right_gripper_cmd`
- `franka_native` 保留当前 Franka 原生字段：
  - `left_gripper_cmd_bin`
  - `right_gripper_cmd_bin`
  - `left/right_gripper_state_norm`
- `send_action()` 同时接受 `*_gripper_cmd` 和 `*_gripper_cmd_bin`
- observation/action features 根据 `schema_mode` 输出
- 保留当前 develop 已有的 clipping、cached RPC state、camera thread 等逻辑

需要确认一点：为了“不改变 develop 现有逻辑”，我建议默认值考虑保持 `franka_native`，然后在 `record_cfg_franka.yaml` 里显式设置 `schema_mode: nero_compatible`。Franka 分支原本默认是 `nero_compatible`。

### 4. run_record 兼容 Franka gripper key

文件：`scripts/core/run_record.py`

只迁移 Franka 相关增量，不替换当前文件整体逻辑：

- 增加 `GRIPPER_COMMAND_KEY_CANDIDATES`
- 增加 `resolve_gripper_command_keys()`
- 增加 `normalize_gripper_command_keys()`
- run_mix 中按 dataset/action schema 自动选择 gripper key
- policy action、expert action、sent action 都做 gripper alias normalization
- `RecordConfig` 读取 Franka 额外 robot 配置：
  - `schema_mode`
  - `rpc_timeout_sec`
  - `open_grippers_on_connect`
  - `reset_opens_grippers`
  - `reset_go_home`
  - `go_home_duration_sec`
  - `go_home_rate_hz`
  - `max_cartesian_delta`
  - `max_rotation_delta`
- 创建 robot config 时通过 `**robot_extra_config` 传入

必须保留当前 develop 的：

- `--config`
- `--dry-run-policy-config`
- `policy_config_utils`
- `sent_action_raw = robot.send_action(...)` 后再记录真实 sent action 的逻辑

### 5. Oculus mirror teleop

文件：

- `teleoperators/oculus_teleoperator/config_oculus_teleop.py`
- `teleoperators/oculus_teleoperator/oculus/oculus_dual_arm_robot.py`
- `teleoperators/oculus_teleoperator/oculus_teleop.py`

需要迁移：

- 新增 `mirror_teleop: bool = False`
- `OculusTeleop.connect()` 传入 `mirror_teleop`
- mirror 模式下：
  - 左右 controller 到左右 arm 的映射互换
  - pose delta 按 `[-1, -1, 1, -1, -1, 1]` 做镜像
  - trigger、gripper、release request、grip_pressed 同步互换

### 6. Franka 专用配置文件

新增文件，作为 Franka 配置备份保留，但需要按当前 develop 的新配置风格整理并采用新的文件名：

- `scripts/config/record_cfg_franka.yaml`
- `scripts/config/train_cfg_franka.yaml`
- `scripts/config/dagger_rounds_cfg_franka.yaml`

不要直接照搬旧分支内容。需要把它们改成当前 develop 风格：

- `record_cfg_franka.yaml` 使用当前 `policy.config_path`
- `train_cfg_franka.yaml` 使用当前 `scripts/policy_config/*_train_config.yaml`
- `dagger_rounds_cfg_franka.yaml` 保留当前 develop 的 `policy` descriptor 和 `dagger_training.sampling` 结构
- Franka 特有字段放在 `record.robot`
- `mirror_teleop: true` 放在 `teleop.oculus_config`
- 不把这些 Franka 备份配置设为任何脚本的默认配置；需要使用时通过 CLI `--config` 显式传入。

### 7. 可选迁移：ACT AMP 训练

文件：`scripts/core/run_train.py`

Franka 分支新增了：

- `_mixed_precision_for_policy()`
- `Accelerator(mixed_precision=...)`
- log `accelerator.mixed_precision`

这不是 Franka robot adapter 的必要代码，但属于 “ACT on Franka worked” 提交的一部分。建议作为单独步骤确认后再迁移，且保留当前 `policy_config_utils`、DAgger sampling、CLI 逻辑。

## 不应迁移的内容

不要迁移这些旧分支行为：

- 把 `run_record.py` 默认配置改成 `record_franka_cfg.yaml`
- 把 `run_train.py` 默认配置改成 `train_franka_cfg.yaml`
- 把 `reset_robot.py`、`run_replay.py`、`run_visualize.py` 默认配置改成 Franka 配置
- 新增旧分支命名的 `record_franka_cfg.yaml`、`train_franka_cfg.yaml`、`dagger_rounds_franka_cfg.yaml`；应改用 `record_cfg_franka.yaml`、`train_cfg_franka.yaml`、`dagger_rounds_cfg_franka.yaml`
- 删除或回退当前 develop 的：
  - `scripts/core/policy_config_utils.py`
  - `scripts/core/dagger_sampling.py`
  - `scripts/core/check_dagger_sampling.py`
  - `scripts/policy_config/*.yaml`
  - `scripts/tools/merge_lerobot_tasks.py`
  - `scripts/tools/patch_lerobot_dataset_metadata.py`
  - `tests/test_dagger_sampling.py`
