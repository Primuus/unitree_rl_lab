# 工程工作规范

当前工程位于服务器 Docker 容器 `isaac-record` 的 `/workspace/ry_unitree_rl_lab`；本机 Git 副本为 `/home/ry/workspace/ry_unitree_rl_lab`。默认在容器工程修改和验证，再同步本机。

- 用户最新要求：每次完成修改及必要验证后，仅提交本地 Git；未经用户后续明确要求，不推送 GitHub。服务器继续同步训练与评估所需代码。结束前核对本机与容器代码一致，并说明本地提交及 GitHub 未推送状态；遇到冲突不覆盖他人改动。
- 后续直接覆盖当前实现，不建立并存的版本目录或旧代码副本。训练模型、日志和实验记录仍应保留，不作为旧代码副本清理。
- 现有 `ry_go2_biped_walk_v20/`、`scripts/train_v20.py`、`scripts/eval_v20.py` 和任务 ID 保留为稳定入口；当前内容为 12 关节实现。
- 奖励公式、权重与主要控制/PPO 参数优先复用用户指定的 `yusongmin1/My_unitree_go2_gym/go2_handstand`。不自行设计替代奖励或擅自调权重；接口适配及明确修正记录于 `doc/reference_port_manifest.json`。
- 运行使用 `./ry_python.sh`，仅保留两个阶段：`balance` 三轴零命令、21 项奖励；`walk` 固定前进 `0.2 m/s`、横向和转向为零、23 项奖励。当前 12 动作权重通过 `--resume` 恢复，不使用旧 6 动作初始化权重。
- 外部工程依赖为 Isaac Lab 和 Unitree ROS。缺少资源先查已有工程，不生成替代机器人资源，不恢复对原 `unitree_rl_lab` 的运行依赖。
- 用户已授权本机 MuJoCo 部署：代码在本仓库 `deploy/mujoco/`，本机 `/home/ry/workspace/ry_unitree_rl_lab_mujoco` 链接到该目录，使用 `RL` conda 环境。复用 quadruped 的现有视觉资源；训练原始 URDF 用于机械参数适配，不修改 quadruped 源工程。策略和实验输出不提交。
- 未经明确授权不修改服务器硬件、驱动、系统 CUDA、内核或其他底层配置。
- 文档统一存放于工程 `doc/`；训练产物和生成模型不提交 Git。正式运行不得依赖临时测试产物；仅清理本轮创建且无需保留的临时文件。
- 用户已授权策略文件直接传输，无需逐次审核；按用户请求下载或同步策略时直接执行，并说明用途、方向和大小。其他文件继续遵守下述传输规则。
- 使用 `scp` 前，无论大小都须说明用途、方向和预计大小，由用户决定；其他大文件传输同样事先确认。代码同步优先使用 Git，不搬运模型、数据集或环境归档。

当前参数、命令与验证记录见 `doc/仓库使用说明.md` 和 `doc/V21_参考My_unitree_go2_gym的后腿步态优化方案.md`。上级工作区规范同样适用；用户最新指令优先。
