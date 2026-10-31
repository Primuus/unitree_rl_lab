# 工程工作规范

当前工程位于服务器 Docker 容器 `isaac-record` 的 `/workspace/ry_unitree_rl_lab`；本机 Git 副本为 `/home/ry/workspace/ry_unitree_rl_lab`。默认在容器工程修改和验证，再同步本机。

- 用户最新要求：每次完成修改及必要验证后，仅提交本地 Git；未经用户后续明确要求，不推送 GitHub。服务器继续同步训练与评估所需代码。结束前核对本机与容器代码一致，并说明本地提交及 GitHub 未推送状态；遇到冲突不覆盖他人改动。
- 称谓约定：参考工程指 GitHub `yusongmin1/My_unitree_go2_gym`；原工程指服务器 `/workspace/unitree_rl_lab`；当前工程指 `/workspace/ry_unitree_rl_lab`。分析时明确区分，避免将原工程固定前腿路线当成参考工程的实现。
- 后续直接覆盖当前实现，不建立并存的版本目录或旧代码副本。训练模型、日志和实验记录仍应保留，不作为旧代码副本清理。
- 现有 `ry_go2_biped_walk_v20/`、`scripts/train_v20.py`、`scripts/eval_v20.py` 和任务 ID 保留为稳定入口；当前内容为 12 关节实现。
- 奖励公式、权重与主要控制/PPO 参数优先复用用户指定的 `yusongmin1/My_unitree_go2_gym/go2_handstand`。不自行设计替代奖励或擅自调权重；接口适配及明确修正记录于 `doc/reference_port_manifest.json`。
- 用户最新批准恢复参考工程完整奖励，并从零训练 1000 轮：23 项非零原始公式及权重、线/角分母 0.25、批量高度均值 >0.70 门控；撤掉此前自定义踏步、欠抬脚、滑移、零转向、航向保持及前腿限定改写。本轮不加载旧权重；旧模型与日志保留。
- 本轮作为奖励对照，固定当前直立初始化和 Actor56/Critic155 观测。balance 仍为 50% 零速度、50% 前进 0.05～0.10 m/s，每 10 秒重采样；walk 固定 0.2 m/s；相位观测始终开启。默认 4096 环境、单卡、每 50 轮保存，结束后分别评估零速、0.05 和 0.10 m/s。
- 完整奖励复用不等于全部训练条件复现。控制、初始化、观测、命令及物理平台差异见 doc/参考奖励恢复与从零训练对照.md；未经用户要求，不再次自行收紧奖励或叠加约束。
- 外部工程依赖为 Isaac Lab 和 Unitree ROS。缺少资源先查已有工程，不生成替代机器人资源，不恢复对原 `unitree_rl_lab` 的运行依赖。
- 用户已授权本机 MuJoCo 部署：代码在本仓库 `deploy/mujoco/`，本机 `/home/ry/workspace/ry_unitree_rl_lab_mujoco` 链接到该目录，使用 `RL` conda 环境。复用 quadruped 的现有视觉资源；训练原始 URDF 用于机械参数适配，不修改 quadruped 源工程。策略和实验输出不提交。
- 未经明确授权不修改服务器硬件、驱动、系统 CUDA、内核或其他底层配置。
- 文档统一存放于工程 `doc/`；训练产物和生成模型不提交 Git。正式运行不得依赖临时测试产物；仅清理本轮创建且无需保留的临时文件。
- 用户已授权策略文件直接传输，无需逐次审核；按用户请求下载或同步策略时直接执行，并说明用途、方向和大小。其他文件继续遵守下述传输规则。
- 使用 `scp` 前，无论大小都须说明用途、方向和预计大小，由用户决定；其他大文件传输同样事先确认。代码同步优先使用 Git，不搬运模型、数据集或环境归档。

当前参数、命令与验证记录见 `doc/仓库使用说明.md` 和 `doc/V21_参考My_unitree_go2_gym的后腿步态优化方案.md`。上级工作区规范同样适用；用户最新指令优先。

## 当前服务器连接

跳板机已更新为 `houruibing@182.92.132.59:32768`，目标服务器仍为 `houruibing@10.29.150.42:22`（admin123-ESC8000A-E12）。本机 `unitree-train` 通过 `ProxyJump unitree-jump` 使用该链路。

```bash
ssh -J houruibing@182.92.132.59:32768 -p 22 houruibing@10.29.150.42
docker exec -it -w /workspace/ry_unitree_rl_lab isaac-record bash
```

2026-10-31 连接恢复后已补齐此前待同步代码；当前同步与续训记录见 `doc/服务器连接恢复与第九轮续训.md`。源码按文件哈希核对，容器 Git HEAD 仍可能落后于工作区实际内容，不据提交号单独判断代码版本。
