# 当前实现：12 关节控制与后腿直立行走

2026-10-31 最新配置：恢复参考工程全部 23 项奖励，从零训练对照；原 27 项配置退出当前任务。直接覆盖原 V20，后续也在当前实现上修改，不建立并存的版本目录或旧代码副本。现有任务 ID、包名和训练/评估入口保留，用于保持调用路径稳定。

## 来源与移植范围

主要来源为 [My_unitree_go2_gym/go2_handstand](https://github.com/yusongmin1/My_unitree_go2_gym/tree/main/legged_gym/envs/Go2_Stand/Go2_Handstand)，对应用户提供的 [演示视频](https://www.bilibili.com/video/BV19fToz9EZ5/)。

上游原文件及 MIT 许可保存在工程的 `third_party/My_unitree_go2_gym/`。活动奖励函数从上游源码提取到 `reference_rewards.py`，参数字面值提取到 `reference_config.py`，以原文件为对照运行正式回归测试。来源文件 Git blob、SHA256、启用权重与适配差异见 [移植清单](reference_port_manifest.json)。

当前逐项恢复固定来源中的原始奖励公式和权重；保留混合命令、直立初始化和 56/155 观测作为对照条件。全部差异及本轮验证见[奖励恢复对照](参考奖励恢复与从零训练对照.md)。

## 当前模型与控制

- 完整模型来自 `/workspace/unitree_ros/robots/go2_description/urdf/go2_description.urdf`，原始资源不修改。
- 全部 12 个关节由策略控制；动作顺序为 FL、FR、RL、RR，每条腿依次 hip、thigh、calf。
- 显式 PD 使用上游 `stiffness=40`、`damping=1`，动作缩放为 `0.25`；力矩、速度限值读取现有 URDF。
- 关节目标为 `0.25 * action + 上游 default_joint_angles`，初始直立姿态使用上游 `descire_joint_angles`。两套角度各有用途，不混用。
- 前腿初始 thigh/calf 为 `0.8 / -1.5 rad`，后腿为 `2.25 / -1.75 rad`；这替代早期方案中的前腿初始角候选。
- 控制周期 `0.02 s`，步态周期 `1.6 s`，机身高度奖励目标 `0.52 m`、前脚高度目标 `0.67 m`、后脚抬脚目标 `0.06 m`，均来自上游。
- 使用已有观测结构扩展到 Actor `56`、Critic `155` 维，保留相位、航向和后脚状态；这与上游的 `45/86` 观测布局不同。
- 镜像按实际关节名映射，独立处理原生观测顺序与动作顺序。包括前腿的左右交换与 hip 符号。

前脚离地、前脚高度、默认姿态、左右关节对称等约束均采用上游已有项，没有额外设计前腿摆动目标。

## 奖励与权重

下表即当前两个入口启用的全部 23 项非零奖励。上游未启用的 feet_air_time 不加入；此前额外的四项奖励已移出当前配置。

| 上游项 | 权重 |
| --- | --- |
| `tracking_lin_vel` | 2.5 |
| `tracking_ang_vel` | 2.5 |
| `lin_vel_z` | 0.2 |
| `ang_vel_xy` | 0.2 |
| `handstand_orientation` | -1 |
| `torques` | -0.0002 |
| `dof_acc` | -2.5e-7 |
| `base_height` | 1.5 |
| `handstand_feet_on_air` | 0.4 |
| `collision` | -2 |
| `action_rate` | -0.05 |
| `default_pos` | -0.1 |
| `default_hip_pos` | -0.1 |
| `feet_clearance` | 0.4 |
| `ang_xz` | -0.5 |
| `contact` | 0.3 |
| `symmetric_joints` | -0.1 |
| `orientation_symmetry` | -0.5 |
| `feet_height_symmetry` | -0.2 |
| `handstand_feet_height_exp` | 5 |
| `default_pos_reward` | 0.5 |
| `dof_pos_limits` | -2 |
| `alive` | 1 |

全部奖励方法 AST 与固定上游源码一致：恢复包含后腿的 default_pos 和 symmetric_joints、批量高度均值 >0.70 门控，以及原始 phase==0.5 边界。高度奖励为 exp(-5*abs(h-0.52))，权重 +1.5。抬脚只计算摆动脚，权重 +0.4；接触只要求恰好一只后脚触地，权重 +0.3。不存在额外欠抬脚、滑移、零转向或航向保持奖励。

接口层按名字重排关节/足端、广播标量 alive、重置速度历史，并先按当前状态计算高度门控，避免奖励顺序读取旧高度分数。该时序适配已在清单明确列出。

## PPO 与训练阶段

PPO 使用上游隐藏层 `[512, 256, 128]`、ELU、初始标准差 `1.0`、学习率 `1e-3`、adaptive schedule、entropy coefficient `0.01`、每环境 rollout `24` 步。其余 PPO 参数来自同一上游配置。使用当前 RSL-RL 的镜像损失接口，系数 `1.0`；不同时启用镜像数据增强。

当前 balance 是混合训练入口：50% 零速度、50% 前进 0.05～0.10 m/s；每 10 秒及重置时重采样，横向和转向为零。walk 保留固定前进 0.2 m/s，暂不作为下一轮主训练入口。

两个入口共用上表 23 项奖励，线/角速度指数分母均为 0.25；参考摆动峰值 6 cm、周期 1.6 s，不再使用自定义双支撑目标。相位观测始终可见。

阶段名称保留稳定入口。初始状态位于直立附近；首版不实现四足起立课程。地形为平地，材质摩擦与恢复系数随机化采用上游范围；上游其他质量、质心、增益、延迟和推扰随机化尚未移植。这是奖励和控制的移植，不宣称复现上游全部训练条件。

任务使用与上游相同的机身坐标速度奖励：直立时机身 `-Z` 对应水平前进，机身 `+X` 对应竖直方向。源码公式保留，没有替换成自拟的世界速度奖励。本轮命令无转向，实际启停、行走与航向保持能力需分别评估。

## 运行入口

在目标容器中执行：

```bash
cd /workspace/ry_unitree_rl_lab
./ry_python.sh scripts/train_v20.py --stage balance \
  --num_envs 4096 --max_iterations 15000 --run_name balance \
  --headless --device cuda:0
```

本轮已获用户授权从零训练 1000 轮，不传 --resume。完整命令与自动评估位置见[运行说明](仓库使用说明.md)和[本轮记录](参考奖励恢复与从零训练对照.md)。

旧的 6 动作权重与新网络不兼容，不再要求 `--warm_start_actor`。已有权重与日志作为实验记录保留，未删除，也不作为当前任务依赖。

评估使用同一阶段和当前权重：

```bash
./ry_python.sh scripts/eval_v20.py --stage walk \
  --checkpoint logs/rsl_rl/ry_unitree_go2_handstand/<训练目录>/model_<迭代>.pt \
  --num_envs 16 --steps 1000 --speed 0.2 \
  --out logs/evaluation/walk --headless --device cuda:0
```

不传 checkpoint 时为零动作执行检查，不算训练策略验收。评估增加前脚触地与速度误差统计，前进速度按命令误差验收；正常超时与异常终止分开记录。原有独立合格落脚、交替率和滑移诊断仍保留，没有为了匹配奖励而放宽这些证据。

## 代码位置与验证

- `ry_go2_biped_walk_v20/`：当前唯一任务实现，历史路径名保留。
- `reference_config.py`、`reference_rewards.py`：从上游提取的配置与公式。
- `gait_mdp.py`：Isaac Lab 数据适配、状态历史和观测。
- `robot_cfg.py`、`gait_env_cfg.py`、`symmetry.py`：完整模型、阶段配置和镜像。
- `scripts/train_v20.py`、`scripts/eval_v20.py`：当前训练与评估入口。
- `tests/test_reference_port.py`：上游公式/参数、门控、相位、镜像回归测试。

当前验证和运行记录见[参考奖励恢复对照](参考奖励恢复与从零训练对照.md)。初次移植的历史验证见[V21 验证报告](V21_实现与验证.md)。只有通过独立步态验收，才认定策略达到目标效果。
