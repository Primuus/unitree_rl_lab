# model_2996 的 MuJoCo 抖动诊断

2026-10-30。用户观看最新策略后指出抖动明显。本次分别检查了策略、关节状态和物理步长；没有重新训练或修改奖励。

## 观测与对照

同一 model_2996，固定初始状态，每组运行 20 秒。MuJoCo 比较 5 ms、2.5 ms、1 ms 三种物理步长，策略周期均为 20 ms，PD 均为 40/1，其余模型、接触、策略、关节限速投影保持不变。关节频谱与速度统计排除前 2 秒初始化，采用逐物理步数据。

| 指标 | MuJoCo 5 ms | MuJoCo 2.5 ms | MuJoCo 1 ms |
| --- | ---: | ---: | ---: |
| 右前小腿速度 RMS rad/s | 10.801 | 0.863 | 0.899 |
| 左前小腿速度 RMS rad/s | 6.476 | 1.137 | 1.162 |
| 右前小腿角度高于 10 Hz 的功率占比 | 50.8% | 1.86% | 1.02% |
| 右前小腿速度触限步比例（2～20 秒） | 14.53% | 0 | 0 |
| 全部关节限速修正次数（0～20 秒） | 772 | 16 | 24 |
| 直立判定时间比例（0～20 秒） | 100% | 100% | 100% |
| 前脚触地时间比例 | 0 | 0 | 0 |
| 平均绝对前向速度 m/s | 0.11445 | 0.13064 | 0.13426 |

5 ms 下，前腿大腿和小腿的主导角度振动为 100 Hz，达到采样 Nyquist 频率，表现为物理步间交替变化。缩小物理步长后，此高频主导成分消失，前腿小腿速度 RMS 大幅下降。两种更小步长的关节速度结果接近，支持原先高频抖动包含明显的数值积分因素；这不是策略输出平滑性已解决的证明。漂移速度反而略增，因此不能把抖动减轻描述成所有指标同时改善。

另在 Isaac Lab 使用同一 checkpoint、1 环境、seed=1、20 秒评估，逐物理步记录（5 ms）。该次评估全程直立、无倒地，水平速度约 0.08093 m/s。其左右前小腿速度 RMS 分别为 1.035 和 0.754 rad/s，没有 MuJoCo 5 ms 下的 100 Hz 主导成分。

## 剩余摆动

Isaac Lab 两个前髋关节的主导角度频率约 7.33 Hz，速度 RMS 约 2.23、2.21 rad/s；MuJoCo 1 ms 下约 8.94 Hz，速度 RMS 约 2.30、2.16 rad/s。训练仿真中也存在前腿摆动，因此剩余问题需要训练侧分析，不能全部归为 MuJoCo 的显示或积分问题。

Isaac Lab 前髋目标角每 20 ms 的差分 RMS 约 0.073、0.067 rad，后腿大腿/小腿约 0.148～0.162 rad。这提示策略与控制闭环仍有明显动作变化；仅凭这些记录不能单独断言具体奖励项是原因。本轮未增加动作滤波、修改 PD 或固定前腿。

## 已采用修正

本地部署 config.json 的 physics_dt 改为 0.001，decimation 改为 20，策略周期仍为 0.02 秒。训练环境仍保持原物理步长与控制设置。此次修改只作用于 MuJoCo 部署。

既有关节速度与位移投影仍保留，是 PhysX 限速的数值近似，并非完整约束求解等价实现。部署限速测试改为显式设置超限初始关节速度，以独立于策略和物理步长验证投影，五项部署检查通过。

使用正式命令行和新默认配置重新运行 20 秒，结果与 1 ms 对照一致：直立判定 100%，前脚触地 0，平均绝对前向速度 0.13426 m/s，有限状态。窗口曾使用新设置重新打开，随后正常关闭；不会自动再次打开。

## 记录

本地：
- deploy/mujoco/outputs/jitter2996_mujoco.npz
- deploy/mujoco/outputs/jitter2996_mujoco_summary.json
- deploy/mujoco/outputs/jitter2996_timestep_comparison.json
- deploy/mujoco/outputs/balance_sigma001_2996_1ms_20s.json
- deploy/mujoco/outputs/viewer_sigma001_2996_1ms.log

服务器：
- logs/evaluation/jitter2996_isaac.json
- logs/evaluation/jitter2996_isaac.trace.json
- logs/evaluation/jitter2996_isaac.joints.npz
- logs/evaluation/jitter2996_isaac.jitter.json
- logs/evaluation/jitter2996_isaac.log

临时远程诊断脚本已删除。模型、日志及上述实验记录保留，不提交 Git。代码和文档仅提交本地 Git，不推送 GitHub。
