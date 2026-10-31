# Go2 后腿直立策略的本机 MuJoCo 部署

代码由本仓库 `deploy/mujoco/` 管理。本机入口 `/home/ry/workspace/ry_unitree_rl_lab_mujoco` 是该目录的符号链接，两条路径对应同一套文件。部署使用本机 `RL` conda 环境，不需要 Isaac Sim。

当前用户指定使用约6000轮站立长训model_5990。本地PT、ONNX及四份配套配置已与服务器六个文件哈希重新核对一致，直接复用现有文件。model_2497继续保留为后续训练基线；此次仅切换本地观看策略。

原工程固定前腿V6模型可用 `./run.sh --config original_stand.json` 回放；当前MuJoCo仍会失稳，选择依据与实测限制见 [原工程站立策略回放](原工程站立策略MuJoCo演示.md)。

## 打开窗口

```bash
cd /home/ry/workspace/ry_unitree_rl_lab_mujoco
./run.sh
```

默认加载用户指定的长训 model_5990 的 TorchScript 策略（累计约6000轮、训练抬脚目标2cm、旧相位模式），平地、零速度命令，从后腿直立附近开始。按 `R` 重置，空格暂停/继续；关闭窗口或 Ctrl+C 退出。窗口按 60 Hz 同步，仿真尽量以实时速度运行。倒下后继续显示实际状态，不自动重置或关闭电机。

也可在已激活的环境中运行：

```bash
conda activate RL
python deploy.py
python deploy.py --policy models/stage1_balance_clearance10_5990/exported/policy.onnx
python deploy.py --headless --duration 20 --output outputs/balance_mixed_height_2497_20s.json
```

`--duration 0` 表示持续运行；`--headless` 不限制到实时速度。输出 JSON 包含统计和每个策略步的轨迹，重置后重新统计。`--config` 可指定部署 JSON 配置；相对策略路径相对于部署代码目录。


## 历史下载记录：model_2497

2026-10-31，用户弃用续训750轮的全部策略结果后，将model_2497导出文件下载到本机并设为默认；没有下载或启用model_3246。

- 服务器训练目录：logs/rsl_rl/ry_unitree_go2_handstand/2026-10-31_09-53-03_balance_height_gate_continue500。
- 本地目录：deploy/mujoco/models/balance_mixed_height_2497/。
- PT、ONNX及四份配置共1,618,058字节，六个SHA256均与服务器一致；TorchScript全部Actor参数与checkpoint完全相同。
- 训练时50%零命令、50%前进0.05～0.10m/s；本地默认零命令。Actor56、动作12、相位always、周期1.6s、抬脚目标6cm。
- 动作顺序、观测关节顺序、动作零点、初始姿态、PD和控制周期与保存配置核对；采用env.yaml中的精确数值，deploy.yaml部分浮点值有格式化舍入。
- 继续使用既有MuJoCo 1ms物理步长和关节限速近似。
- 本机RL环境7项部署测试通过，包含PT/ONNX输出一致性；旧相位测试改为显式模拟旧来源，不再假定默认策略一定为旧相位。
- 20秒固定初态回放：状态有限、无基座接触失败、无前脚触地，平均绝对前向速度0.05404m/s、平均绝对世界Z角速度0.50095rad/s、最大水平位移0.97659m。姿态与高度同时达标比例2.4%，主要表现为约0.42m的低姿态，站立验收未通过。
- 结果：deploy/mujoco/outputs/balance_mixed_height_2497_20s.json。此验证确认部署链路可运行，不代表已经解决sim2sim差异或站稳。
- 权重、配套下载配置及回放数据不提交Git，旧模型全部保留。下文model_5990/2996等记录为历史对照。

## 当前长训model_5990与本地依赖

- 对应训练代码本地提交：`f434084`（未推送 GitHub）。
- 训练目录：`2026-10-31_04-13-39_balance_balance_clearance10_continue500`。
- checkpoint：`model_5990.pt`，累计 6000 次更新（恢复时迭代标签有重叠）；导出文件为 `policy.pt` 和 `policy.onnx`。
- 部署文件保存在 `deploy/mujoco/models/stage1_balance_clearance10_5990/`，包括 `exported/` 和 `params/`，不提交 Git。
- 使用 `/home/ry/miniconda3/envs/RL`；验证版本为 Python 3.13.13、MuJoCo 3.8.1、PyTorch 2.12.0+cpu、ONNX Runtime 1.27.0。
- 视觉资源复用 `/home/ry/project/quadruped/src/controller/rl/resources/robots/go2/go2.xml` 和 `obj/`，由 `config.json` 的 `visual_source` 指定。不修改该工程，不复制或传输其约 28 MB 网格。
- 策略来源、文件哈希和机器人资源来源见 [来源清单](mujoco_deployment_manifest.json)。新机器需要提供这些本地依赖；从 Git 拉取代码不会自动下载模型和网格。

## 观测与控制契约

只执行 Actor：输入 `[1,56]`，输出 `[1,12]`，无历史堆叠、无 Critic、无观测归一化器。部署按评估模式不加观测噪声，不额外裁剪整体观测。

| 切片 | 内容 |
| --- | --- |
| 0:3 | 直立航向坐标系线速度；使用基座质心速度 |
| 3:6 | 本体角速度 × 0.25 |
| 6:9 | 本体重力投影 |
| 9:12 | 命令，balance 为零 |
| 12:14 | 步态 sin/cos；由策略来源记录选择模式，旧 balance 为零，新 balance 持续运行 |
| 14:26 | 关节位置减动作零点，按 Isaac 原生关节顺序 |
| 26:38 | 关节速度 × 0.05，按同一原生顺序 |
| 38:50 | 上一步裁剪后的动作，按动作顺序 |
| 50:52 | 直立航向 sin/cos |
| 52:54 | 两只后脚相对最低脚的高度差 × 10，截断至 [0,1] |
| 54:56 | 后脚世界 Z 接触合力 / (16.087 × 9.81)，截断至 [0,1.5] |

观测关节顺序为 hip 四关节、thigh 四关节、calf 四关节，每组 FL/FR/RL/RR；动作顺序为 FL/FR/RL/RR，每腿 hip/thigh/calf。名称映射与训练 `provenance.json` 一致。

策略周期 20 ms，MuJoCo 物理步长 1 ms，decimation=20；训练环境物理步长仍为 5 ms。缩小部署积分步长用于减轻高频数值抖动，见 [抖动诊断](MuJoCo抖动诊断.md)。PD 每个物理步重算，Kp=40、Kd=1；动作裁剪 ±100，目标角为 `action_offset + 0.25 × action`。力矩限幅取训练 URDF：hip/thigh ±23.7 Nm、calf ±45.43 Nm。

初始高度 0.52 m，四元数 wxyz 为 `[√0.5,0,-√0.5,0]`。初始关节角取训练 desired angles，动作零点取 default angles，两者分别配置；零动作不等于保持初始直立姿态。

## 机器人资源适配

`resources/go2_description.urdf` 是从服务器 `/workspace/unitree_ros/robots/go2_description/urdf/go2_description.urdf` 读取的原始文件，SHA256 已核对。它不依赖旧六动作资源。

`model.py` 在加载时转换该 URDF 的运动链、质量、惯量、原始碰撞基元和关节限位，复用参考工程 XML 中的视觉网格与材质。保留原始转子、头部和足端质量，并融合固定关节，总质量为 16.087 kg；通过 site 保留足端位置。圆柱按训练导入选项转换为胶囊。关节无额外被动阻尼、摩擦或 armature，控制阻尼由 PD 提供。地面/机器人采用固定摩擦参数，不执行训练时的材质随机化。

本部署借鉴 `quadruped` 的策略加载、逐步 PD、关节名称映射和 viewer 关闭流程；观测和初始化针对本任务独立实现。PhysX 与 MuJoCo 的接触求解并不相同，参数对应不等于两引擎轨迹一致。

接触力使用 `mj_contactForce`，从接触坐标转换到世界坐标，并按 geom1/geom2 分别累加负/正作用力。依据 [MuJoCo 接触力说明](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)。

## 历史策略验证记录

历史结果使用 [stationary-v2 站立验收](站立验收条件.md)。model_5990 的 20 秒回放在前向速度、转动角速度、航向偏差、最大位移和水平路径五项上失败；全程直立不能视为原地站稳。新结果保存在 `outputs/balance_clearance10_5990_acceptance_v2_20s.json`，历史结果保留。

历史 model_5990：PT、ONNX 和四个配套配置共 1,615,482 字节，六个 SHA256 与服务器一致。7 项部署检查通过，PT/ONNX 输出一致。MuJoCo 固定初态 20 秒验证直立比例 100%、前脚接触 0、平均绝对前向速度 0.11642 m/s、状态有限；仍有运动，不能将 Isaac 的 0.02075 m/s 直接视为 MuJoCo 效果。输出：`deploy/mujoco/outputs/balance_clearance10_5990_1ms_20s.json`。

该历史model_5990训练时目标为2cm；另存的model_2497已按6cm训练；当前用户选定model_5990。奖励高度不会在部署时直接修改动作。

```bash
cd /home/ry/workspace/ry_unitree_rl_lab_mujoco
conda run -n RL python -B -m unittest test_deployment -v
```

七项检查覆盖观测坐标和关节排列、接触力方向与重量、URDF 质量及限位、首步关节限速与位移一致性、PT/ONNX 推理一致性，以及转动验收与失败后停止计分、新旧相位配置和复位行为。模型文件是正式部署输入，不依赖测试产物。

历史 model_2996 已下载 PT、ONNX 和配套配置，共 1,613,197 字节，六个文件 SHA256 均与服务器一致。部署控制配置与旧策略相同。固定初始状态的 20 秒 MuJoCo 验证中，直立判定比例 100%，前脚触地比例 0，平均绝对前向速度 0.13426 m/s，数值有限；仍有漂移。当前默认物理步长为 1 ms，记录位于 `outputs/balance_sigma001_2996_1ms_20s.json`；此前 5 ms 的记录保留。本结果使用下述既有限速数值近似，不代表两引擎物理等价，也不是多初始状态的稳健性验收。

历史 model_1998：补齐限速前，该策略在 MuJoCo 的 20 秒无界面运行中数值有限，但约 2 秒内倒下，姿态与高度同时达标比例为 3.8%，前脚触地比例为 17.6%。同一 checkpoint 在 Isaac Lab 的 4 环境、20 秒独立评估中平均高度为 0.505 m，姿态与高度同时达标比例为 95.3%，速度误差约 0.216 m/s，无摔倒，但平衡验收仍未通过。MuJoCo 中的倾倒比 Isaac 明显，说明跨仿真的表现差异仍未解决；部署链路可运行不代表 sim2sim 成功。

Isaac 评估记录保留于服务器 `logs/evaluation/stage1_continue_final_1998.*`；MuJoCo 正式评估记录留在本机 `outputs/balance_continue1998_velocity_limits_20s.json`。后续可直接替换同观测/动作契约的策略文件观看新结果。

历史 model_1998 保留于 `models/stage1_balance_continue1998/`，可用 `./run.sh --policy models/stage1_balance_continue1998/exported/policy.pt` 对照。

历史 model_999 的策略保留于 `models/stage1_balance/`，历史 MuJoCo 记录为 `outputs/balance_20s.json`。可用 `./run.sh --policy models/stage1_balance/exported/policy.pt` 对照查看。

视觉适配显式设置 `type="mesh"`，保留参考 XML 的网格类型；移除默认类时不能丢失该属性，否则外观会被渲染为球体。已检查 33 个视觉网格，并确认修正前后的 20 秒物理轨迹完全一致。

## 关节限速适配与验证

此前部署只读取 URDF 的角度和力矩限制，遗漏了训练 PhysX 实际启用的关节速度限制。运行时核对：hip/thigh 为 30.1 rad/s，calf 为 15.7 rad/s；armature 和关节摩擦均为零。相同初始状态下，Actor 的首步输出一致，服务器导出策略与 checkpoint Actor 在前 100 个控制步的输出误差为零。初始关节和足端运动学也对应。

当前转换器将原始 URDF 的 `limit.velocity` 保存在 `jnt_user[:,0]`。每个物理步之后，把受控关节的速度投影到该范围，并按 `implicitfast` 的积分方式修正这一步的关节位置增量；不修改浮动基座位姿或速度。只裁剪速度、不修正该步关节位移，在诊断中仍会失稳。

这是对 PhysX 求解器限速的数值近似，未复现约束冲量向整个系统的耦合传播，也不等同于真实电机的力矩—速度曲线。它会改变 MuJoCo 动力学轨迹；不能据此宣称两引擎物理等价或完成了可靠的 sim2sim 迁移。配置中的 PD、策略周期、物理步长、质量和接触参数保持原值，未采用诊断中尝试的接触刚度修改。

固定初始状态、同一个 model_1998 的 20 秒无界面验证中，限速适配后未再直接倾倒，姿态与高度同时达标比例为 96.5%，前脚触地比例为零；平均绝对前向速度仍为 0.425 m/s，尚未达到原地站稳。约第 18 秒出现短暂下蹲和倾斜，最低高度 0.350 m、最低直立余弦 0.840，随后恢复；不能描述为全程稳定站立。该结果只代表当前初始状态下的一次回放，不是多初始状态的稳健性验收。

修正前的记录保留在 `outputs/balance_continue1998_20s.json`，修正后为 `outputs/balance_continue1998_velocity_limits_20s.json`。服务器的初始状态/运行时参数诊断保存在 `logs/evaluation/sim2sim_exact_initial_1998.*`。MuJoCo 关节属性与积分语义参见 [XML 参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint) 和 [数值积分说明](https://mujoco.readthedocs.io/en/stable/computation.html#numerical-integration)。

## 相位观测配置

默认 `gait_clock_mode=auto` 从所选策略同级运行目录的 `params/provenance.json` 读取 `layout.gait_clock`。旧来源文件无该字段时，按历史 `moving_only` 模式和 1.6 s 周期回放；新 balance 使用 `always`。下载新策略时须携带其配套来源文件。缺少来源文件时不猜测模式；自定义策略可通过 `--config` 指定明确的 `always` 或 `moving_only`。实际模式打印在启动日志和输出 JSON 中。

历史 model_5990 自动选择旧模式；20 s 回放与修复前的 1000 个轨迹样本及原有统计字段完全一致，记录于 `outputs/balance_5990_phase_contract_legacy_20s.json`。完整说明见 [相位与欠抬脚接入说明](站立相位观测与欠抬脚惩罚.md)。
