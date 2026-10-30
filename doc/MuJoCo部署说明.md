# Go2 后腿直立策略的本机 MuJoCo 部署

代码由本仓库 `deploy/mujoco/` 管理。本机入口 `/home/ry/workspace/ry_unitree_rl_lab_mujoco` 是该目录的符号链接，两条路径对应同一套文件。部署使用本机 `RL` conda 环境，不需要 Isaac Sim。

## 打开窗口

```bash
cd /home/ry/workspace/ry_unitree_rl_lab_mujoco
./run.sh
```

默认加载本次 stage1 最终 TorchScript 策略，平地、零速度命令，从后腿直立附近开始。按 `R` 重置，空格暂停/继续；关闭窗口或 Ctrl+C 退出。窗口按 60 Hz 同步，仿真尽量以实时速度运行。倒下后继续显示实际状态，不自动重置或关闭电机。

也可在已激活的环境中运行：

```bash
conda activate RL
python deploy.py
python deploy.py --policy models/stage1_balance_continue1998/exported/policy.onnx
python deploy.py --headless --duration 20 --output outputs/balance_continue1998_20s.json
```

`--duration 0` 表示持续运行；`--headless` 不限制到实时速度。输出 JSON 包含统计和每个策略步的轨迹，重置后重新统计。`--config` 可指定部署 JSON 配置；相对策略路径相对于部署代码目录。

## 策略和本地依赖

- 训练提交：`c92c7b908fad828afd1bfd17b1dd5c6b67f68ddd`。
- 训练目录：`2026-10-30_13-35-22_balance_stage1_balance_continue1000`。
- checkpoint：`model_1998.pt`，首轮 1000 次加续训 1000 次，共 2000 次更新（恢复时迭代标签从 999 开始）；导出文件为 `policy.pt` 和 `policy.onnx`。
- 部署文件保存在 `deploy/mujoco/models/stage1_balance_continue1998/`，包括 `exported/` 和 `params/`，不提交 Git。
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
| 12:14 | 步态 sin/cos，balance 为零 |
| 14:26 | 关节位置减动作零点，按 Isaac 原生关节顺序 |
| 26:38 | 关节速度 × 0.05，按同一原生顺序 |
| 38:50 | 上一步裁剪后的动作，按动作顺序 |
| 50:52 | 直立航向 sin/cos |
| 52:54 | 两只后脚相对最低脚的高度差 × 10，截断至 [0,1] |
| 54:56 | 后脚世界 Z 接触合力 / (16.087 × 9.81)，截断至 [0,1.5] |

观测关节顺序为 hip 四关节、thigh 四关节、calf 四关节，每组 FL/FR/RL/RR；动作顺序为 FL/FR/RL/RR，每腿 hip/thigh/calf。名称映射与训练 `provenance.json` 一致。

策略周期 20 ms，物理步长 5 ms，decimation=4。PD 每个物理步重算，Kp=40、Kd=1；动作裁剪 ±100，目标角为 `action_offset + 0.25 × action`。力矩限幅取训练 URDF：hip/thigh ±23.7 Nm、calf ±45.43 Nm。

初始高度 0.52 m，四元数 wxyz 为 `[√0.5,0,-√0.5,0]`。初始关节角取训练 desired angles，动作零点取 default angles，两者分别配置；零动作不等于保持初始直立姿态。

## 机器人资源适配

`resources/go2_description.urdf` 是从服务器 `/workspace/unitree_ros/robots/go2_description/urdf/go2_description.urdf` 读取的原始文件，SHA256 已核对。它不依赖旧六动作资源。

`model.py` 在加载时转换该 URDF 的运动链、质量、惯量、原始碰撞基元和关节限位，复用参考工程 XML 中的视觉网格与材质。保留原始转子、头部和足端质量，并融合固定关节，总质量为 16.087 kg；通过 site 保留足端位置。圆柱按训练导入选项转换为胶囊。关节无额外被动阻尼、摩擦或 armature，控制阻尼由 PD 提供。地面/机器人采用固定摩擦参数，不执行训练时的材质随机化。

本部署借鉴 `quadruped` 的策略加载、逐步 PD、关节名称映射和 viewer 关闭流程；观测和初始化针对本任务独立实现。PhysX 与 MuJoCo 的接触求解并不相同，参数对应不等于两引擎轨迹一致。

接触力使用 `mj_contactForce`，从接触坐标转换到世界坐标，并按 geom1/geom2 分别累加负/正作用力。依据 [MuJoCo 接触力说明](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)。

## 验证与当前效果

```bash
cd /home/ry/workspace/ry_unitree_rl_lab_mujoco
conda run -n RL python -B -m unittest test_deployment -v
```

四项检查覆盖观测坐标和关节排列、接触力方向与重量、URDF 质量及限位、PT/ONNX 推理一致性。模型文件是正式部署输入，不依赖测试产物。

续训最终策略在 MuJoCo 的 20 秒无界面运行中数值有限，但约 2 秒内倒下，姿态与高度同时达标比例为 3.8%，前脚触地比例为 17.6%。同一 checkpoint 在 Isaac Lab 的 4 环境、20 秒独立评估中平均高度为 0.505 m，姿态与高度同时达标比例为 95.3%，速度误差约 0.216 m/s，无摔倒，但平衡验收仍未通过。MuJoCo 中的倾倒比 Isaac 明显，说明跨仿真的表现差异仍未解决；部署链路可运行不代表 sim2sim 成功。

Isaac 评估记录保留于服务器 `logs/evaluation/stage1_continue_final_1998.*`；MuJoCo 正式评估记录留在本机 `outputs/balance_continue1998_20s.json`。后续可直接替换同观测/动作契约的策略文件观看新结果。

上一轮 model_999 的策略保留于 `models/stage1_balance/`，历史 MuJoCo 记录为 `outputs/balance_20s.json`。可用 `./run.sh --policy models/stage1_balance/exported/policy.pt` 对照查看。

视觉适配显式设置 `type="mesh"`，保留参考 XML 的网格类型；移除默认类时不能丢失该属性，否则外观会被渲染为球体。已检查 33 个视觉网格，并确认修正前后的 20 秒物理轨迹完全一致。
