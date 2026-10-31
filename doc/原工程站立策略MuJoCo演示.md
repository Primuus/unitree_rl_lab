# 原工程站立策略的本地 MuJoCo 回放

2026-10-31，按用户要求只读评估服务器原工程 /workspace/unitree_rl_lab 的候选策略，导出 V6 model_3999；没有启动训练或修改原工程。此模型是固定前腿、后腿6动作，与当前工程12动作策略不同。

## 选择依据

四个候选统一采用16环境、20秒、seed=1、零命令、关闭观测/材质随机化，在各自原任务中评估。按摔倒更少、直立比例更高、前向速度误差更小选择。

| 候选 | 摔倒环境数 | 直立比例 | 平均高度 m | 平均绝对前向速度 m/s | 平均绝对世界Z角速度 rad/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| V6 model_3999 | 0/16 | 1.000 | 0.52556 | 0.01590 | 0.24638 |
| V7 model_800 | 0/16 | 1.000 | 0.51624 | 0.03646 | 0.28990 |
| V7 model_3100 | 16/16 | 0.81591 | 0.51323 | 0.11554 | 0.38687 |
| V11 model_40 | 0/16 | 1.000 | 0.52205 | 0.03450 | 0.27765 |

V6是本次四个候选中站立效果最好者，不代表已穷尽原工程所有checkpoint；它仍有转动，未达到严格原地静止验收。原始评估与选择记录保留在服务器当前工程 logs/evaluation/original_stand_selection*。

## 本地运行

```bash
cd /home/ry/workspace/ry_unitree_rl_lab_mujoco
./run.sh --config original_stand.json
```

按 R 重置，空格暂停。使用本机 RL conda 环境。该命令明确选择原工程V6；默认 ./run.sh 仍加载用户此前指定的12动作长训model_5990。

```bash
./run.sh --config original_stand.json --headless --duration 20 --output outputs/original_v6_model3999_final_20s.json
```

**目前只能回放，尚未复现原工程稳定站立。** 最终固定初态20秒回放在2.70秒触发高度/倾角失败，3秒时高度降至约0.174m，之后前腿触地。失败前平均绝对前向速度0.22206m/s、世界Z角速度1.27214rad/s。此结果不能称为成功sim2sim或稳定站立演示。

## 导出及适配

- checkpoint：/workspace/unitree_rl_lab/logs/rsl_rl/unitree_go2_biped_walk_v6/2026-10-29_11-29-33_balance_first_heading_v6_20261029/model_3999.pt
- checkpoint SHA256：e329c6d855afd6ba5af17edbcfc4a2707218c78bb58d49b256a270dee5c88277
- 导出目录：服务器当前工程 logs/evaluation/original_stand_selection_v6_3999_export/。
- 下载方向为服务器到本机 deploy/mujoco/models/original_stand_selected/，PT、ONNX和contract.json共488611字节。文件哈希已逐项核对；TorchScript所有Actor参数与checkpoint一致。模型与实验结果不提交Git。
- PT SHA256：fac5d13c7d44b2fdd819fefe771dde363f401d77953e477c91719ffb7c115bd6
- ONNX SHA256：ed5a0bd6213ecbca6a1ca02455c73eb23d5ac4959b797196c429e6effdd92387
- contract.json SHA256：9f5bb084ffe67cbfd6146ddaa0f2fc541d4740ad910d1f54099c7ebfcd62be2b
- 34维Actor、6动作，无历史与归一化器。动作按RL三关节、RR三关节；位置/速度观测按左右hip、左右thigh、左右calf。前腿由原始固定关节URDF约束。
- 复用仓库已有的 go2_biped_walk/resources/go2_biped_walk.urdf，SHA256与原工程一致；运行不依赖服务器原工程。视觉资源沿用quadruped现有网格。
- PD=60/2.5，动作缩放0.4，使用原任务处理后目标角限幅，以及Unitree的20.2/23.4Nm力矩—速度曲线。原始动作不裁剪，作为下一步动作观测。
- 控制周期20ms；沿用本机1ms积分、20倍降采样。训练是5ms；诊断中改为训练步长也未恢复站立。关节限速仍为已有数值投影近似，不等价于PhysX约束求解。
- 原工程关节摩擦系数0.01没有直接换成MuJoCo的frictionloss扭矩；两者不是同一物理参数。

## 已核对的问题与验证

初始观测两端误差小于6e-8，初始Actor动作一致，初始后足位置一致，排除了首帧关节顺序和导出权重错误。

本机MuJoCo 3.8.1对该固定前腿URDF启用fusestatic时，整机零姿态质心变为[-0.043985,0.053280,-0.022080]；关闭融合后为[0.001639,0,-0.015207]。因此原工程配置显式关闭融合，保留固定连接。按训练时合并基座的实际COM [0.074616663,0,-0.002222844]计算线速度观测；用独立点Jacobian验证该速度换算。官方概念说明见[MuJoCo刚体坐标与惯性坐标](https://github.com/google-deepmind/mujoco/blob/main/doc/overview.rst)。以上偏差是本机实测，未断言所有MuJoCo版本都有此问题。

关闭融合后各部件质量保留，不额外生成机器人资源。既有12动作配置保持原设置；此项修正仅由original_stand.json启用。

部署9项回归检查通过，覆盖原有12动作、新增6动作/34维与38维排列、PT/ONNX一致性、固定部件质心、观测COM速度及低高度失败判定。MuJoCo记录对原工程配置增加0.35m高度和0.95rad倾角失效阈值，防止趴低但基座未碰地被误记为无失败；此处不等同于原Isaac任务所有接触终止项。

当前能确认导出正确且部署存在动力学差异；修正已发现的质心问题仍未消除失稳。不能据此断言原工程策略在Isaac中也站不住。后续若继续排查，应对比同状态下执行器力矩与接触响应；本次未修改训练奖励、未重训、未推送GitHub。
