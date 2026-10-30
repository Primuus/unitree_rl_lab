# unitree_rl_lab

Go2 后腿双足行走的 RY v20 训练工作区，基于 Isaac Lab 和 RSL-RL。

本仓库包含 v20 任务、所需公共基础代码、双足 URDF、训练/评估入口及一份必需的热启动权重；运行不依赖原 `unitree_rl_lab` 工程目录。外部工程依赖为 Isaac Lab 和 `unitree_ros`。

- [仓库使用说明](doc/仓库使用说明.md)
- [工程结构及运行说明](doc/V20副本说明.md)
- [独立依赖迁移与验证](doc/独立依赖迁移说明.md)
- [训练与策略导出 Demo 实测](doc/训练与导出Demo_20261030.md)
- [代码来源与第三方许可](doc/代码来源说明.md)

已验证短训练、PT/ONNX 导出和 Isaac Lab 评估。Demo 权重尚未通过稳定交替行走验收。本仓库当前工作范围为训练与 Isaac Lab 评估。

训练生成的模型、日志、视频和导出包不纳入 Git；`checkpoints/warm_start_v11_model_40.pt` 是唯一随仓库保留的初始化权重。
