# unitree_rl_lab

Go2 后腿直立训练工作区，基于 Isaac Lab 和 RSL-RL。当前已直接覆盖原 V20，实现完整 12 关节控制，奖励公式、权重与主要控制/PPO 参数来自 My_unitree_go2_gym 的 go2_handstand。

任务 ID 和入口路径保持稳定；后续直接修改当前代码。外部工程依赖仅为 Isaac Lab 和 unitree_ros，不依赖原 unitree_rl_lab。

- [运行说明](doc/仓库使用说明.md)
- [当前 12 关节训练实现与参数](doc/V21_参考My_unitree_go2_gym的后腿步态优化方案.md)
- [实现验证报告](doc/V21_实现与验证.md)
- [上游来源与适配清单](doc/reference_port_manifest.json)
- [代码来源与许可证](doc/代码来源说明.md)

旧权重和实验记录保留用于溯源，不作为当前 12 动作策略的初始化依赖。训练模型、日志和导出包不纳入 Git。
