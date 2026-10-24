# Dependencies and attribution

本仓库代码以 MIT 分发。下列依赖独立安装，不重新打包其代码或二进制；
其许可证由对应项目负责。外部 Q1 XML、meshes 和重定向数据不随源码分发。

| 项目 | 用途 | 来源 / 许可证 |
|---|---|---|
| PyTorch | 网络、梯度、编译与 distributed | [pytorch/pytorch](https://github.com/pytorch/pytorch) / BSD-style |
| MuJoCo | CPU 物理、FK 数值对照、渲染 | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) / Apache-2.0 |
| MuJoCo Warp | GPU 物理 | [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) / Apache-2.0 |
| NVIDIA Warp | GPU kernel/runtime | [NVIDIA/warp](https://github.com/NVIDIA/warp) / Apache-2.0 |
| NumPy | 数值和数据交换 | [numpy/numpy](https://github.com/numpy/numpy) / BSD-3-Clause |
| Pillow | 录像标注与预览 | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) / HPND |
| tyro | 类型化 CLI | [brentyi/tyro](https://github.com/brentyi/tyro) / MIT |
| TensorBoard | 训练日志 | [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard) / Apache-2.0 |
| joblib (可选) | 可信 PKL 导入 | [joblib/joblib](https://github.com/joblib/joblib) / BSD-3-Clause |

FFmpeg 和 OSMesa 为用户安装的系统工具，各自适用其分发许可证。FFmpeg
实际许可证取决于构建选项；本包不分发 FFmpeg。

方法引用 [SuperMimic 论文](https://arxiv.org/abs/2610.28480)。
参考了 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
公开文档中的入口组织、性能口径和工程经验。本项目算法和框架独立实现，
未导入其代码或 mjlab/rsl-rl 框架。相似方法不意味着相同资产、观测、数据或成绩。

MuJoCo Warp 使用公开 Git commit
`875c4caf06d71b12b2036c327e7a07ce08a78b9a`。该 commit 的开发索引不参与
依赖解析，MuJoCo/Warp 从公开 PyPI 获取。完整版本与来源见 `uv.lock`。
