# SuperMimic-Q1

基于 PyTorch、MuJoCo 和 MuJoCo Warp 的 Q1 V2 全身动作跟踪训练。
支持单卡、多卡、从零训练、检查点续训、指定动作播放和整段录像。

方法参考 [SuperMimic 论文](https://arxiv.org/abs/2610.28480)，代码独立实现。
当前为研究版本，动作跟踪质量仍在验证。

## 安装

Linux、Python 3.13；GPU 训练需要 NVIDIA CUDA。

```bash
git clone https://code.iqiyuan-robot.com/sw/SuperMimic-Q1.git
cd SuperMimic-Q1
uv sync --python 3.13 --group dev
uv run --no-sync supermimic --help
```

以下命令在仓库根目录执行。首次 `uv sync` 会下载依赖，完成后再启动训练。
后续命令使用 `--no-sync` 复用已安装的环境。
启动器默认将编译缓存放在本机 `/tmp`；可用 `--compiler-cache-dir` 指定目录。

录像另需安装 FFmpeg；CPU 无屏渲染需要 OSMesa：

```bash
sudo apt-get install ffmpeg libosmesa6
```

## 资产与数据

自行准备 Q1 V2 XML、引用的 meshes 和 50 Hz Q1 动作。
源码不附带机器人资产、数据集或训练权重。

```text
assets/q1_v2/xml/q1_v2_mc.xml
assets/q1_v2/meshes/...
data/q1_motion_library_filtered/<source>/*.npz
data/q1_motion_library/lafan1/*.npz
```

训练 NPZ 字段：

| 字段 | 要求 |
|---|---|
| `qpos` | `[T,29]`，根位置、WXYZ 四元数、22 关节角 |
| `robot_joint_names` | 22 个 Q1 关节名称 |
| `robot_name` | `q1_v2` |
| `root_quat_order` | `wxyz` |
| `fps` | `50.0` |

筛选动作：

```bash
uv run --no-sync supermimic filter   --xml assets/q1_v2/xml/q1_v2_mc.xml   --motion-dir data/q1_motion_library --output-dir logs/data_screening   --workers 8 --sample-hz 10
```

## 八卡从零训练

```bash
cp configs/q1-fresh.example.toml configs/local-train.toml
# 编辑 robot_xml、motion_dir 和新的 output_dir
uv run --no-sync supermimic launch --config configs/local-train.toml --gpus 8 --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync supermimic launch --config configs/local-train.toml --gpus 8
```

该配置从第 0 轮初始化，训练 250000 轮，开启编译、参考缓存和整段 CUDA graph。
环境数和 batch 都是全局总量，按 GPU 数量均分。

| 配置 | `num_envs` | `model_batch_size` | `policy_batch_size` |
|---|---:|---:|---:|
| 八卡示例 | 8192 | 4096 | 4096 |
| 单卡起步 | 1024 | 512 | 512 |

## 单卡从零训练

```bash
cp configs/q1-fresh-single.example.toml configs/local-train-single.toml
# 编辑 robot_xml、motion_dir 和新的 output_dir
uv run --no-sync supermimic launch --config configs/local-train-single.toml --gpus 1 --dry-run
CUDA_VISIBLE_DEVICES=0 uv run --no-sync supermimic launch --config configs/local-train-single.toml --gpus 1
```

单卡配置使用 1024 环境、model/policy batch 各 512，默认输出到
`logs/q1_fresh_single`。单卡和八卡使用同一训练实现及随机域。

建议在独立 tmux 终端运行。训练配置、逐轮日志、TensorBoard 和检查点保存在
`output_dir`。查看训练记录：

```bash
uv run --no-sync supermimic monitor --run-root logs/q1_fresh
```

## 续训

复制训练配置，保留资产、数据和训练参数，改用新的输出目录并加入：

```toml
resume = "../logs/q1_fresh/checkpoint_5000.pt"
output_dir = "../logs/q1_resume"
iterations = 250000
```

使用相同 GPU 数量启动。搬运检查点时，保留主 `.pt` 及其对应的完整 `.ranks`
目录。`iterations` 是累计训练目标。

## 选择动作播放

```bash
uv run --no-sync supermimic play   --checkpoint logs/q1_fresh/checkpoint_5000.pt   --motion data/q1_motion_library/lafan1/walk__clip000.npz   --viewer on
```

省略 `--motion` 时，可用 `--motion-index 0` 选择训练动作。
`--speed 0.5` 半速，`--loops 3` 循环；无显示器时保存轨迹：

```bash
uv run --no-sync supermimic play   --checkpoint logs/q1_fresh/checkpoint_5000.pt   --motion data/q1_motion_library/lafan1/walk__clip000.npz   --viewer off --output logs/play_selected.npz
```

## 整段录像

录像使用保留全部分段的原始 LAFAN 目录，输出策略与参考并排的 MP4。

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa uv run --no-sync supermimic record   --checkpoint logs/q1_fresh/checkpoint_5000.pt   --motion-dir data/q1_motion_library/lafan1   --output-dir logs/video_5000 --seed 0
```

每 5000 轮自动抽取一整条录像：

```bash
uv run --no-sync supermimic watch-videos   --run-dir logs/q1_fresh   --motion-dir data/q1_motion_library/lafan1 --interval 5000
```

使用 NVIDIA GPU 无屏绘图时，可为 watcher 添加 `--render-device 0`。

## 开发

```bash
make check
make test
uv build
```

其他导入、评测和参数入口见 `uv run --no-sync supermimic --help`。
代码采用 [MIT License](LICENSE)，依赖与来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
