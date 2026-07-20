# ACT 数据处理与训练

本文说明如何将 Astribot 录制的 HDF5 数据转换为 ACT 数据集，并使用转换后的数据训练 ACT Policy。

所有命令均在 RoboHub 仓库根目录执行：

```bash
cd /home/astribot/workspace/lcy/RoboHub
conda activate base
```

## 1. 安装依赖

安装 RoboHub 和 ACT 所需依赖：

```bash
pip install -e ".[act]"
```

如果需要使用 Weights & Biases：

```bash
pip install -e ".[act,act-wandb]"
```

## 2. 处理 Astribot 数据

### 2.1 输入数据要求

转换脚本接受一个 HDF5 文件，或包含多个 HDF5 文件的目录。每个文件需要包含：

```text
joints_dict/
├── joints_position_state      # (T, 25)
└── joints_position_command    # (T, 25)

images_dict/
├── head/
│   ├── rgb
│   └── rgb_size
├── left/
│   ├── rgb
│   └── rgb_size
└── right/
    ├── rgb
    └── rgb_size
```

脚本会完成以下转换：

- 将 Astribot 录制关节顺序转换为 RoboHub 的 25 维关节顺序。
- 将 `head`、`left`、`right` 分别映射为
  `cam_high`、`cam_left_wrist`、`cam_right_wrist`。
- 将三路 RGB 图像缩放到 `640x480`。
- 生成 ACT 所需的 `qpos`、`action` 和图像数据。
- 逐帧解码和写入图像，避免一次加载整段视频。

### 2.2 转换单个文件

```bash
python -m robohub.policies.act.data.convert_astribot \
  --input data/robot/astribot/task_episode_0.hdf5 \
  --output-dir data/act/my_task \
  --task my_task
```

也可以使用位置参数：

```bash
python -m robohub.policies.act.data.convert_astribot \
  data/robot/astribot/task_episode_0.hdf5 \
  data/act/my_task \
  --task my_task
```

### 2.3 转换整个目录

目录中的 `.h5` 和 `.hdf5` 文件会按文件名自然排序，并依次输出为
`episode_0.hdf5`、`episode_1.hdf5`：

```bash
python -m robohub.policies.act.data.convert_astribot \
  --input data/robot/astribot/hdf5_output_my_task \
  --output-dir data/act/my_task \
  --task my_task
```

### 2.4 下采样

使用 `--downsample N` 每隔 N 帧保留一帧。例如将约 30 Hz 的数据降至约 15 Hz：

```bash
python -m robohub.policies.act.data.convert_astribot \
  --input data/robot/astribot/hdf5_output_my_task \
  --output-dir data/act/my_task \
  --task my_task \
  --downsample 2
```

如果输出文件已经存在，脚本默认拒绝覆盖。确认需要重新生成时使用：

```bash
python -m robohub.policies.act.data.convert_astribot \
  --input data/robot/astribot/hdf5_output_my_task \
  --output-dir data/act/my_task \
  --task my_task \
  --overwrite
```

### 2.5 输出结构

转换完成后会得到：

```text
data/act/my_task/
├── dataset.yaml
├── episode_0.hdf5
├── episode_1.hdf5
└── ...
```

每个 episode 的内容为：

```text
/action                                      # (T, 25), float32
/observations/qpos                           # (T, 25), float32
/observations/images/cam_high                # (T, 480, 640, 3), uint8
/observations/images/cam_left_wrist          # (T, 480, 640, 3), uint8
/observations/images/cam_right_wrist         # (T, 480, 640, 3), uint8
```

`dataset.yaml` 记录数据目录、episode 数量、每段长度、相机名称、维度和下采样倍数，训练脚本会自动读取并校验。

## 3. 训练 ACT

### 3.1 创建训练配置

默认配置位于：

```text
src/robohub/policies/act/config/default.yaml
```

建议为每个任务单独创建配置，例如 `configs/act/my_task.yaml`：

```yaml
dataset:
  path: data/act/my_task

training:
  batch_size: 16
  epochs: 1000
  device: cuda:0
  workers: 1
  save_freq: 100
  ckpt_dir: checkpoints/act/my_task

model:
  chunk_size: 30
  hidden_dim: 512
  dim_feedforward: 3200
  kl_weight: 10.0
```

自定义配置会与默认配置递归合并，因此只需要填写需要覆盖的字段。

常用配置项：

- `dataset.path`：转换后的 ACT 数据集目录。
- `training.device`：训练设备，例如 `cuda:0` 或 `cpu`。
- `training.batch_size`：训练和验证 batch size。
- `training.epochs`：训练轮数。
- `training.workers`：DataLoader worker 数量。
- `training.downsample`：从已转换数据中再次按间隔采样 action chunk。
- `training.arm_delay`：动作延迟补偿帧数。
- `training.save_freq`：周期 checkpoint 的保存间隔。
- `training.ckpt_dir`：训练产物目录。
- `training.pretrain`：可选的预训练 checkpoint 路径。
- `model.chunk_size`：每次预测的动作序列长度。
- `model.kl_weight`：CVAE KL loss 权重。

ACT 的 `state_dim` 和 `action_dim` 固定为 25，相机顺序固定为三路 Astribot 相机，不应在任务配置中修改。

### 3.2 启动训练

```bash
python -m robohub.policies.act.scripts.train \
  --config configs/act/my_task.yaml
```

也可以直接通过命令行覆盖常用参数：

```bash
python -m robohub.policies.act.scripts.train \
  --config configs/act/my_task.yaml \
  --dataset-dir data/act/my_task \
  --ckpt-dir checkpoints/act/my_task \
  --device cuda:0 \
  --epochs 1000 \
  --batch-size 16
```

不提供 `--config` 时会使用默认配置：

```bash
python -m robohub.policies.act.scripts.train \
  --dataset-dir data/act/my_task \
  --ckpt-dir checkpoints/act/my_task \
  --device cuda:0
```

### 3.3 使用 Weights & Biases

默认不启用 wandb。安装 `act-wandb` 依赖后，可以通过以下命令启用：

```bash
python -m robohub.policies.act.scripts.train \
  --config configs/act/my_task.yaml \
  --wandb
```

使用 `--no-wandb` 可以覆盖配置并关闭 wandb。

### 3.4 训练输出

训练目录中会生成：

```text
checkpoints/act/my_task/
├── config.yaml
├── dataset_stats.pkl
├── policy_best.ckpt
├── policy_last.ckpt
├── policy_epoch_100.ckpt
├── policy_epoch_200.ckpt
├── train_val_l1.png
├── train_val_kl.png
└── train_val_loss.png
```

- `config.yaml`：训练时使用的完整配置，部署时用于重建模型。
- `dataset_stats.pkl`：qpos 和 action 的归一化统计。
- `policy_best.ckpt`：验证集 loss 最低的模型。
- `policy_last.ckpt`：最后一个 epoch 的模型。
- `policy_epoch_*.ckpt`：按 `save_freq` 保存的周期 checkpoint。

部署 `ACTPolicy` 时必须保留 `config.yaml`、`dataset_stats.pkl` 和 checkpoint 文件，并将它们放在同一目录。

## 4. 快速流程

```bash
# 1. 转换数据
python -m robohub.policies.act.data.convert_astribot \
  --input data/robot/astribot/hdf5_output_my_task \
  --output-dir data/act/my_task \
  --task my_task \
  --downsample 2

# 2. 训练
python -m robohub.policies.act.scripts.train \
  --dataset-dir data/act/my_task \
  --ckpt-dir checkpoints/act/my_task \
  --device cuda:0 \
  --epochs 1000 \
  --batch-size 16
```
