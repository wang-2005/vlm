# YOLO + Qwen2-VL 机器人视觉理解原型

这是一个 ROS 2 Humble 的视觉感知原型：相机图像先经过 YOLO 目标检测，再按检测框裁剪出感兴趣区域（ROI），最后交给 Qwen2-VL-2B-Instruct 生成简短的中文描述。它输出目标的颜色和场景状态描述，不包含导航决策或机械控制。

## 数据流

```text
Berxel RGB 图像 (/color/color_raw)
       ├──> YOLO 检测 ──> /yolo/detections (类别、置信度、框、原图时间戳)
       └──> ROI 节点的 10 帧缓存
                         │ 按时间戳匹配检测结果与原图
                         ▼
             扩边裁剪 + 等比例缩放/填充
                         │
                         ▼
             /vlm/task (VlmTask: ROI 图像 + 提示词)
                         │
                         ▼
             Qwen2-VL 推理 ──> /vlm/description
```

三个节点的职责如下：

| 节点 | 输入 | 输出 | 关键处理 |
| --- | --- | --- | --- |
| `yolo_node` | `/color/color_raw` | `/yolo/detections`、`/yolo/visualization` | YOLOv8 检测；阈值、模型路径、图像话题可配置 |
| `roi_node` | 原图和检测结果 | `/vlm/task`、`/vlm/debug_roi` | 时间戳匹配、检测框扩边、letterbox 缩放 |
| `vlm_node` | `/vlm/task` | `/vlm/description` | 根据目标类别生成提示词，调用 Qwen2-VL 描述 ROI |

自定义消息 [`VlmTask.msg`](src/robot_interfaces/msg/VlmTask.msg) 把图像和提示词放在同一条 ROS 消息中，避免两个话题再同步。ROI 节点只处理每帧的第一个检测目标；找不到原图对应帧时丢弃该检测。VLM 节点在推理期间跳过新任务，避免图像任务无限排队。

## 环境和依赖

- Ubuntu 与 ROS 2 Humble（需要 `rclpy`、`sensor_msgs`、`std_msgs`、`cv_bridge`）。
- Python 3.10，PyTorch；请先按本机 CPU/CUDA 环境安装合适的 PyTorch。
- 其余 Python 依赖见 [`requirements.txt`](requirements.txt)。本机开发环境使用过 `torch 2.7.1+cu118`、`transformers 4.46.0` 和 `ultralytics 8.4.41`。
- 可发布彩色图像的 RGB 相机；默认话题是 Berxel 驱动的 `/color/color_raw`。`yolo_node` 的话题名可通过 ROS 参数调整，ROI 节点目前使用固定话题名。
- 首次运行会获取 `yolov8n.pt` 和 `Qwen/Qwen2-VL-2B-Instruct` 模型权重。权重不包含在此仓库中。

## 构建与运行

在包含本仓库的目录中构建消息与 Python 包：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select robot_interfaces vlm_1
source install/setup.bash
```

先在所用的 Python 虚拟环境中安装依赖（只需安装一次）：

```bash
source /path/to/your/venv/bin/activate
pip install -r requirements.txt
```

启动相机驱动并确认 `/color/color_raw` 有图像。随后打开三个终端；每个终端都要先载入 ROS 2、工作空间和安装了依赖的虚拟环境：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
source /path/to/your/venv/bin/activate
```

```bash
python3 src/vlm_1/vlm_1/yolo_node.py
python3 src/vlm_1/vlm_1/ROI_node.py
python3 src/vlm_1/vlm_1/vlm_node.py
```

上面三条 `python3` 命令应分别在三个终端运行。调试时可查看 `/yolo/visualization`、`/vlm/debug_roi` 和 `/vlm/description`：

```bash
ros2 topic echo /vlm/description
```

## 当前范围

这是工程原型，不是训练或改进 YOLO、Qwen2-VL 模型的项目。ROS 2 推理节点以 `bfloat16` 加载 Qwen2-VL，**没有启用 4-bit 量化**。项目目前没有公开的准确率、延迟或端到端评测，因此不应据此宣称精度提升或实时性。后续可补充多目标任务队列、丢帧统计、延迟测量和固定数据集评估。
