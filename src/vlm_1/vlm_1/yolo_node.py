import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
from ultralytics import YOLO

import json
import time

class YOLONode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        # ===== 参数配置 =====
        # 允许通过外部参数修改话题和模型
        self.declare_parameter('image_topic', '/color/color_raw')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.4)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value

        self.get_logger().info(f"Subscribing to: {image_topic}")
        self.get_logger().info(f"Loading model: {model_path}")

        # ===== 初始化工具 =====
        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        # ===== ROS 接口 =====
        # 订阅相机原始图像
        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        # 发布 JSON 格式的检测数据
        self.pub = self.create_publisher(
            String,
            '/yolo/detections',
            10
        )

        # 新增：发布带有检测框的可视化图像
        self.vis_pub = self.create_publisher(
            Image,
            '/yolo/visualization',
            10
        )

        # FPS 统计
        self.last_time = time.time()

    def image_callback(self, msg):
        try:
            # 将 ROS 图像消息转换为 OpenCV 格式
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # ===== YOLO 推理 =====
        # 运行推理并获取结果
        results = self.model(frame, verbose=False)[0]

        # --- 1. 生成可视化图像 ---
        # 使用 YOLO 内置方法绘制检测框和标签
        annotated_frame = results.plot()

        # 将处理后的图像转换回 ROS 消息并发布
        vis_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        vis_msg.header = msg.header  # 保持时间戳同步
        self.vis_pub.publish(vis_msg)

        # --- 2. 提取检测信息 (JSON 格式) ---
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        # 发布 JSON 字符串
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        output_data = {
            "timestamp": {"sec": sec, "nanosec": nanosec},
            "detections": detections
        }

        # 发布 JSON 字符串
        msg_out = String()
        msg_out.data = json.dumps(output_data)
        self.pub.publish(msg_out)
        # 打印检测日志
        if len(detections) > 0:
            self.get_logger().info(f"Detections: {len(detections)} objects found")

        # 计算 FPS
        now = time.time()
        fps = 1.0 / (now - self.last_time)
        self.last_time = now
        # self.get_logger().debug(f"YOLO FPS: {fps:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() # 确保脚本可以直接运行
