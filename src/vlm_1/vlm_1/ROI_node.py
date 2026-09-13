import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import json
import cv2
import numpy as np
from collections import deque
# 新增导入自定义消息
from robot_interfaces.msg import VlmTask

class ROINode(Node):
    def __init__(self):
        super().__init__('roi_node')
        self.bridge = CvBridge()

        # ===== 参数配置 =====
        self.declare_parameter('target_size', [448, 448]) # Qwen2-VL 在 448 甚至更大尺寸下表现更佳
        self.declare_parameter('padding_ratio', 0.15)    # 增加到 25% 的背景，有助于模型识别物体处于“办公桌”
        self.target_size = tuple(self.get_parameter('target_size').get_parameter_value().integer_array_value)
        self.padding_ratio = self.get_parameter('padding_ratio').get_parameter_value().double_value

        # ===== 图像缓存（解决 YOLO 延迟导致的异步问题）=====
        # 使用队列存储最近的 10 帧图像，确保裁剪时对应的是推理时的那一帧
        self.image_buffer = deque(maxlen=10)

        # ===== ROS 接口 =====
        self.sub_img = self.create_subscription(
            Image, '/color/color_raw', self.img_callback, 10
        )
        self.sub_det = self.create_subscription(
            String, '/yolo/detections', self.det_callback, 10
        )

        # 发布 ROI 图像和对应的“提示词”消息
        self.vlm_task_pub = self.create_publisher(VlmTask, '/vlm/task', 10)
        self.debug_roi_pub = self.create_publisher(Image, '/vlm/debug_roi', 10)
        self.get_logger().info("Optimized ROI Node Started")

    def img_callback(self, msg):
        # 将图像及其时间戳存入缓存
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.image_buffer.append((msg.header.stamp, cv_img))

    def letterbox(self, img, new_shape, color=(0, 0, 0)):
        """等比例缩放并填充，防止物体变形"""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

    def det_callback(self, msg):
        if not self.image_buffer:
            return

        try:
            data = json.loads(msg.data)
            target_sec = data["timestamp"]["sec"]
            target_nano = data["timestamp"]["nanosec"]
            detections = data["detections"]
        except (KeyError, json.JSONDecodeError):
            return

        if not detections:
            return

        # --- 核心优化：精准匹配时间戳 ---
        target_img = None
        for stamp, img in list(self.image_buffer): # 遍历双端队列
            if stamp.sec == target_sec and stamp.nanosec == target_nano:
                target_img = img
                break

        # 如果缓存中找不到对应的帧（可能延迟太高被挤出去了），直接丢弃
        if target_img is None:
            self.get_logger().warn("Frame dropped: YOLO detection too slow, image out of buffer.")
            return

        # 遍历所有检测到的目标
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["class_name"]

            # 扩充边缘（Context Padding）
            w_box, h_box = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - w_box * self.padding_ratio))
            y1 = max(0, int(y1 - h_box * self.padding_ratio))
            x2 = min(target_img.shape[1], int(x2 + w_box * self.padding_ratio))
            y2 = min(target_img.shape[0], int(y2 + h_box * self.padding_ratio))

            # 直接裁剪，抛弃 Letterbox
            crop = target_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            processed_roi = self.letterbox(crop, self.target_size) # 缩放并填充

            # --- 生成自定义的打包消息 ---
            task_msg = VlmTask()

            # 填入头部时间戳
            task_msg.header.stamp.sec = target_sec
            task_msg.header.stamp.nanosec = target_nano

            # 填入图像
            task_msg.roi_image = self.bridge.cv2_to_imgmsg(processed_roi, 'bgr8')
            task_msg.roi_image.header.stamp = task_msg.header.stamp

            # 填入提示词
            task_msg.prompt_hint = f"当前目标物体：{label}。"

            # 发布给 VLM
            self.vlm_task_pub.publish(task_msg)

            self.debug_roi_pub.publish(task_msg.roi_image) # 这样 RViz 订阅 /vlm/debug_roi 就能看到了

            self.get_logger().info(f"Published task for: {label}")
            break # 示例仅处理第一个目标
def main(args=None):
    rclpy.init(args=args)
    node = ROINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
