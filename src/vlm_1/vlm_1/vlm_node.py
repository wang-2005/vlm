import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image as PILImage
import cv2
# 新增导入自定义消息
from robot_interfaces.msg import VlmTask

class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_node')
        self.bridge = CvBridge()

        # 状态锁：防止大模型推理时堆积过多的图像导致 OOM
        self.is_processing = False
        self.current_hint = "请识别图中的物体。"

        self.get_logger().info("🔄 Loading Qwen2-VL-2B Model (bfloat16)... This may take a while.")

        # ===== 加载 Qwen2-VL 模型 =====
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,

            trust_remote_code=True
        )
        self.get_logger().info("✅ Model loaded successfully!")

        # ===== ROS 接口 =====
        self.sub_task = self.create_subscription(VlmTask, '/vlm/task', self.task_callback, 1)

        self.pub_desc = self.create_publisher(String, '/vlm/description', 10)

    def hint_callback(self, msg):
        # 实时更新 YOLO 传过来的提示词
        self.current_hint = msg.data

    def task_callback(self, msg):
        # 如果模型正在推理，直接丢弃新任务
        if self.is_processing:
            return

        self.is_processing = True
        self.get_logger().info("🧠 Start VLM Inference...")

        try:
            # 1. 直接从打包消息中提取图像和提示词
            cv_img = self.bridge.imgmsg_to_cv2(msg.roi_image, desired_encoding='rgb8')
            pil_img = PILImage.fromarray(cv_img)
            #current_hint = msg.prompt_hint

            # 2. 构造 Prompt (使用验证型引导，降低幻觉)
            # 优化后的 Prompt 策略
            system_prompt = (
                f"{msg.prompt_hint} \n"
                "请仔细观察并回答：\n"
                "请用一句话描述它的颜色以及它当前所处的环境状态。"
                "要求：直接回答结果，不要猜测，不要推测图片之外的内容。"
                  )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ]

            # --- 下面的预处理、推理、解码逻辑与你原本的代码完全一致 ---
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,   # 限制输出长度，防止模型太啰嗦
                    do_sample=True,      # 开启随机采样，打破“死板”的回答
                    temperature=0.3,     # 温度越低越严谨，0.4 是一个平衡点
                    top_p=0.9            # 过滤掉不靠谱的词，提高回答质量
                     )

            generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
            result = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

            desc_msg = String()
            desc_msg.data = result
            self.pub_desc.publish(desc_msg)

            self.get_logger().info(f"💡 VLM Output: {result}")

        except Exception as e:
            self.get_logger().error(f"VLM Inference failed: {e}")
        finally:
            self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
