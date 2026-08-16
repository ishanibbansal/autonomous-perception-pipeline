import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageTester(Node):
    def __init__(self):
        super().__init__('image_tester')
        
        # Publisher for the camera topic
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Subscriber for the heatmap output
        self.subscription = self.create_subscription(
            Image,
            '/perception/bev_heatmap',
            self.listener_callback,
            10)
            
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Dynamically resolve the absolute path to the repository root
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_path = os.path.join(self.repo_root, 'test_frame.jpg')

    def timer_callback(self):
        img = cv2.imread(self.image_path)
        if img is None:
            self.get_logger().error(f"Could not read image: {self.image_path}")
            return
            
        # Ensure the image matches the TRT engine input dimensions (W=1920, H=1280)
        img_resized = cv2.resize(img, (1920, 1280))
        
        # Convert OpenCV BGR to ROS 2 Image message
        msg = self.bridge.cv2_to_imgmsg(img_resized, encoding="bgr8")
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published {self.image_path} to /camera/image_raw')

    def listener_callback(self, msg):
        self.get_logger().info('Received heatmap from TensorRT node!')
        
        # Convert ROS 2 Image message back to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Save dynamically to the repo root
        output_path = os.path.join(self.repo_root, 'heatmap_output.jpg')
        cv2.imwrite(output_path, cv_image)
        
        self.get_logger().info(f'Success! Saved heatmap to {output_path}')
        
        # Shut down after receiving one successful heatmap
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    tester = ImageTester()
    try:
        rclpy.spin(tester)
    except SystemExit:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()