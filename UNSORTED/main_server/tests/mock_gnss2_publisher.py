import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import random


class MockLongitudePublisher(Node):
    def __init__(self):
        super().__init__("sim_gnss2_publisher")
        self.publisher_ = self.create_publisher(NavSatFix, "/gx5/gnss2/fix", 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.fixed_latitude = 40.7128  # Same latitude as the fixed publisher
        self.longitude_offset = 0.01  # Example offset

    def timer_callback(self):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "offset_gps_frame"
        msg.latitude = self.fixed_latitude
        msg.longitude = -74.0060 - self.longitude_offset
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing offset GPS: {msg}")


def main(args=None):
    rclpy.init(args=args)
    offset_longitude_publisher = MockLongitudePublisher()
    rclpy.spin(offset_longitude_publisher)
    offset_longitude_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
