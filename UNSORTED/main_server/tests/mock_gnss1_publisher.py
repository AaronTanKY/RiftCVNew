import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


class MockFrontLatLongPublisher(Node):
    def __init__(self):
        super().__init__("sim_gnss1_publisher")
        self.publisher_ = self.create_publisher(NavSatFix, "/gx5/gnss1/fix", 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.fixed_latitude = 40.7128  # Example latitude
        self.fixed_longitude = -74.0060  # Example longitude

    def timer_callback(self):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fixed_gps_frame"
        msg.latitude = self.fixed_latitude
        msg.longitude = self.fixed_longitude
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing fixed GPS: {msg}")


def main(args=None):
    rclpy.init(args=args)
    fixed_latlong_publisher = MockFrontLatLongPublisher()
    rclpy.spin(fixed_latlong_publisher)
    fixed_latlong_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
