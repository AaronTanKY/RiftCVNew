import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


class GNSSMsgRelayNode(Node):
    def __init__(self):
        super().__init__("gnss1_message_relay_node")
        self.subscription = self.create_subscription(
            NavSatFix, "/gx5/gnss2/fix", self.listener_callback, 10
        )
        self.subscription  # prevent unused variable warning

        self.publisher = self.create_publisher(NavSatFix, "/gx5/gnss2/fix", 10)

    def listener_callback(self, msg):
        self.get_logger().info('Received: "%s"' % msg)
        # Publishing the same message
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    relay_node = GNSSMsgRelayNode()
    rclpy.spin(relay_node)
    # Clean up
    relay_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
