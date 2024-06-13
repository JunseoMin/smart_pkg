#!/usr/bin/env python3
import rospy

from Control import PID
from geometry_msgs.msg import Twist

def main(args=None):
    rospy.init_node('test_node')
    my_pid = PID(False)
    # p control example
    
    cmd_pub = rospy.Publisher("/turtle1/cmd_vel", Twist, queue_size=10)

    my_pid.set_input(1,0,1,1,0,1)    
    twist = my_pid.get_twist(-100,-100)
    cnt = 0
    while(not rospy.is_shutdown()):
        if not cnt % 10000:
            print("publishing cmd_vel ...")
        cmd_pub.publish(twist)
        cnt += 1

    my_pid.get_input()
    rospy.spin()


if __name__ == '__main__':
    main()
