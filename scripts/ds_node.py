#!/usr/bin/env python
import rospy
import cv_bridge
import sys
import os

from math import cos,sin

import numpy as np

import cv2
sys.path.append("/home/nvidia/catkin_ws/src/smart_pkg/scripts/")

from sensor_msgs.msg import Image,LaserScan
from geometry_msgs.msg import Twist
from Control import PID

sys.path.append("/home/nvidia/catkin_ws/src/smart_pkg/")
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

from ultralytics import YOLO


class Deep_Sort_Node:
  def __init__(self):
    rospy.init_node("name_node")
    rospy.loginfo("Starting Deep_Sort_Node as name_node.")

    self.img_subs = rospy.Subscriber("/usb_cam/image_raw", Image, self.camera_cb)
    self.lidar_subs = rospy.Subscriber("/scan", LaserScan, self.lidar_cb)
    
    self.cmd_pub  = rospy.Publisher("/cmd_vel", Twist, queue_size=1000)
    
    self.bridge = cv_bridge.CvBridge()
    
    self.id_dict = {}

    self.desire_id = 1  # need change

    ## set pid params
    kp_x = rospy.get_param('~kp_x',1.0)   
    ki_x = rospy.get_param('~ki_x',1.0)   
    kd_x = rospy.get_param('~kd_x',1.0)   

    kp_y = rospy.get_param('~kp_y',1.0)   
    ki_y = rospy.get_param('~ki_y',1.0)   
    kd_y = rospy.get_param('~kd_y',1.0)   
    
    self.angref = True   
    self.control_period = rospy.get_param('~control_period',1.0)   
    
    ## init models
    track_model_path = rospy.get_param('~track_model_path',"")   
    deep_age = rospy.get_param('~deep_age',3)   

    self.track_model = DeepSort(model_path=track_model_path,max_age=deep_age)

    detect_model_path = rospy.get_param('~detect_model_path',"")   
    self.detect_model = YOLO("yolov8n.pt")

    # self.model.

    ## pid controller set
    self.controller = PID()    
    self.controller.set_input(kp_x,ki_x,kd_x,kp_y,ki_y,kd_y)

    ## algorithm output
    self.twist = Twist()

    self._dict = {}

    rospy.loginfo('-----------------------')
    rospy.loginfo('===== Gain set =====')
    rospy.loginfo('Kp_x : {}'.format(kp_x))
    rospy.loginfo('Ki_x : {}'.format(ki_x))
    rospy.loginfo('Kd_x : {}'.format(kd_x))
    rospy.loginfo('Kp_y : {}'.format(kp_y))
    rospy.loginfo('Ki_y : {}'.format(ki_y))
    rospy.loginfo('Kd_y : {}'.format(kd_y))
    rospy.loginfo('control mode: PID')
    rospy.loginfo('-----------------------')
    rospy.loginfo('-----------------------')


    #calib param
    #rvec
    self.rvec = np.array([[-0.60134519],[-1.86203176],[-1.86509407]],dtype="double")
    self.tvec = np.array([[1.81745753],[0.00457257],[-2.86750681]],dtype="double")
    self.cammat = np.array([[1.05219633e+3,1.28607498e+01,6.9360786e+02],
                            [0.0,1.05627925e+03,3.66707863e+02,],
                            [0,0,1]],dtype="double")

  def get_lidar_points_in_bbox(self,scan, bounding_box):

    angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
    
    # Calculate FOV limits in radians from degrees
    fov_deg = 75.0
    fov_rad = np.radians(fov_deg)
    fov_min = -fov_rad / 2.0
    fov_max = fov_rad / 2.0

    points_lidar = []
    for r, a in zip(scan.ranges, angles):
        if r < scan.range_max and r > scan.range_min:
            if a >= fov_min and a <= fov_max:
                points_lidar.append((r * np.cos(a), r * np.sin(a), 0))

    points_lidar = np.array(points_lidar)

    points_img, _ = cv2.projectPoints(points_lidar, self.rvec, self.tvec, self.cammat, self.dist_coeffs)
    points_img = points_img.reshape(-1, 2)
    
    x_min, y_min, x_max, y_max = bounding_box
    lidar_points_in_bbox = []

    for (x, y), (x_lidar, y_lidar) in zip(points_img, points_lidar[:, :2]):
      if x_min <= x <= x_max and y_min <= y <= y_max:
        distance = np.sqrt(x_lidar**2 + y_lidar**2)
        angle = np.arctan2(y_lidar, x_lidar)
        lidar_points_in_bbox.append((x_lidar, y_lidar, distance, angle))

    return lidar_points_in_bbox

  def deepsort(self,img):
    yolo_output = self.detect_model(img,device='cpu',classes = [0],conf = 0.8)
    ## get bounding box info

    cv2.imshow("yoloy",img)
    cv2.waitKey(1)

    for object in yolo_output:
      # Bounding Box of Object Detection

      boxes = object.boxes  
      
      # Connvert Class of Objects Tensor to List
      # class_list = boxes.cls.tolist()  
      
      # Coordinate of Bounding Box
      # xyxy = boxes.xyxy
      xywh = boxes.xywh
      
      # Confidence Score of Bounding Box  
      conf = boxes.conf
      xywh_np = np.array(xywh, dtype = float)
      tracks = self.track_model.update(xywh_np, conf, img)

      try:      
        x, y, width, height = xywh[0][0], xywh[0][1], xywh[0][2], xywh[0][3] 

        self.start_x = int(x - (width // 2))
        self.start_y = int(y - (height // 2))
        self.end_x = int(x + (width // 2))
        self.end_y = int(y + (width // 2))
        cv2.rectangle(img, (self.start_x, self.start_y), (self.end_x, self.end_y), color = (0, 0, 255), thickness = 2)

        cv2.imshow("detect",img)
        cv2.waitKey(1)
      except:
        rospy.loginfo("not detected")
      
      for track in self.track_model.tracker.tracks:
        rospy.loginfo("dict added id: {}".format(track.track_id))
        # print("id added {}".format(track.track_id))11
        self.id_dict[track.track_id] = track


  def camera_cb(self,msg):
    ## convert image to gray cv
    # print("!!!!imgcb!!!!!")
    # try:
    gray = self.bridge.imgmsg_to_cv2(msg)
    # rospy.loginfo("Image converted")
    self.deepsort(gray)
    # except Exception as e:
      # rospy.logerr(f"Error in camera_cb: {e}")

  def lidar_cb(self,msg):
    #TODO: check angle reference, check if tracking id done well, Gain have to be tune

    ranges = msg.ranges
    angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

    if self.track_model.tracker.tracks == []:
      self.twist.linear.x = 0.0
      self.twist.angular.z = 0.0
      self.cmd_pub.publish(self.twist)

    # Get the LiDAR data within the bounding box region

    print(self.id_dict)
    # track only A object
    track = self.id_dict.get(self.desire_id,False)

    if not track:
      rospy.logwarn('box not detected')
      return
    
    x1, y1, x2, y2 = track.to_tlbr()
    bbox = track.to_tlbr()
    
    boxlidar = self.get_lidar_points_in_bbox(bbox)
    
    mean_dis = 0.0
    mean_ang = 0.0
    tmp = 1

    for x, y, distance, angle in boxlidar:
      rospy.loginfo(f"Point in bbox - Distance: {distance:.2f} m, Angle: {np.degrees(angle):.2f} degrees")
      mean_dis += distance
      mean_ang += angle
      tmp+=1
    
    mean_dis/=tmp
    mean_ang/=tmp

    x = mean_dis * cos(mean_ang)      
    y = mean_dis * sin(mean_ang)

    rospy.loginfo('*** !debug! angle start: {} '.format(mean_dis))
    rospy.loginfo('*** !debug! angle end: {} '.format(mean_ang))
    
    self.twist = self.controller.get_twist_angular(mean_dis,mean_ang)  # publish lin/ang val
    rospy.sleep(self.control_period)  # control period

    # if self.angref:
    # else:
    #   self.twist = self.controller.get_twist(x,y) # publish x/y val
    #   rospy.sleep(self.control_period)  # control period


    self.cmd_pub.publish(self.twist)


if __name__ == "__main__":
  name_node = Deep_Sort_Node()
  rospy.spin()