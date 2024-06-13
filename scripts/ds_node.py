#!/usr/bin/env python
import rospy
import cv_bridge
from math import cos,sin

import numpy as np

import cv2

from sensor_msgs.msg import Image,LaserScan
from geometry_msgs.msg import Twist
from Control import PID

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

    self.desire_id = 0  # need change

    ## set pid params
    kp_x = rospy.get_param('~kp_x',1.0)   
    ki_x = rospy.get_param('~ki_x',1.0)   
    kd_x = rospy.get_param('~kd_x',1.0)   

    kp_y = rospy.get_param('~kp_y',1.0)   
    ki_y = rospy.get_param('~ki_y',1.0)   
    kd_y = rospy.get_param('~kd_y',1.0)   
    
    self.angref = rospy.get_param('~angref',False)   
    
    ## init models
    track_model_path = rospy.get_param('~track_model_path',"$(find smart_pkg)/config/")   
    self.track_model = DeepSort(model_path=track_model_path,max_age=2)

    detect_model_path = rospy.get_param('~detect_model_path',"$(find smart_pkg)/config/")   
    self.detect_model = YOLO(detect_model_path)

    # self.model.

    ## pid controller set
    self.controller = PID()    
    self.controller.set_input(kp_x,ki_x,kd_x,kp_y,ki_y,kd_y)

    ## algorithm output
    self.twist = Twist()


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


  def deepsort(self,img):
    yolo_output = self.detect_model(img,device='cpu',classes = [0],conf = 0.8)
    ## get bounding box info
    for object in yolo_output:
      # Bounding Box of Object Detection
      boxes = object.boxes  
      
      # Connvert Class of Objects Tensor to List
      class_list = boxes.cls.tolist()  
      
      # Coordinate of Bounding Box
      xyxy = boxes.xyxy
      xywh = boxes.xywh
      
      # Confidence Score of Bounding Box  
      conf = boxes.conf

      conf = conf.detach().cpu().numpy()
      xyxy = xyxy.detach().cpu().numpy()
      xywh = xywh.detach().cpu().numpy()
      xywh_np = np.array(xywh, dtype = float)
      tracks = self.track_model.update(xywh_np, conf, img)
      for track in self.track_model.tracker.tracks:
        self.id_dict[track.track_id] = track

  def camera_cb(self,msg):
    ## convert image to gray cv
    gray = self.bridge.imgmsg_to_cv2(msg,desired_encoding="mono8")
    self.deepsort(gray)

  def lidar_cb(self,msg):
    #TODO: check angle reference, check if tracking id done well, Gain have to be tune

    ranges = msg.ranges
    angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

    if self.track_model.tracker.tracks == []:
      self.twist.linear.x = 0.0
      self.twist.angular.z = 0.0
      self.cmd_pub.publish(self.twist)

    # Get the LiDAR data within the bounding box region

    # track only A object
    track = self.id_dict[self.desire_id]
    
    x1, y1, x2, y2 = track.to_tlbr()

    k1 = int((250 / 640 * x1) + 850)
    k2 = int((250 / 640 * x2) + 850)

    tmp = 0
    sums = 0.
        
    valid = msg.ranges[k1:k2]
    boxrange = len(valid)

    if not boxrange:
      rospy.loginfo('box not detected')
      return  #case out of camera angle

    sums = np.sum([val for val in valid if val != 0.0])
    avg = sums / len(valid) if len(valid) > 0 else 0.0  #linear distance


    box_angle_start = angles[k1]
    box_angle_end = angles[k2]
    box_center_angle = (box_angle_start + box_angle_end) / 2

    x = avg * cos(box_center_angle)      
    y = avg * sin(box_center_angle)

    rospy.loginfo('*** !debug! angle start: {} '.format(box_angle_start))
    rospy.loginfo('*** !debug! angle end: {} '.format(box_angle_end))
    rospy.loginfo('*** !debug! anglemin: {} '.format(box_center_angle))

    if self.angref:
      self.twist = self.controller.get_twist_angular(avg,box_center_angle)  # publish lin/ang val
    else:
      self.twist = self.controller.get_twist(x,y) # publish x/y val


    self.cmd_pub.publish(self.twist)


if __name__ == "__main__":
  name_node = Deep_Sort_Node()
  rospy.spin()