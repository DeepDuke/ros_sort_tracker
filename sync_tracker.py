#!/usr/bin/env python
from __future__ import print_function
import time
import rospy
import message_filters
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from generate_colors import get_colors


class Node:
    def __init__(self):
        self.colors = get_colors(80)
        self.bridge = CvBridge()
        self.color_image_sub = message_filters.Subscriber(
            '/camera/color/image_raw', Image)
        self.depth_image_sub = message_filters.Subscriber(
            '/camera/depth/image_rect_raw', Image)
        self.bboxes_sub = message_filters.Subscriber(
            '/darknet_ros/bounding_boxes', BoundingBoxes)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_image_sub, self.depth_image_sub, self.bboxes_sub], 10,
            1000000000)
        self.result_pub = rospy.Publisher("sync_tracker_result",
                                          Image,
                                          queue_size=1)

    def callback(self, color_image_msg, depth_image_msg, bboxes_msg):

        print(
            '[Color Image TimeStamp: {}]\t[Depth Image TimeStamp: {}]\t[Bboxes TimeStamp: {}]'
            .format(color_image_msg.header.stamp, depth_image_msg.header.stamp,
                    bboxes_msg.header.stamp))
        # convert color_image into color_cv_image
        try:
            color_cv_image = self.bridge.imgmsg_to_cv2(color_image_msg, 'bgr8')
        except CvBridgeError as error:
            print(error)
        # draw bboxes on color_cv_image
        bboxes = bboxes_msg.bounding_boxes
        for bbox in bboxes:
            cv2.rectangle(color_cv_image,
                          pt1=(bbox.xmin, bbox.ymin),
                          pt2=(bbox.xmax, bbox.ymax),
                          color=self.colors[bbox.id],
                          thickness=2)
            cv2.putText(color_cv_image,
                        bbox.Class,
                        org=(bbox.xmin, bbox.ymin - 5),
                        fontFace=1,
                        fontScale=1,
                        color=self.colors[bbox.id],
                        thickness=2,
                        lineType=cv2.LINE_AA)
        # cv2.imshow("ImageWindow", color_cv_image)
        # cv2.waitKey(10)
        try:
            self.result_pub.publish(self.bridge.cv2_to_imgmsg(color_cv_image, 'bgr8'))
        except CvBridgeError as error:
            print(error)
        # TODO: tracking bboxes

    def run(self):
        print("Enter sync_tracker's run function ...")
        self.sync.registerCallback(self.callback)


if __name__ == '__main__':
    rospy.init_node('sync_tracker_node', anonymous=True)
    rospy.loginfo('Starting sync_trakcer_node ...')
    sync_trakcer_node = Node()
    sync_trakcer_node.run()
    rospy.spin()

# def callback(depth_image_msg, bboxes_msg):
#     print('hello world!')
#     print('[Depth Image TimeStamp: {}]\t[Bboxes TimeStamp: {}]'.format(
#         depth_image_msg.header.stamp, bboxes_msg.header.stamp))

# if __name__ == '__main__':
#     rospy.init_node('sync_tracker_node', anonymous=True)
#     rospy.loginfo('Starting sync_trakcer_node ...')

#     depth_image_sub = message_filters.Subscriber(
#         '/camera/depth/image_rect_raw', Image)
#     bboxes_sub = message_filters.Subscriber('/darknet_ros/bounding_boxes',
#                                             BoundingBoxes)
#     sync = message_filters.ApproximateTimeSynchronizer(
#         [depth_image_sub, bboxes_sub], queue_size=10, slop=1000000000)
#     sync.registerCallback(callback)

#     rospy.spin()