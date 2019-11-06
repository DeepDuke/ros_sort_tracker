#!/usr/bin/env python
from __future__ import print_function
import rospy
import message_filters
import time
import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes


def iou(bb_test, bb_gt):
    """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array(
            [x[0] - w / 2., x[1] - h / 2., x[0] + w / 2.,
             x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([
            x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score
        ]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self, bbox):
        """
    Initialises a tracker using initial bounding box.
    """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[
            4:,
            4:] *= 1000.  #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        # tracker positions and velocities
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.update_pos_vel_time = time.time()
        self.first_detected = True

    def update(self, bbox):
        """
    Updates the state vector with observed bbox.
    """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
    Advances the state vector and returns the predicted bounding box estimate.
    """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
    Returns the current bounding box estimate.
    """
        return convert_x_to_bbox(self.kf.x)

    def update_pos_and_vel(self, pos_vec):
        """
        @param pos_vec: np.array([x, y, z])
        """
        last_pos = self.pos
        self.pos = pos_vec
        new_time = time.time()
        time_delta = new_time - self.update_pos_vel_time
        self.update_pos_vel_time = new_time
        if self.first_detected == True:
            self.first_detected = False
        else:
            self.vel = (self.pos - last_pos) / time_delta


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
    if (len(trackers) == 0):
        return np.empty(
            (0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5),
                                                                     dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(
        unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks)

        #update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if ((trk.time_since_update < 1)
                    and (trk.hit_streak >= self.min_hits
                         or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(
                    1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


MOT = Sort()
bridge = CvBridge()
result_image_pub = rospy.Publisher('/tracker/image_result',
                                   Image,
                                   queue_size=1)
obstacles_pub = rospy.Publisher('/tracker/obstacles_information',
                                String,
                                queue_size=1)


def callback(color_image_msg, depth_image_msg, bboxes_msg):
    rospy.loginfo('Enter callback function ...')
    # convert color_image_msg into color_cv_image
    try:
        color_cv_image = bridge.imgmsg_to_cv2(color_image_msg, 'bgr8')
    except CvBridgeError as error:
        print(error)
    # convert depth_image_msg into depth_cv_image
    try:
        depth_cv_image = bridge.imgmsg_to_cv2(depth_image_msg)
    except CvBridgeError as error:
        print(error)
    # pick up bboxes from bboxes_msg
    detections = []
    for bbox in bboxes_msg.bounding_boxes:
        detections.append(
            [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.probability])
    # update trackers
    detections = np.array(detections)
    MOT.update(detections)

    obstacles_info = ''
    # draw tracker's state on color_cv_image, and show their ids
    for tracker in MOT.trackers:
        # global obstacles_info
        bbox = tracker.get_state()[0]  # [x1,y1,x2,y2]
        bbox = list(map(int, bbox))
        text = 'ID: ' + str(tracker.id) + ' W: {} px'.format(bbox[3] - bbox[1])
        cv2.rectangle(color_cv_image,
                      pt1=(bbox[0], bbox[1]),
                      pt2=(bbox[2], bbox[3]),
                      color=(0, 0, 0),
                      thickness=2)
        # update tracker's pos and vel
        bbox_center_x = int((bbox[0] + bbox[2]) / 2)
        bbox_center_y = int((bbox[1] + bbox[3]) / 2)
        # TODO: using camera intrinsics to calculate
        pos_vec = np.array(
            [0.0, 0.0, depth_cv_image[bbox_center_y, bbox_center_x]])
        tracker.update_pos_and_vel(pos_vec)
        # fill obstacles_info

        obstacles_info += '{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f};'.format(
            tracker.id, tracker.pos[0], tracker.pos[1], tracker.pos[2],
            tracker.vel[0], tracker.vel[1], tracker.vel[2])
        # draw obstacle depth and vel on color_cv_image
        text += ' Dist:{:.2f} m Vel:{:.2f} m/s'.format(tracker.pos[2]/1000.0,
                                                 tracker.vel[2]/1000.0)
        cv2.putText(color_cv_image,
                    text,
                    org=(bbox[0], bbox[1] - 5),
                    fontFace=1,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA)
    # publish track result image
    try:
        result_image_pub.publish(bridge.cv2_to_imgmsg(color_cv_image, 'bgr8'))
    except CvBridgeError as error:
        print(error)
    # publish obstacles info
    obstacles_pub.publish(String(obstacles_info))


if __name__ == '__main__':
    rospy.init_node('tracker_utils_node', anonymous=True)
    rospy.loginfo('Starting tracker_utils_node ...')
    color_image_sub = message_filters.Subscriber('/camera/color/image_raw',
                                                 Image)
    depth_image_sub = message_filters.Subscriber(
        '/camera/depth/image_rect_raw', Image)
    bboxes_sub = message_filters.Subscriber('/darknet_ros/bounding_boxes',
                                            BoundingBoxes)
    sync = message_filters.ApproximateTimeSynchronizer(
        [color_image_sub, depth_image_sub, bboxes_sub],
        queue_size=10,
        slop=100000000)
    sync.registerCallback(callback)
    rospy.spin()