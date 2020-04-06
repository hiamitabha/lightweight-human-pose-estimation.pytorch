from modules.pose import Pose
import numpy as np

#reimplementation of the https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
# except that we don't need to take 0.9 max of the sholder to elbow into account because we want the hand bounding box be smaller when hand is stright toward camera
# this makes it more robust
def detect_hand(current_pose: Pose):
    keypoints = current_pose.keypoints
    r_arm = keypoints[3:5]
    l_arm = keypoints[6: 8]
    r_hand_center = np.array([-1, -1], dtype=int)
    l_hand_center = np.array([-1, -1], dtype=int)
    l_hand_width = np.array([0], dtype=int)
    r_hand_width = np.array([0], dtype=int)
    if -1 not in r_arm:
        r_hand_center = 0.33 * (r_arm[1] - r_arm[0]) + r_arm[1]
        r_hand_width = 0.2 * np.linalg.norm(r_arm[1] - r_arm[0])
    if -1 not in l_arm:
        l_hand_center = 0.33 * (l_arm[1] - l_arm[0]) + l_arm[1]
        l_hand_width = 0.2 * np.linalg.norm(l_arm[1] - l_arm[0])

    r_hand_center = r_hand_center.astype(int)
    r_hand_width = r_hand_width.astype(int)
    l_hand_center = l_hand_center.astype(int)
    l_hand_width = l_hand_width.astype(int)
    return r_hand_center, r_hand_width, l_hand_center, l_hand_width


# simply taking all the detected of the face into account and multiply it by a number
# we can make it better by going around 
def detect_face(current_pose: Pose):
    keypoints = current_pose.keypoints
    face = np.concatenate((keypoints[0].reshape(1, 2), keypoints[14:]), axis=0)
    face = face[face.min(axis=1) >= 0, :]
    if face.shape[0] < 2:
        return np.array([-1, -1]), 0

    min = np.min(face, axis=0)
    max = np.max(face, axis=0)
    width = int((max[0] - min[0]) * 1.2 / 2)
    center = np.average(face, axis=0)

    center = center.astype(int)
    return center, width


# from https://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/
def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return x, y, w, h


# from https://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/
def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0: return (0,0,0,0)  # or (0,0,0,0) ?
    return x, y, w, h


def detect_touch(rec1_center, rec1_width, rec2_center, rec2_width):
    rec1 = [rec1_center[0] - rec1_width, rec1_center[1] - rec1_width, 2 * rec1_width, 2 * rec1_width]
    rec2 = [rec2_center[0] - rec2_width, rec2_center[1] - rec2_width, 2 * rec2_width, 2 * rec2_width]
    x, y, w, h = intersection(rec1, rec2)
    score = h*w / (4*rec2_width**2) # calculates how many percentage of the hand bounding box has intersection with face
    return x,y,w,h,score

