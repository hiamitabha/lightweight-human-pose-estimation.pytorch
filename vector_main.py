import threading

import anki_vector
from anki_vector import events
import numpy as np

import argparse

import cv2
import torch
import anki_vector
from anki_vector.events import Events
from anki_vector.util import degrees
import threading
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

from utils import detect_hand,detect_face, detect_touch
import time
from demo import infer_fast



previous_poses = []


# taken from demo.py
def run_on_image(net, height_size, cpu, track, smooth,img, stride, upsample_ratio, num_keypoints,threshold):
        global previous_poses
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)
        score = 0
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            # cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),


            r_hand_center, r_hand_width, l_hand_center, l_hand_width, = detect_hand(pose)

            if -1 not in r_hand_center:
                cv2.circle(img, (r_hand_center[0], r_hand_center[1]), 5, (255, 0, 0), 5)
                cv2.rectangle(img, (r_hand_center[0]-r_hand_width, r_hand_center[1]-r_hand_width),
                              (r_hand_center[0] + r_hand_width, r_hand_center[1] + r_hand_width), (0, 255, 255))
            if -1 not in l_hand_center:
                cv2.circle(img, (l_hand_center[0], l_hand_center[1]), 5, (255, 0, 0), 5)
                cv2.rectangle(img, (l_hand_center[0]-l_hand_width, l_hand_center[1]-l_hand_width),
                              (l_hand_center[0] + l_hand_width, l_hand_center[1] + l_hand_width), (0, 255, 255))

            face_center, face_width = detect_face(pose)
            if -1 not in face_center:
                cv2.rectangle(img, (face_center[0] - face_width, face_center[1] - face_width),
                          (face_center[0] + face_width, face_center[1] + face_width), (0, 0, 255))

                #               (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                if track:
                    cv2.putText(img, 'id: {}'.format(pose.id), (face_center[0] - face_width, face_center[1] - face_width - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

            if -1 not in r_hand_center:
                x,y,h,w, score= detect_touch(face_center,face_width,r_hand_center,r_hand_width)
                if h!=0:
                    cv2.rectangle(img, (x,y),
                              (x+h,y+w), (255, 0, 255))
                    cv2.putText(img, f'Score: {score:0.2f}', (x, y - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))
            if -1 not in l_hand_center:
                x, y, h, w, score = detect_touch(face_center, face_width, l_hand_center, l_hand_width)
                if h != 0:
                    cv2.rectangle(img, (x, y),
                                  (x +h, y + w), (255, 0, 255))
                    cv2.putText(img, f'Score: {score:0.2f}', (x, y - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        delay = 1
        detect = False

        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 33:
                delay = 0
            else:
                delay = 33
        return score>threshold


said_text = False
touched = False
last_touched = 0

def on_robot_observed_touch(robot, event_type, event):
    print("Vector sees a touch")
    global said_text
    global last_touched

    if not said_text:
        last_touched = time.time()
        said_text = True
        robot.behavior.say_text("Don't touch your face")
        anim = robot.anim.play_animation('anim_rtpickup_loop_09', ignore_head_track=True)
        said_text = False

        robot.behavior.set_head_angle(degrees(25.0))
        robot.behavior.set_lift_height(0.0)



def on_new_raw_camera_image(robot, event_type, event,net):
    print("Display new camera image " , time.time())
    global previous_poses
    global last_touched

    # opencvImage = cv2.cvtColor(np.array(event.image), cv2.COLOR_RGB2BGR)      #This has lower latency but when a touch is detected this will lag behind
    opencvImage = cv2.cvtColor(np.array(robot.camera.latest_image.raw_image), cv2.COLOR_RGB2BGR)

    # print(opencvImage.shape)
    # cv2.imshow('hello', opencvImage)
    # cv2.waitKey(1)

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    threshold = 0.15 #score for detecting face touch

    touched = run_on_image(net, 256, cpu=False, track=True, smooth=True, img=opencvImage, stride=stride, upsample_ratio=upsample_ratio, num_keypoints=num_keypoints, threshold= threshold)
    if touched and 2 < time.time()-last_touched:
        last_touched = time.time()
        robot.conn.run_coroutine(robot.events.dispatch_event_by_name('face touch detected', event_name='touched'))


def main():

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)
    net = net.cuda()
    done = threading.Event()

    with anki_vector.AsyncRobot() as robot:
        robot.camera.init_camera_feed()
        robot.camera.image_streaming_enabled()

        # preparing robot pose ready
        robot.behavior.set_head_angle(degrees(25.0))
        robot.behavior.set_lift_height(0.0)

        #events for detection and new camera feed
        robot.events.subscribe(on_new_raw_camera_image, events.Events.new_raw_camera_image, net)
        robot.events.subscribe_by_name(on_robot_observed_touch, event_name='touched')

        print("------ waiting for camera events, press ctrl+c to exit early ------")

        try:
            if not done.wait(timeout=600):
                print("------ Did not receive a new camera image! ------")
        except KeyboardInterrupt:
            pass

if __name__=="__main__":
    main()