import sys
sys.path.append('../../')
import numpy as np
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image, ImageOps
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation
import pandas as pd
from scipy.spatial import distance
import time
import os
import math
import scipy.io as sio
import cv2
from utils.renderer import Renderer
from utils.image_operations import expand_bbox_rectangle
from utils.pose_operations import get_pose
from img2pose import img2poseModel
from model_loader import load_model
from PIL import Image
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def render_plot(img, poses, bboxes):
    renderer = Renderer(
        vertices_path="./pose_references/vertices_trans.npy", 
        triangles_path="./pose_references/triangles.npy"
    )

    (w, h) = img.size
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
    
    trans_vertices = renderer.transform_vertices(img, poses)
    img = renderer.render(img, trans_vertices, alpha=1)
    # for bbox in bboxes:
    #     bbox = bbox.astype(np.uint8)
    #     print(bbox)
    #     img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
    return img

def get_pose(img, res_only=False):
    threed_points = np.load('./pose_references/reference_3d_68_points_trans.npy')

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    transform = transforms.Compose([transforms.ToTensor()])

    DEPTH = 18
    MAX_SIZE = 1400
    MIN_SIZE = 600

    POSE_MEAN = "./models/WIDER_train_pose_mean_v1.npy"
    POSE_STDDEV = "./models/WIDER_train_pose_stddev_v1.npy"
    MODEL_PATH = "./models/img2pose_v1.pth"

    pose_mean = np.load(POSE_MEAN)
    pose_stddev = np.load(POSE_STDDEV)

    img2pose_model = img2poseModel(
        DEPTH, MIN_SIZE, MAX_SIZE, 
        pose_mean=pose_mean, pose_stddev=pose_stddev,
        threed_68_points=threed_points,
    )
    load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
    img2pose_model.evaluate()

    (w, h) = img.size
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

    res = img2pose_model.predict([transform(img)])[0]
    if res_only:
        return res

    all_bboxes = res["boxes"].cpu().numpy().astype('float')

    poses = []
    bboxes = []
    threshold = 0.9
    for i in range(len(all_bboxes)):
        if res["scores"][i] > threshold:
            bbox = all_bboxes[i]
            pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
            pose_pred = pose_pred.squeeze()
            print(pose_pred*180)

            poses.append(pose_pred)  
            bboxes.append(bbox)

    return render_plot(img.copy(), poses, bboxes)

def test_pose():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.flip(frame, 0)
        key = cv2.waitKey(1) & 0xFF
        time_0 = time.time()
        frame = get_pose(frame)
        logging.info("reference time: {}".format(time.time()-time_0))
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

        cv2.imshow("", frame)
        if key == ord("q"):
            break

        # if key == ord("d"):
        #     import ipdb; ipdb.set_trace()

def get_pose_from_imgs_folder(folder_path):
    import glob
    import os
    import tqdm

    folder_path = os.path.join(folder_path, "*.jpg")
    for img_path in tqdm.tqdm(glob.glob(folder_path)):
        img = cv2.imread(img_path)
        res = get_pose(img, res_only=True)
        threshold = 0.9
        dofs = []
        for i in range(len(res['boxes'])):
            if res["scores"][i] > threshold:
                dofs.append(res['dofs']*180/np.pi)
        print("{} - dofs {}".format(img_path, dofs))
        print("++++++++++++++++++++++++++++++++++++++")

def get_pose_from_img_file(img_path):
    import glob
    import os

    img = cv2.imread(img_path)
    res = get_pose(img, res_only=True)
    threshold = 0.9
    dofs = []
    for i in range(len(res['boxes'])):
        if res["scores"][i] > threshold:
            dofs.append(res['dofs']*180/np.pi)
    print("{} - dofs {}".format(img_path, dofs))
    print("++++++++++++++++++++++++++++++++++++++")

if __name__=="__main__":
    # test_pose()
    get_pose_from_img_file("pose_data/42.0.jpg")