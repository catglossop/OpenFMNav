from collections import deque, defaultdict
from itertools import count
import os

import networks.depth_decoder_learned_camera
import networks.resnet_encoder
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import time
import json
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time
import random
import networks
from Grounded_SAM.gsam import GSAM, convert_SAM
#torch
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from skimage import measure
import skimage.morphology

import cv2

#ROS
# import rospy
# from geometry_msgs.msg import Twist
# from sensor_msgs.msg import Image
# import cv2
from cv_bridge import CvBridge
from agents.llm import LLM
from vl_prompt.p_manager import object_query_constructor

class BackprojectDepth_learned_camera(nn.Module):
    """Backproject function"""
    def __init__(self, batch_size, height, width, bin_size, eps=1e-7):
        super(BackprojectDepth_learned_camera, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

        if self.height == self.width:
            hw = self.height

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.ones_hw = nn.Parameter(torch.ones(self.batch_size), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = nn.Parameter(self.pix_coords.repeat(batch_size, 1, 1).view(self.batch_size, 2, self.height, self.width), requires_grad=False)
        self.x_range_k = nn.Parameter(self.pix_coords[:,0:1,:,:], requires_grad=False)
        self.y_range_k = nn.Parameter(self.pix_coords[:,1:2,:,:], requires_grad=False)

        self.bin_size = bin_size
        self.sfmax = nn.Softmax(dim=2)

        self.cx = nn.Parameter(torch.ones((self.batch_size))*(self.width-1)/2.0, requires_grad=False)
        self.cy = nn.Parameter(torch.ones((self.batch_size))*(self.height-1)/2.0, requires_grad=False)

        self.relu = nn.ReLU()

    def forward(self, depth, alpha, beta, cam_range, cam_offset):

        index_x = cam_range[:,0:1] + self.eps
        index_y = cam_range[:,1:2] + self.eps

        offset_x = cam_offset[:,0:1] + self.eps
        offset_y = cam_offset[:,1:2] + self.eps

        self.x_range = offset_x.view(self.batch_size, 1, 1, 1) + index_x.view(self.batch_size, 1, 1, 1)*(self.x_range_k - self.cx.view(self.batch_size, 1, 1, 1))/self.cx.view(self.batch_size, 1, 1, 1)
        self.y_range = offset_y.view(self.batch_size, 1, 1, 1) + index_y.view(self.batch_size, 1, 1, 1)*(self.y_range_k - self.cy.view(self.batch_size, 1, 1, 1))/self.cy.view(self.batch_size, 1, 1, 1)

        xy_t = torch.sqrt(self.x_range**2 + self.y_range**2 + self.eps)

        bin_height_group = (alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1))*(xy_t.unsqueeze(4)) + beta.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        bin_height_c, _ = torch.min(bin_height_group, dim=4)
        bin_height = torch.clamp(bin_height_c, min=-0.0, max=1.0)

        XY_t = xy_t/(bin_height + self.eps)*depth
          
        cos_td = self.x_range/(xy_t + self.eps)
        sin_td = self.y_range/(xy_t + self.eps)

        X_t = XY_t*cos_td
        Y_t = XY_t*sin_td
        Z_t = depth

        cam_points = torch.cat([X_t, Y_t, Z_t], dim = 1).view(self.batch_size, 3, -1)
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points
    
class OpenFMNav: 

    def __init__(self): 
        print("OpenFMNav initialized")
        #device(CPU or GPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lm = LLM(goal_name="object", prompt_type="scoring")
        # resize parameters
        self.rsizex = 128
        self.rsizey = 128

        # image size for depth estimation
        self.hsize = 128
        self.wsize = 416

        #bin size of our camera model
        self.bsize = 16

        # TODO: 
        self.map_size_cm = 2400
        self.map_resolution = 5

        self.bridge = CvBridge()
        # self.obs_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.obs_callback)
        # self.depth_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.depth_callback)
        self.xc = 310
        self.yc = 321

        self.yoffset = 310 
        self.xoffset = 310
        self.xyoffset = 280
        self.xplus = 661
        self.XYf = [(self.xc-self.xyoffset, self.yc-self.xyoffset), (self.xc+self.xyoffset, self.yc+self.xyoffset)]
        self.XYb = [(self.xc+self.xplus-self.xyoffset, self.yc-self.xyoffset), (self.xc+self.xplus+self.xyoffset, self.yc+self.xyoffset)]

        # define depth model 
        self.enc_depth = networks.resnet_encoder.ResnetEncoder(18, True, num_input_images = 1)
        path = os.path.join("./models", "encoder.pth")
        model_dict = self.enc_depth.state_dict()
        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.enc_depth.load_state_dict(model_dict)
        self.enc_depth.eval().to(self.device)

        self.dec_depth = networks.depth_decoder_learned_camera.DepthDecoder_learned_camera(self.enc_depth.num_ch_enc, [0, 1, 2, 3], 16)
        path = os.path.join("./models", "depth.pth")
        model_dict = self.dec_depth.state_dict()

        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.dec_depth.load_state_dict(model_dict)
        self.dec_depth.eval().to(self.device)

        ### Back-projection function of our learned camera model ###
        self.backproject_depth = BackprojectDepth_learned_camera(1, self.hsize, self.wsize, self.bsize)
        self.backproject_depth.to(self.device)

        self.transform = T.Resize(size = (self.hsize, self.wsize))
        self.transform_raw = T.Resize(size = (2*self.xyoffset,4*self.xyoffset))
        # for masking
        self.mask = torch.zeros([1, self.hsize, self.wsize])
        for i in range(self.wsize):
            for j in range(self.hsize):
                if ((i - (self.wsize/2))**2)/(self.wsize/2)**2 + ((j - (self.hsize/2))**2)/(self.hsize/2)**2 < 1.0:
                    self.mask[:, j, i] = 1.0
        self.mask_gpu = self.mask.repeat(3,1,1).unsqueeze(dim=0).to(self.device)

        #lens_parameters
        self.lens_zero = torch.zeros((2, 1)).to(self.device)
        self.binwidth_zero = torch.zeros((2, 1)).to(self.device)

        map_size = self.map_size_cm // self.map_resolution
        self.full_w, self.full_h = map_size, map_size # 2400/5=480
        self.full_map = torch.zeros(self.full_w, self.full_h).float()
        self.done = False
    
    def preprocess_image(self, msg):
        ## image preprocess for Ricoh THETA S
        cv2_msg_img = self.bridge.imgmsg_to_cv2(msg)
        pil_msg_img = cv2.cvtColor(cv2_msg_img, cv2.COLOR_BGR2RGB)
        pil_msg_imgx = Image.fromarray(pil_msg_img)

        fg_img = Image.new('RGBA', pil_msg_imgx.size, (0, 0, 0, 255))    
        draw = ImageDraw.Draw(fg_img)
        draw.ellipse(self.XYf, fill = (0, 0, 0, 0))
        draw.ellipse(self.XYb, fill = (0, 0, 0, 0))

        pil_msg_imgx.paste(fg_img, (0, 0), fg_img.split()[3])
        cv2_img = cv2.cvtColor(pil_msg_img, cv2.COLOR_RGB2BGR)

        cv_cutimg_F = cv2_img[self.yc-self.xyoffset:self.yc+self.xyoffset, self.xc-self.xyoffset:self.xc+self.xyoffset]
        cv_cutimg_B = cv2_img[self.yc-self.xyoffset:self.yc+self.xyoffset, self.xc+self.xplus-self.xyoffset:self.xc+self.xplus+self.xyoffset]

        cv_cutimg_FF = cv2.transpose(cv_cutimg_F)
        cv_cutimg_F = cv2.flip(cv_cutimg_FF, 1)
        cv_cutimg_Bt = cv2.transpose(cv_cutimg_B)
        cv_cutimg_B = cv2.flip(cv_cutimg_Bt, 0)
        cv_cutimg_BF = cv2.flip(cv_cutimg_Bt, -1)

        cv_cutimg_n = np.concatenate((cv_cutimg_F, cv_cutimg_B), axis=1)
        return cv_cutimg_n

    def preprocess_image_pil(self, pil_img):
        ## image preprocess for Ricoh THETA S
        pil_msg_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2RGB)
        pil_msg_img = cv2.resize(pil_msg_img, (1280, 640))
        pil_msg_imgx = Image.fromarray(pil_msg_img)

        fg_img = Image.new('RGBA', pil_msg_imgx.size, (0, 0, 0, 255))    
        draw = ImageDraw.Draw(fg_img)
        draw.ellipse(self.XYf, fill = (0, 0, 0, 0))
        draw.ellipse(self.XYb, fill = (0, 0, 0, 0))

        pil_msg_imgx.paste(fg_img, (0, 0), fg_img.split()[3])
        cv2_img = cv2.cvtColor(pil_msg_img, cv2.COLOR_RGB2BGR)
        cv_cutimg_F = cv2_img[self.yc-self.xyoffset:self.yc+self.xyoffset, self.xc-self.xyoffset:self.xc+self.xyoffset]
        cv_cutimg_B = cv2_img[self.yc-self.xyoffset:self.yc+self.xyoffset, self.xc+self.xplus-self.xyoffset:self.xc+self.xplus+self.xyoffset]
        cv_cutimg_FF = cv2.transpose(cv_cutimg_F)
        cv_cutimg_F = cv2.flip(cv_cutimg_FF, 1)

        cv_cutimg_Bt = cv2.transpose(cv_cutimg_B)
        cv_cutimg_B = cv2.flip(cv_cutimg_Bt, 0)
        cv_cutimg_BF = cv2.flip(cv_cutimg_Bt, -1)
        cv_cutimg_n = np.concatenate((cv_cutimg_F, cv_cutimg_B), axis=1)
        return cv_cutimg_n

    def disp_to_depth(self, disp, min_depth, max_depth):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth
    
    def obs_callback(self, msg):
        self.obs_img = self.preprocess_image(msg)

    def get_depth(self, img):
        img = self.preprocess_image_pil(img)
        cv_trans = img.transpose(2, 0, 1)
        cv_trans_np = np.array([cv_trans], dtype=np.float32)
            
        double_fisheye_gpu = torch.from_numpy(cv_trans_np).float().to(self.device)

        image_d = torch.cat((self.mask_gpu*self.transform(double_fisheye_gpu[:,:,:,0:2*self.xyoffset]).clone(), self.mask_gpu*self.transform(double_fisheye_gpu[:,:,:,2*self.xyoffset:4*self.xyoffset]).clone()), dim=0)/255.0
        image_d_flip = torch.flip(image_d, [3])
        image_dc = torch.cat((image_d, image_d_flip), dim=0)

        #depth estimation
        with torch.no_grad():
            features = self.enc_depth(image_dc)   
            outputs_c, camera_param_c, binwidth_c, camera_range_c, camera_offset_c = self.dec_depth(features)                 
            
            outputs = (outputs_c[("disp", 0)][0:2,:,:,:] + torch.flip(outputs_c[("disp", 0)][2:4],[3]))*0.5
            camera_param = (camera_param_c[0:2] + camera_param_c[2:4])*0.5
            binwidth = (binwidth_c[0:2] + binwidth_c[2:4])*0.5
            self.camera_range = (camera_range_c[0:2] + camera_range_c[2:4])*0.5                                        
            self.camera_offset = (camera_offset_c[0:2] + camera_offset_c[2:4])*0.5

        #camera model process
        cam_lens_x = []
        bdepth, _, _, _ = image_d.size()
        lens_zero = torch.zeros((bdepth, 1)).to(self.device)
        binwidth_zero = torch.zeros((bdepth, 1)).to(self.device)
        for i in range(self.bsize):
            lens_height = torch.zeros(bdepth, 1, device=self.device)
            for j in range(0, i+1):
                lens_height += camera_param[:, j:j+1]
            cam_lens_x.append(lens_height)
        cam_lens_c = torch.cat(cam_lens_x, dim=1)
        cam_lens = 1.0 - torch.cat([lens_zero, cam_lens_c], dim=1)

        lens_bincenter_x = []
        for i in range(self.bsize):
            bin_center = torch.zeros(bdepth, 1, device=self.device)
            for j in range(0, i+1):
                bin_center += binwidth[:, j:j+1]
            lens_bincenter_x.append(bin_center)
        lens_bincenter_c = torch.cat(lens_bincenter_x, dim=1)
        lens_bincenter = torch.cat([binwidth_zero, lens_bincenter_c], dim=1)
                    
        self.lens_alpha = (cam_lens[:,1:self.bsize+1] - cam_lens[:,0:self.bsize])/(lens_bincenter[:,1:self.bsize+1] - lens_bincenter[:,0:self.bsize] + 1e-7)
        self.lens_beta = (-cam_lens[:,1:self.bsize+1]*lens_bincenter[:,0:self.bsize] + cam_lens[:,0:self.bsize]*lens_bincenter[:,1:self.bsize+1] + 1e-7)/(lens_bincenter[:,1:self.bsize+1] - lens_bincenter[:,0:self.bsize] + 1e-7)                
                    
        double_disp = torch.cat((outputs[0:1], outputs[1:2]), dim=3)
        pred_disp_pano, pred_depth = self.disp_to_depth(double_disp, 0.1, 100.0)
                    
        # backprojection to have point clouds
        # cam_points_f: estimated point clouds from front fisheye image on the front camera image coordinate
        # cam_points_b: estimated point clouds from back fisheye image on the back camera image coordinate
        self.cam_points_f = self.backproject_depth(pred_depth[:,:,:,0:self.wsize], self.lens_alpha[0:1], self.lens_beta[0:1], self.camera_range[0:1], self.camera_offset[0:1])
        self.cam_points_b = self.backproject_depth(pred_depth[:,:,:,self.wsize:2*self.wsize], self.lens_alpha[1:2], self.lens_beta[1:2], self.camera_range[1:2], self.camera_offset[1:2])
        self.disp = pred_disp_pano
        self.depth = pred_depth
        return self.cam_points_f, self.cam_points_b, self.disp
    
    def depth_callback(self, msg):

        img = self.preprocess_image(msg)
        cv_trans = img.transpose(2, 0, 1)
        cv_trans_np = np.array([cv_trans], dtype=np.float32)
            
        double_fisheye_gpu = torch.from_numpy(cv_trans_np).float().to(self.device)

        image_d = torch.cat((self.mask_gpu*self.transform(double_fisheye_gpu[:,:,:,0:2*self.xyoffset]).clone(), self.mask_gpu*self.transform(double_fisheye_gpu[:,:,:,2*self.xyoffset:4*self.xyoffset]).clone()), dim=0)/255.0
        image_d_flip = torch.flip(image_d, [3])
        image_dc = torch.cat((image_d, image_d_flip), dim=0)

        #depth estimation
        with torch.no_grad():
            features = self.enc_depth(image_dc)   
            outputs_c, camera_param_c, binwidth_c, camera_range_c, camera_offset_c = self.dec_depth(features)                 
            
            outputs = (outputs_c[("disp", 0)][0:2,:,:,:] + torch.flip(outputs_c[("disp", 0)][2:4],[3]))*0.5
            camera_param = (camera_param_c[0:2] + camera_param_c[2:4])*0.5
            binwidth = (binwidth_c[0:2] + binwidth_c[2:4])*0.5
            self.camera_range = (camera_range_c[0:2] + camera_range_c[2:4])*0.5                                        
            self.camera_offset = (camera_offset_c[0:2] + camera_offset_c[2:4])*0.5

        #camera model process
        cam_lens_x = []
        bdepth, _, _, _ = image_d.size()
        lens_zero = torch.zeros((bdepth, 1)).to(self.device)
        binwidth_zero = torch.zeros((bdepth, 1)).to(self.device)
        for i in range(self.bsize):
            lens_height = torch.zeros(bdepth, 1, device=self.device)
            for j in range(0, i+1):
                lens_height += camera_param[:, j:j+1]
            cam_lens_x.append(lens_height)
        cam_lens_c = torch.cat(cam_lens_x, dim=1)
        cam_lens = 1.0 - torch.cat([lens_zero, cam_lens_c], dim=1)

        lens_bincenter_x = []
        for i in range(self.bsize):
            bin_center = torch.zeros(bdepth, 1, device=self.device)
            for j in range(0, i+1):
                bin_center += binwidth[:, j:j+1]
            lens_bincenter_x.append(bin_center)
        lens_bincenter_c = torch.cat(lens_bincenter_x, dim=1)
        lens_bincenter = torch.cat([binwidth_zero, lens_bincenter_c], dim=1)
                    
        self.lens_alpha = (cam_lens[:,1:self.bsize+1] - cam_lens[:,0:self.bsize])/(lens_bincenter[:,1:self.bsize+1] - lens_bincenter[:,0:self.bsize] + 1e-7)
        self.lens_beta = (-cam_lens[:,1:self.bsize+1]*lens_bincenter[:,0:self.bsize] + cam_lens[:,0:self.bsize]*lens_bincenter[:,1:self.bsize+1] + 1e-7)/(lens_bincenter[:,1:self.bsize+1] - lens_bincenter[:,0:self.bsize] + 1e-7)                
                    
        double_disp = torch.cat((outputs[0:1], outputs[1:2]), dim=3)
        pred_disp_pano, pred_depth = self.disp_to_depth(double_disp, 0.1, 100.0)
                    
        # backprojection to have point clouds
        # cam_points_f: estimated point clouds from front fisheye image on the front camera image coordinate
        # cam_points_b: estimated point clouds from back fisheye image on the back camera image coordinate
        self.cam_points_f = self.backproject_depth(pred_depth[:,:,:,0:self.wsize], self.lens_alpha[0:1], self.lens_beta[0:1], self.camera_range[0:1], self.camera_offset[0:1])
        self.cam_points_b = self.backproject_depth(pred_depth[:,:,:,self.wsize:2*self.wsize], self.lens_alpha[1:2], self.lens_beta[1:2], self.camera_range[1:2], self.camera_offset[1:2])
        self.disp = pred_disp_pano
        self.pred_depth = pred_depth
        return self.cam_points_f, self.cam_points_b, self.disp

    def mask_to_pose(self, mask, cam_points_f, cam_points_b, disp):
        
        # get the location of the goal object in the scene 
        mask_cpu = mask.cpu().detach().numpy()
        depth_mask = mask_cpu * np.array(disp)
        depth_mask = np.resize(depth_mask, (1, self.hsize, 2*self.wsize))
        depth_mask = depth_mask[0, :, :self.wsize]
        self.goal_points = self.cam_points_f.cpu().detach().numpy().reshape((4, depth_mask.shape[0], depth_mask.shape[1]))
        obj_inds = np.argwhere(depth_mask > 0)
        try: 
            self.obj_points = self.goal_points[:, obj_inds[:, 0], obj_inds[:, 1]]
        except: 
            print("Object not found in front camera view")
            return None

        # get the pose of the goal object in the scene
        self.goal_pose = np.mean(self.obj_points,axis =1)
        print("Object pose: ", self.goal_pose)

        return self.goal_pose

    
    def main(self):

        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Explored Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories, versatile
        # Calculating full and local map sizes

        instruction = input("Please input the instruction: ")
        print("Instruction: ", instruction)

        # Get object proposals 
        O_pri = self.lm.imagine_candidate(instruction)
        print("Object proposals: ", O_pri)

        O_dis = []

        while not self.done: 

            # Get the current observation
            # obs = self.obs_img
            obs_orig = Image.open("img/image_bike.png")
            obs = self.preprocess_image_pil(obs_orig)
            self.cam_points_f, self.cam_points_b, self.disp = self.get_depth(obs_orig)
            self.disp_resized = self.disp.to("cpu").detach().numpy().squeeze()
            self.disp_resized = cv2.resize(self.disp_resized, (obs.shape[1], obs.shape[0]))
            self.disp_resized = np.array(self.disp_resized)
            self.disp_resized = Image.fromarray(self.disp_resized)  
            self.depth_resized = self.depth.to("cpu").detach().numpy().squeeze()
            self.depth_resized = cv2.resize(self.depth_resized, (obs.shape[1], obs.shape[0]))
            self.depth_resized = np.array(self.depth_resized)
            plt.imshow(self.depth_resized)
            plt.show()

            obs = Image.fromarray(obs)
            # Discover what is in the observation
            O_dis_new = self.lm.discover_objects(obs, O_dis)
            
            print("New discovered objects: ", O_dis_new)
            O_dis.append(O_dis_new)

            # Group the discovered objects and the proposed objects from the instruction
            O_group = self.lm.group_candidate(O_dis, O_pri)

            print("Grouped objects: ", O_group)

            # Format objects into a prompt
            prompt = object_query_constructor(O_group)

            # Feed into 'PerceptVLM' 
            self.gsam = GSAM(O_group, text_threshold=0.55, device=self.device)

            # Predict objects from the observation 
            sam_semantic_pred = self.gsam.predict(obs)
            # See in target object in sam prediction
            masks = sam_semantic_pred[0]
            for i in range(len(O_group)):
                if masks[:,:,].max() > 0:
                    print("Target object found: ", O_pri[i])
                    print(sam_semantic_pred[2][i])
                    goal = O_pri[i]
                    goal_mask = masks[i]
                    bbox = sam_semantic_pred[1][i]
                    break
                else: 
                    print("Target object not found")
            
            # Convert mask to pose for planner 
            self.mask_to_pose(goal_mask, self.cam_points_f, self.cam_points_b, self.depth_resized)

            # Plan the path to the goal
            print("Done!")
            self.done = True

if __name__ == "__main__":
    openfmnav = OpenFMNav()
    openfmnav.main()
