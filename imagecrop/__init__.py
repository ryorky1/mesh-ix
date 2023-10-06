import torch
import cv2
import numpy as np
import imagecrop

from imagecrop.models.with_mobilenet import PoseEstimationWithMobileNet
from imagecrop.modules.keypoints import extract_keypoints, group_keypoints
from imagecrop.modules.load_state import load_state
from imagecrop.modules.pose import Pose, track_poses
# import demo
import pifuhd

from imagecrop.val import normalize, pad_width

# from pifuhd.recon import reconWrapper

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad



def get_rect(net, images, height_size):
    _runOnCPU = not hasattr(net,'gpu')
    net = net.eval()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 33
    for image in images:
        
        rect_path = image.replace('.%s' % (image.split('.')[-1]), '_rect.txt')
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        orig_img = img.copy()
        orig_img = img.copy()
        
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu=_runOnCPU)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []

        rects = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            valid_keypoints = []
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
            valid_keypoints = np.array(valid_keypoints)
            
            if pose_entries[n][10] != -1.0 or pose_entries[n][13] != -1.0:
              pmin = valid_keypoints.min(0)
              pmax = valid_keypoints.max(0)

              center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int)
              radius = int(0.65 * max(pmax[0]-pmin[0], pmax[1]-pmin[1]))
            elif pose_entries[n][10] == -1.0 and pose_entries[n][13] == -1.0 and pose_entries[n][8] != -1.0 and pose_entries[n][11] != -1.0:
              # if leg is missing, use pelvis to get cropping
              center = (0.5 * (pose_keypoints[8] + pose_keypoints[11])).astype(np.int)
              radius = int(1.45*np.sqrt(((center[None,:] - valid_keypoints)**2).sum(1)).max(0))
              center[1] += int(0.05*radius)
            else:
              center = np.array([img.shape[1]//2,img.shape[0]//2])
              radius = max(img.shape[1]//2,img.shape[0]//2)

            x1 = center[0] - radius
            y1 = center[1] - radius

            rects.append([x1, y1, 2*radius, 2*radius])
        #
        # @TODO: Revisit the output path or keep it in memory
        #
        np.savetxt(rect_path, np.array(rects), fmt='%d')

def crop(**_args):
    """
    :model_path
    :device
    :image_path
    """
    _path='checkpoint_iter_370000.pth' if 'model_path' not in _args else _args['model_path']
    _image_path = _args['image_path']
    _device='cpu' if 'device' not in _args else _args['device']

    _nnetwork = PoseEstimationWithMobileNet()
    _checkpoint = torch.load(_path, map_location=_device)
    load_state(_nnetwork, _checkpoint)

    get_rect(_nnetwork, _image_path, 512)      



from pifuhd.recon import reconWrapper

def mesh(**_args) :
    """
    :input_path     location of the image
    :out_path       path to output the mesh (obj file)
    :model_path     This is the path of the pre-trained model (not the one for cropping)
                    This model is located at https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt
    """
    _RESOLUTION = '512'
    _USE_RECTANGLE = True
    _LOADSIZE = '1024'
    _CHECKPOINT = 'checkpoints/pifuhd.pt' if 'model_path' not in _args else _args['model_path']
    _cmd = ['--dataroot',_args['image_folder'],'--results_path',_args['output_folder'],'--loadSize','1024','--resolution',_RESOLUTION,
            '--load_netMR_checkpoint_path',_CHECKPOINT,
            '--start_id','%d'%-1,'--end_id','%d'%-1
            ]
    reconWrapper(_cmd,_USE_RECTANGLE)

image_path='/home/rhino/dev/data/images/004.jpg'
# image_path='/home/rhino/dev/data/images/test.png'
#
# we proceed as follows :
#   1. crop the image (models/checkpoint included)
#   2. generate the mesh (pre-trained model included)
#    
_args = {'model_path':'imagecrop/checkpoint_iter_370000.pth','image_path':[image_path]}
crop(**_args)
image_folder = '/home/rhino/dev/data/images'
output_folder = '/home/rhino/tmp/images'
mesh(image_folder=image_folder,output_folder=output_folder)