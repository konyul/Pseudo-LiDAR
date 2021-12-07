import numpy as np
from ...utils import box_utils
import torch


def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2

def project_image_to_cam(points,cam_to_img):
    ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
    c_u = cam_to_img[0, 2]
    c_v = cam_to_img[1, 2]
    f_u = cam_to_img[0, 0]
    f_v = cam_to_img[1, 1]
    b_x = cam_to_img[0, 3] / (-f_u)  # relative
    b_y = cam_to_img[1, 3] / (-f_v)
    
    x = ((points[:, 0] - c_u) * points[:, 2]) / f_u + b_x
    y = ((points[:, 1] - c_v) * points[:, 2]) / f_v + b_y

    pts_3d_rect = points.clone()
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    #pts_3d_rect[:, 2] = points[:, 2]
    return pts_3d_rect

def cart2hom(pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = torch.hstack((pts_3d, torch.ones((n, 1)).to(pts_3d.device)))
        return pts_3d_hom

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = torch.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = torch.transpose(Tr[0:3, 0:3],0,1)
    inv_Tr[0:3, 3] = torch.matmul(-torch.transpose(Tr[0:3, 0:3],0,1), Tr[0:3, 3])
    return inv_Tr.to(Tr.device)

def project_cam_to_velo(pts_3d_cam, lidar_to_cam):
    cam_to_lidar = inverse_rigid_trans(lidar_to_cam)

    return torch.transpose(torch.matmul(cam_to_lidar, torch.transpose(cart2hom(pts_3d_cam), 0, 1)), 0, 1)

def project_image_to_velo(points, lidar_to_cam, cam_to_img):
    pts_3d_cam = project_image_to_cam(points, cam_to_img)
    return project_cam_to_velo(pts_3d_cam, lidar_to_cam)

