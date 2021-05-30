# Reference
# https://github.com/FlagArihant2000/visual-odometry
# https://github.com/v-shetty/Visual-Odometry-for-Monocular-Camera
# https://stackoverflow.com/questions/63413018/opencv-triangulate-points-from-2-images-to-estimate-the-pose-on-a-third

import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))
    
    def get_line_set(self, center, world_corner):
        points = [center, world_corner[0],world_corner[1],world_corner[2],world_corner[3]]
        lines = [[0, 1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[1,4]]
        colors = [[1, 0, 0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
        
    
    def run(self):
        self.process_frames()

    def length_of_t(self, t):
        return np.linalg.norm(t,2)
    
    def calc_scale(self, old_cloud, new_cloud):
        new_cloud_roll = np.roll(new_cloud, shift = -3)
        old_cloud_roll = np.roll(old_cloud, shift = -3)
        d_ratio = (np.linalg.norm(old_cloud - old_cloud_roll, axis = -1))/(np.linalg.norm(new_cloud - new_cloud_roll,axis = -1))
        return np.median(d_ratio)
    
    def triangulation(self, R, t, kp0, kp1, K):
        p0 = np.array([[1, 0, 0, 0], 
                       [0, 1, 0, 0], 
                       [0, 0, 1, 0]])
        p0 = K.dot(p0)
        p1 = np.hstack((R, t))
        p1 = K.dot(p1)
        points1 = kp0.reshape(2, -1)
        points2 = kp1.reshape(2, -1)
        point_cloud = cv.triangulatePoints(p0, p1, points1, points2).reshape(-1, 4)[:,:3]
        return point_cloud
    
    def get_world_pyramid_corner(self, rot_matrix, t, cameraMatrix):
        t = t.reshape(3)
        corner = np.array([[0,0, 1],[639, 0, 1],[639, 359, 1],[0, 359, 1]]).T
        corner_w = np.dot(rot_matrix, np.dot(np.linalg.inv(cameraMatrix), corner)).T + t
        return t, corner_w
    
    
    def process_frames(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width = 800, height = 800)
        
        
        current_pos_all = []
        current_rot_all = []
        
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        
        current_pos = np.zeros((3, 1), dtype=np.float64)
        current_rot = np.eye(3, dtype=np.float64)
        
        current_pos_all.append(current_pos.copy())
        current_rot_all.append(current_rot.copy())
        
        orb = cv.ORB_create()
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        
        prev_img = cv.imread(self.frame_paths[0])
        
        for now_idx, frame_path in enumerate(self.frame_paths[1:]):
            if now_idx == 0:
                curr_img = cv.imread(frame_path)

                kp1, des1 = orb.detectAndCompute(prev_img, None)
                kp2, des2 = orb.detectAndCompute(curr_img, None)

                matches = bf.match(des1,des2)
                matches = sorted(matches, key = lambda x:x.distance)

                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                E, mask = cv.findEssentialMat(pts2, pts1, self.K, cv.RANSAC, 0.999, 1, None)
                _, R, t, mask = cv.recoverPose(E, pts2, pts1, cameraMatrix=self.K, mask=mask)
            
                current_pos += current_rot.dot(t)
                current_rot = R.dot(current_rot)
                
                current_pos_all.append(current_pos.copy())
                current_rot_all.append(current_rot.copy())
                
                prev_img = curr_img
                R_prev = R
                t_prev = t
            
            # rescale if not the first one
            if now_idx != 0:
                curr_img = cv.imread(frame_path)
                prev_prev_img = cv.imread(self.frame_paths[now_idx - 1])
                print(f"current {frame_path}    ", end="\r")

                kp0, des0 = orb.detectAndCompute(prev_prev_img, None)
                kp1, des1 = orb.detectAndCompute(prev_img, None)
                kp2, des2 = orb.detectAndCompute(curr_img, None)
                
                matches_prev_state = bf.match(des1,des0)
                matches_prev_state = sorted(matches_prev_state, key = lambda x:x.distance)
                matches = bf.match(des1,des2)
                matches = sorted(matches, key = lambda x:x.distance)
                
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                E, mask = cv.findEssentialMat(pts2, pts1, self.K, cv.RANSAC, 0.999, 1, None)
                _, R_curr, t_curr, mask = cv.recoverPose(E, pts2, pts1, cameraMatrix=self.K, mask=mask)
                
                intersection = set([m.queryIdx for m in matches]).intersection([m.queryIdx for m in matches_prev_state])
                
                kp0_index = []
                kp0_reindex = []
                kp1_index = []
                kp1_prev_index = []
                kp2_index = []
                for m in matches:
                    if (m.queryIdx in intersection):
                        kp1_index.append(m.queryIdx)
                        kp2_index.append(m.trainIdx)
                
                for m in matches_prev_state:
                    if (m.queryIdx in intersection):
                        kp1_prev_index.append(m.queryIdx)
                        kp0_index.append(m.trainIdx)
                        
                for q_idx in kp1_index:
                    t_idx = kp1_prev_index.index(q_idx)
                    kp0_reindex.append(kp0_index[t_idx])
                
                
                pts0 = np.float32([kp0[idx].pt for idx in kp0_reindex]).reshape(-1, 1, 2)
                pts1 = np.float32([kp1[idx].pt for idx in kp1_index]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[idx].pt for idx in kp2_index]).reshape(-1, 1, 2)
                
                new_cloud = self.triangulation(R_curr, t_curr, pts1, pts2, self.K)
                old_cloud = self.triangulation(R_prev, t_prev, pts0, pts1, self.K)

                med_ratio = self.calc_scale(old_cloud, new_cloud)
                
                current_pos += current_rot.dot(t_curr) * med_ratio
                current_rot = R_curr.dot(current_rot)
                
                current_pos_all.append(current_pos.copy())
                current_rot_all.append(current_rot.copy())
                
                prev_img = curr_img
                R_prev = R_curr
                t_prev = t_curr
                
            img = cv.drawKeypoints(curr_img, kp2, None, color=(0,255,0))
            cv.imshow('frame', img)
            
            center, world_corner = self.get_world_pyramid_corner(current_rot, current_pos, self.K)
            line_set = self.get_line_set(center, world_corner)
            vis.add_geometry(line_set)
            vis.poll_events()
            if cv.waitKey(30) == 27: break
            
            
        current_pos_all = np.array(current_pos_all)
        current_rot_all = np.array(current_rot_all)
        cv.destroyWindow('frame')
        
        vis.run()
        vis.destroy_window()
        
        self.current_pos_all = current_pos_all
        self.current_rot_all = current_rot_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="./frames/", help='directory of sequential frames')
    parser.add_argument('--camera_parameters', type=str, default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
