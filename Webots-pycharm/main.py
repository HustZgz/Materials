from controller import Supervisor
import cv2
import numpy as np
import matplotlib.pyplot as plt

class MyEnv(Supervisor):
    def __init__(self):
        super(MyEnv,self).__init__()
        self.resolution =(480,480)
        self.up_camera = self.getDevice("camera")
        self.up_camera.enable(10)
        self.up_camera_node = self.getFromDef('camera')
        self.up_camera_translation = self.up_camera_node.getField('translation')
        self.up_camera.recognitionEnable(10)
        self.up_camera.enableRecognitionSegmentation()
        self.max_mean = 0
        self.robot = self.getFromDef('robot')
        self.robot_position = [0,0,0.12]
        self.robot_translation_filed = self.robot.getField('translation')
        self.robot_translation_filed.setSFVec3f(self.robot_position)
        self.varx = []
        self.vary = []

    def ShowImg(self):
        self.step()
        rgb_img = self.GetSegImage()
        bool_martrix = rgb_img[:,:,0]
        bool_martrix = np.array(bool_martrix != 0,dtype=np.int)

        leftside = bool_martrix[:,0]
        upside = bool_martrix[0,:]
        rightside = bool_martrix[:,self.resolution[0]-1]
        downside = bool_martrix[self.resolution[0]-1,:]
        downside = downside[::-1]
        leftside = leftside[::-1]

        pool = np.concatenate((upside, rightside), axis=0)
        pool = np.concatenate((pool, downside), axis=0)
        pool = np.concatenate((pool, leftside), axis=0)
        edge_idx_array = range(0,4*self.resolution[0])
        pool = pool*edge_idx_array
        if np.count_nonzero(pool) != 0:
            avg_idx = int(np.sum(pool)/np.count_nonzero(pool))
            if (avg_idx<1*self.resolution[0]):
                direction = (avg_idx,0)
            if(2*self.resolution[0]<=avg_idx and avg_idx<3*self.resolution[0]):
                direction = (3*self.resolution[0]- avg_idx,self.resolution[0])
            if (3 * self.resolution[0] <= avg_idx and avg_idx <= 4 * self.resolution[0]):
                direction = (0, 4*self.resolution[0]-avg_idx)
            elif (1*self.resolution[0]<=avg_idx and avg_idx<2*self.resolution[0]):
                direction = (self.resolution[0], avg_idx-self.resolution[0])
            cv2.arrowedLine(rgb_img, (int(self.resolution[0] / 2), int(self.resolution[0] / 2)), direction, (0, 0, 255), 2,0, 0, 0.2)
            scale = 0.00002
            self.MoveCamera(scale*(direction[0]-self.resolution[0] / 2),scale*(direction[1]-self.resolution[0] / 2))
            self.varx.append(scale*(direction[0]-self.resolution[0] / 2))
            self.vary.append(scale*(direction[1]-self.resolution[0] / 2))
            print(scale * (direction[0] - self.resolution[0] / 2), scale * (direction[1] - self.resolution[0] / 2))

        if np.count_nonzero(pool) == 0 and np.count_nonzero(bool_martrix.flatten())>0:
            flatten_bool_vec_1 = bool_martrix.flatten()
            flatten_idx_array = range(0, self.resolution[0] ** 2)
            flatten_bool_vec_1 = flatten_idx_array * flatten_bool_vec_1
            sum = np.sum(flatten_bool_vec_1,dtype=np.int64)
            num = np.count_nonzero(flatten_bool_vec_1)
            flatten_bool_mean = int(sum / num)
            y = flatten_bool_mean // self.resolution[0] + 1

            flatten_bool_vec_2 = bool_martrix.flatten('F')
            flatten_idx_array = range(0, self.resolution[0] ** 2)
            flatten_bool_vec_2 = flatten_idx_array * flatten_bool_vec_2
            sum = np.sum(flatten_bool_vec_2,dtype=np.int64)
            num = np.count_nonzero(flatten_bool_vec_2)
            flatten_bool_mean = int(sum / num)
            x = flatten_bool_mean // self.resolution[0] + 1
            cv2.arrowedLine(rgb_img, (int(self.resolution[0] / 2), int(self.resolution[0] / 2)), (x, y), (255, 0, 0), 2,0, 0, 0.2)
            scale = 0.00002
            self.MoveCamera(scale*(x-self.resolution[0] / 2),scale*(y-self.resolution[0] / 2))
            self.varx.append(scale * (x - self.resolution[0] / 2))
            self.vary.append(scale * (y - self.resolution[0] / 2))
            print(scale * (x - self.resolution[0] / 2), scale * (y - self.resolution[0] / 2))
        cv2.imshow("test", rgb_img)
        cv2.waitKey(1)
        if self.stop():
           return True

    def MoveCamera(self,dx,dy):
        self.robot_position[0]+=dx
        self.robot_position[1] -= dy
        self.robot_translation_filed.setSFVec3f(self.robot_position)
        pass


    def GetRawImage(self):
        imgbytes = self.up_camera.getImage()
        imgarry = np.frombuffer(imgbytes, dtype=np.uint8)
        img_width = int((len(imgarry) / 4) // self.resolution[0])
        imgarry = imgarry.reshape((img_width, self.resolution[0], 4))
        return imgarry

    def GetSegImage(self):
        imgbytes = self.up_camera.getRecognitionSegmentationImage()
        imgarry = np.frombuffer(imgbytes, dtype=np.uint8)
        img_width = int((len(imgarry) / 4) // self.resolution[0])
        imgarry = imgarry.reshape((img_width, self.resolution[0], 4))
        return imgarry

    def stop(self):
        if abs(self.varx[-1])<8e-05 and abs(self.vary[-1])<3.0000000000000004e-05:
          self.robot_translation_filed.disableSFTracking()
          return True

if __name__ == '__main__':
    Env = MyEnv()
    while(True):
        if Env.ShowImg():
            break