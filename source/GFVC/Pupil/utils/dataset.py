import os
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms


def cropImage(img,start_xy,size=(32,32)):
    _img = img[start_xy[1]:start_xy[1]+size[1],start_xy[0]:start_xy[0]+size[0]]
    return _img

def equalevelOffset(data,offset):
    _data = np.array(data)
    _offset = np.array(offset)
    if _offset.shape == _data.shape:
        if len(_data.shape) == 1:
            for idx in range(_data.size):
                _data[idx] = _data[idx] - _offset[idx]
            return _data
        elif len(_data.shape) == 0:
            _data = _data - _offset
            return _data
        else:
            for idx in range(_data.shape[0]):
                _data[idx] = equalevelOffset(_data[idx], _offset[idx])
            return _data
    elif len(_data.shape) == 1 and _offset.shape==(1,) :
        for idx in range(_data.size):
            _data[idx] = _data[idx] - _offset

        return _data
    else:
        print("The shape both sub_data and offset is not equal!")


def allOffset(data, offset):
    _data = np.array(data)
    _offset = np.array(offset)
    if len(_data.shape) == len(_offset.shape):
        _data = equalevelOffset(_data, _offset)
        return _data
    else:
        for idx in range(_data.shape[0]):
            _data[idx] = allOffset(_data[idx], _offset)
        return _data

class myDataset(Dataset):
    """

    """

    def __init__(self, root_dir, id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, eye='right', datasetID=0):
        self.root_dir = root_dir
        # self.videos = os.listdir(root_dir)
        # self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.eye = eye
        self.datasetID = datasetID

        if datasetID == 0:
            #默认分割
            if is_train:
                datasetlist = os.listdir(os.path.join(self.root_dir,'train','Image'))
                dataset_out = []
                for list in datasetlist:
                    sub_list = os.listdir(os.path.join(self.root_dir,'train','Image',list))
                    for list_ in sub_list:
                        dataset_out.append(list_ + '_' + list)
                self.videos = dataset_out
            else:
                datasetlist = os.listdir(os.path.join(self.root_dir, 'test', 'Image'))
                dataset_out = []
                for list in datasetlist:
                    sub_list = os.listdir(os.path.join(self.root_dir, 'test', 'Image', list))
                    for list_ in sub_list:
                        dataset_out.append(list_ + '_' + list)
                self.videos = dataset_out
        else:
            dataset_out = ['id0000']
            self.videos = dataset_out


        self.is_train = is_train

        if eye == "right":
            self.eye_flag = 1
        else:
            self.eye_flag = 0





    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.datasetID == 0:
            if self.is_train:
                mod = 'train'
            else:
                mod = 'test'

            id_name, cam = self.videos[idx].split('_')

            frames_list = os.listdir(os.path.join(self.root_dir, mod, 'Image', cam, id_name))
            num_frames = len(frames_list)
            if self.is_train:
                frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            else:
                frame_idx = range(num_frames)
            frames_array = [cv.imread(os.path.join(self.root_dir, mod, 'Image', cam, id_name, frames_list[idx])) for idx in
                            frame_idx]

            landmarks_list = []
            for idx in frame_idx:
                file_name = frames_list[idx].split('.')[0]
                label_path = os.path.join(self.root_dir, mod, 'Label', cam, id_name)

                landmarks = []
                with open(os.path.join(label_path, file_name + '.txt'), 'r') as file:
                    for line in file:
                        line = line.strip('\n').split(',')
                        if int(line[0]) == self.eye_flag:
                            landmarks_idx = line[1]
                            x = int(line[2])
                            y = int(line[3])

                            landmarks.append((x, y))
                file.close()

                landmarks_list.append(landmarks)

            img = []
            landmarks_crop = []
            w = 128
            h = 128
            for i in range(len(frames_array)):
                cx = int((landmarks_list[i][0][0] + landmarks_list[i][3][0]) / 2)
                cy = int((landmarks_list[i][0][1] + landmarks_list[i][3][1]) / 2)
                L = abs(landmarks_list[i][0][0] - landmarks_list[i][3][0])
                # w = (round(L * 1.5) // 8 + 1) * 8
                # h = (round(w * 3 / 4) // 8 + 1) * 8
                # w = (round(L*1.5)//16+1)*16
                # h = int(w/2)
                cy_ = int(cy - h / 16)
                sx = int(cx - w / 2)
                sy = int(cy_ - h / 2)
                img.append(cropImage(frames_array[i], (sx, sy), size=(w, h)).astype(np.float32))
                landmarks_crop.append(allOffset(landmarks_list[i], (sx, sy)))

            out={}

            # transform = transforms.Compose([
            #     transforms.ToTensor,
            #     transforms.Normalize([w / 2, h / 2], [w / 2, h / 2])
            # ])

            if self.is_train:
                # out["source_img"]=self.transform(img[0])
                # out["driving_img"]=self.transform(img[1])
                out["source_img"] = img[0].transpose((2, 0, 1))/255
                out["driving_img"] = img[1].transpose((2, 0, 1))/255
                out["source_lms"] = (np.array(landmarks_crop[0], dtype='float32')-64)/64
                out["driving_lms"] = (np.array(landmarks_crop[1], dtype='float32')-64)/64
            else:

                img = np.array(img, dtype='float32')/255
                landmarks_crop = np.array(landmarks_crop, dtype='float32')
                out["video_img"] = img.transpose((3, 0, 1, 2))
                out["video_lms"] = (landmarks_crop-64)/64
            out["name"] = id_name+'_'+cam


            return out
        else:
            id_name = self.videos[idx]

            frames_list = os.listdir(os.path.join(self.root_dir, 'Image', id_name))
            num_frames = len(frames_list)
            if self.is_train:
                frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            else:
                frame_idx = range(num_frames)
            frames_array = [cv.imread(os.path.join(self.root_dir, 'Image', id_name, frames_list[idx])) for idx in
                            frame_idx]

            landmarks_list = []
            for idx in frame_idx:
                file_name = frames_list[idx].split('.')[0]
                label_path = os.path.join(self.root_dir, 'Label', id_name)

                landmarks = []
                with open(os.path.join(label_path, file_name + '.txt'), 'r') as file:
                    for line in file:
                        line = line.strip('\n').split(',')
                        if int(line[0]) == self.eye_flag:
                            landmarks_idx = line[1]
                            x = int(line[2])
                            y = int(line[3])

                            landmarks.append((x, y))
                file.close()

                landmarks_list.append(landmarks)

            img = []
            landmarks_crop = []
            # w = 32
            # h = 32
            for i in range(len(frames_array)):
                cx = int((landmarks_list[i][0][0] + landmarks_list[i][3][0]) / 2)
                cy = int((landmarks_list[i][0][1] + landmarks_list[i][3][1]) / 2)
                L = abs(landmarks_list[i][0][0] - landmarks_list[i][3][0])
                w = round(L*1.2)
                h = w
                # h = round(((w * 3 / 4) // 8 + 1) * 8)
                # w = (round(L*1.5)//16+1)*16
                # h = int(w/2)
                cy_ = int(cy - h / 16)
                sx = int(cx - w / 2)
                sy = int(cy_ - h / 2)
                img.append(cropImage(frames_array[i], (sx, sy), size=(w, h)).astype(np.float32))
                landmarks_crop.append(allOffset(landmarks_list[i], (sx, sy)))

            out = {}

            # transform = transforms.Compose([
            #     transforms.ToTensor,
            #     transforms.Normalize([w / 2, h / 2], [w / 2, h / 2])
            # ])

            if self.is_train:
                # out["source_img"]=self.transform(img[0])
                # out["driving_img"]=self.transform(img[1])
                # """调试代码"""
                # cv.imwrite('./pic/test_dataset_001.png', img[0])
                # cv.imwrite('./pic/test_dataset_002.png', img[1])
                # """调试代码/>"""
                img[0] = cv.resize(img[0], (128, 128))
                img[1] = cv.resize(img[1], (128, 128))
                # """调试代码"""
                # cv.imwrite('./pic/test_dataset_003.png', img[0])
                # cv.imwrite('./pic/test_dataset_004.png', img[1])
                # exit()
                # """调试代码/>"""
                out["source_img"] = img[0].transpose((2, 0, 1))/255
                out["driving_img"] = img[1].transpose((2, 0, 1))/255
                out["source_lms"] = (np.array(landmarks_crop[0], dtype='float32')-w/2)/w/2
                out["driving_lms"] = (np.array(landmarks_crop[1], dtype='float32')-w/2)/w/2
            else:
                img = cv.resize(img, (128, 128))
                img = np.array(img, dtype='float32')/255
                landmarks_crop = np.array(landmarks_crop, dtype='float32')
                out["video_img"] = img.transpose((3, 0, 1, 2))
                out["video_lms"] = (landmarks_crop-w/2)/w/2
            out["name"] = id_name


            return out



class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


