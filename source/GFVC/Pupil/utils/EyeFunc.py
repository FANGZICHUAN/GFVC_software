from GFVC.Pupil.utils import eye
import numpy as np
import cv2 as cv
import copy
from scipy.spatial import distance
from GFVC.Pupil.utils.OneEuroFilter import *
eyeName = ["right_eye", "left_eye"]
eyeSubName = ["eyelid", "outer_corner", "inner_corner", "iris", "pupil", "full_eye", "eye7"]


def DrawPoints(image,points,r=0):
    #画点
    _image = image.copy()
    for point in points:
        x,y = point
        cv.circle(_image, center=(int(x), int(y)), radius=r, color=(0, 0, 255))
    return _image

def mp_getLandmarks(landmarks,frame_width,frame_height):
    #landmarks:output.multi_face_landmarks[i].landmark[n:m]
    #计算lanmarks的图像坐标
    _landmarks = []
    for landmark in landmarks:
        x = int(landmark.x*frame_width)
        y = int(landmark.y*frame_height)
        _landmarks.append([x,y])
    return _landmarks

def mp_getLandmarks_norm(landmarks):
    #landmarks:output.multi_face_landmarks[i].landmark[n:m]
    #计算lanmarks的图像坐标
    _landmarks = []
    for landmark in landmarks:
        x = landmark.x
        y = landmark.y
        _landmarks.append([x, y])
    return _landmarks

def mp_copyLandmarks(landmarks, idx_array):
    #landmarks:output.multi_face_landmarks[i]
    #根据索引复制相应landmarks
    _landmarks=[]
    for idx in idx_array:
        _landmarks.append(landmarks.landmark[idx])
    return _landmarks

def mp_getEyeLandmarks(landmarks, frame_w, frame_h, flip = 0):
    #landmarks:output.multi_face_landmarks[i]
    #计算眼部landmarks关键点，返回一个字典
    _landmarks = {}
    #这里的左右是指人的左右
    eyeName=["right_eye", "left_eye"] if not flip else ["left_eye", "right_eye"]
    global eyeSubName
    eyeIdx=[[eye.rightEyelid_6,
             eye.rightOuterCorner,
             eye.rightInnerCorner,
             eye.rightEyeIris,
             eye.rightEyePupil,
             eye.rightEye,
             eye.rightEyelid_7],
            [eye.leftEyelid_6,
             eye.leftOuterCorner,
             eye.leftInnerCorner,
             eye.leftEyeIris,
             eye.leftEyePupil,
             eye.leftEye,
             eye.leftEyelid_7]]
    for idx,name in enumerate(eyeName):
        sub_landmarks={}
        for idxx,subname in enumerate(eyeSubName):
            copyLandmarks = mp_copyLandmarks(landmarks,eyeIdx[idx][idxx])
            sub_landmarks[subname+'_norm'] = mp_getLandmarks_norm(copyLandmarks)
            sub_landmarks[subname] = mp_getLandmarks(copyLandmarks,frame_w,frame_h)
        _landmarks[name] = sub_landmarks
    return _landmarks

def cropImage(img,start_xy,size=(32,32)):
    _img = img[start_xy[1]:start_xy[1]+size[1],start_xy[0]:start_xy[0]+size[0]]
    return _img

def eyeOffset(eye_dict,dx,dy,eyeName):
    global eyeSubName
    _eye_dict = copy.deepcopy(eye_dict)
    for subName in eyeSubName:
        newData = []
        for i,data in enumerate(eye_dict[eyeName][subName]):
            _data = [data[0]-dx,data[1]-dy]
            newData.append(_data)
        _eye_dict[eyeName][subName]=newData
    return _eye_dict

def eyeOffset_norm(eye_dict,dx,dy,eyeName):
    global eyeSubName
    _eye_dict = copy.deepcopy(eye_dict)
    for subName in eyeSubName:
        newData = []
        for i,data in enumerate(eye_dict[eyeName][subName + '_norm']):
            _data = [data[0]-dx,data[1]-dy]
            newData.append(_data)
        _eye_dict[eyeName][subName+'_norm']=newData
    return _eye_dict


# def cropEyeImage(img,eye_dict, w=0, h=0, size=None, filter_ctr=True, te=1/25, mincutoff=1.0, beta=0.007, dcutoff=1.0):
#     global eyeName, _size
#
#     if filter_ctr:
#         _filter = {}
#         for name in eyeName:
#             _filter[name] = OneEuroFilter(te, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
#     crop_w = None
#     _eye_dict = copy.deepcopy(eye_dict)
#     if size is not None:
#         for idx, name in enumerate(eyeName):
#             outer = eye_dict[eyeName[idx]]['outer_corner'][0]
#             inner = eye_dict[eyeName[idx]]['inner_corner'][0]
#             # if filter_ctr:
#             #     outer, inner= _filter[name].predict(np.array([outer, inner]), te).astype(np.uint8)
#             center_x = int((outer[0] + inner[0]) / 2)
#             center_y = int((outer[1] + inner[1]) / 2)
#             if filter_ctr:
#                 center_x, center_y = _filter[name].predict(np.array([center_x, center_y]), te)
#             start_x = center_x - int((size[0] / 2))
#             start_y = center_y - int((size[1] / 2))
#             _img = cropImage(img, (start_x, start_y), size)
#             _eye_dict[eyeName[idx]]['img'] = _img
#             _eye_dict[eyeName[idx]]['start_xy'] = (start_x, start_y)
#             _eye_dict = eyeOffset(_eye_dict, start_x,  start_y, name)
#
#             #norm
#             if h*w != 0:
#                 sx = start_x/w
#                 sy = start_y/h
#                 _eye_dict = eyeOffset_norm(_eye_dict, sx, sy, name)
#     else:
#         for idx, name in enumerate(eyeName):
#             outer = eye_dict[eyeName[idx]]['outer_corner'][0]
#             inner = eye_dict[eyeName[idx]]['inner_corner'][0]
#             # if filter_ctr:
#             #     outer, inner= _filter[name].predict(np.array([outer, inner]), te).astype(np.uint8)
#             center_x = int((outer[0] + inner[0]) / 2)
#             center_y = int((outer[1] + inner[1]) / 2)
#             if filter_ctr:
#                 center_x, center_y = _filter[name].predict(np.array([center_x, center_y]), te)
#             if _size[name] is None:
#                 crop_w = abs(int((outer[0] - inner[0]) * 1.1))
#                 _size[name] = crop_w
#                 # if filter_ctr:
#                 #     crop_w = _filter[name+'_w'].predict(crop_w, te)
#                 # crop_h = int(crop_w * 3/4)
#
#             else:
#                 crop_w = _size[name]
#             crop_h = crop_w
#             start_x = center_x - int(crop_w / 2)
#             start_y = center_y - int(crop_h / 2)
#
#             _img = cropImage(img, (start_x, start_y), (crop_w, crop_h))
#             _eye_dict[eyeName[idx]]['img'] = _img
#             _eye_dict[eyeName[idx]]['start_xy'] = (start_x, start_y)
#             _eye_dict = eyeOffset(_eye_dict, start_x,  start_y, name)
#
#             # norm
#             if h * w != 0:
#                 sx = start_x / w
#                 sy = start_y / h
#                 _eye_dict = eyeOffset_norm(_eye_dict, sx, sy, name)
#     return _eye_dict


def getMask(img, landmarks, color_mod="Our"):
    if color_mod == "Our":
        mod = cv.COLOR_RGB2GRAY
    elif color_mod == "BGR":
        mod = cv.COLOR_BGR2GRAY
    else:
        print("The mod:{} hasn't been defined!".format(color_mod))
        exit(0)
    img_gray = cv.cvtColor(img, mod)
    mask = np.zeros_like(img_gray)

    points = np.array(landmarks,dtype=np.int32)
    convexhull = cv.convexHull(points)
    cv.fillConvexPoly(mask, convexhull, 255)
    return mask

# def getEyeMask(full_dict, crop_dict, frame, mod="BGR"):
#     global eyeName
#     mask_dict={}
#     _mask_dict={}
#     maskName=[]
#     non={}
#     for name in eyeName:
#         _mask = getMask(frame, full_dict[name]["full_eye"],mod)
#         _mask_dict["mask"] = _mask
#         _mask_dict["img"] = frame
#         mask_dict[name + "_full"] = _mask_dict
#         _mask_dict = copy.deepcopy(_mask_dict)
#         maskName.append(name+"_full")
#         _mask = getMask(crop_dict[name]["img"], crop_dict[name]["full_eye"], mod)
#         _mask_dict["mask"] = _mask
#         _mask_dict["img"] = crop_dict[name]["img"]
#         mask_dict[name + "_crop"] = _mask_dict
#         _mask_dict = copy.deepcopy(_mask_dict)
#         maskName.append(name + "_crop")
#     _mask = cv.bitwise_or(mask_dict[eyeName[0]+"_full"]["mask"], mask_dict[eyeName[1]+"_full"]["mask"])
#     _mask_dict["mask"] = _mask
#     _mask_dict["img"] = frame
#     mask_dict["both_eye"] = _mask_dict
#     maskName.append("both_eye")
#     return mask_dict, maskName

def getEyeMask(crop_dict, mod="BGR"):
    global eyeName
    mask_dict={}
    _mask_dict={}
    maskName=[]
    for name in eyeName:
        _mask = getMask(crop_dict[name]["img"], crop_dict[name]["full_eye"], mod)
        _mask_dict["mask"] = _mask
        _mask_dict["img"] = crop_dict[name]["img"]
        mask_dict[name + "_crop"] = _mask_dict
        _mask_dict = copy.deepcopy(_mask_dict)
        maskName.append(name + "_crop")
    return mask_dict, maskName


def calculate_ear(eye):
    # 计算垂直方向的欧氏距离
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])

    # 计算水平方向的欧氏距离
    horizontal = distance.euclidean(eye[0], eye[3])

    # 计算纵横比
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def getVertical2D(vec, dir=0):
    #0-逆时针，1-顺时针
    r = np.linalg.norm(vec, ord=2)
    if dir == 0:
        return np.array([-vec[1]/r, vec[0]/r])
    elif dir == 1:
        return np.array([vec[1]/r, -vec[0]/r])

def Point2Line(p, parse):
    A, B, C = parse
    return abs(A*p[0]+B*p[1]+C)/(np.linalg.norm([A, B], ord=2))

def encCoordinate(inner, outer, pupil, eye='left'):
    if not isinstance(inner, np.ndarray):
        inner = np.array(inner)
    if not isinstance(outer, np.ndarray):
        outer = np.array(outer)
    if not isinstance(pupil, np.ndarray):
        pupil = np.array(pupil)
    px = outer - inner
    r = np.linalg.norm(px, ord=2)
    # px = px/r
    if eye == 'left':
        py = getVertical2D(px)
    elif eye == 'right':
        py = getVertical2D(px, dir=1)
    else:
        print('the eye is error!')
        return 0
    A = px[1] / px[0]
    B = -1
    C = inner[1] - A * inner[0]
    dy = Point2Line(pupil, np.array([A, B, C]))
    sgn = 1 if (A * inner[0] + B * (inner[1] - 1) + C) * (A * pupil[0] + B * pupil[1] + C) >= 0 else -1
    dy = sgn * dy
    dx = np.linalg.norm((pupil - inner - dy * py), ord=2) / r
    dy = dy / r
    return np.array([dx, dy])

def decCoordinate(p, inner, outer, eye = 'left'):
    if not isinstance(p, np.ndarray):
        p = np.array(p)
    if not isinstance(inner, np.ndarray):
        inner = np.array(inner)
    if not isinstance(outer, np.ndarray):
        outer = np.array(outer)
    dx, dy = p
    px = outer - inner
    r = np.linalg.norm(px, ord=2)
    if eye == 'left':
        py = getVertical2D(px)
    elif eye == 'right':
        py = getVertical2D(px, dir=1)
    else:
        print('the eye is error!')
        return 0
    return dx*px + dy*r*py + inner

class CropEyeImage:
    def __init__(self, filter_ctr=False, te=1/25, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.size = {"right_eye": None, "left_eye": None}
        self.filter_ctr = filter_ctr
        self.te = te
        if filter_ctr:
            self.filter = {}
            for name in eyeName:
                self.filter[name] = OneEuroFilter(te, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)

    def cropEyeImage(self, img,eye_dict, w=0, h=0, size=None):
        global eyeName
        _eye_dict = copy.deepcopy(eye_dict)
        if size is not None:
            for idx, name in enumerate(eyeName):
                outer = eye_dict[eyeName[idx]]['outer_corner'][0]
                inner = eye_dict[eyeName[idx]]['inner_corner'][0]
                center_x = int((outer[0] + inner[0]) / 2)
                center_y = int((outer[1] + inner[1]) / 2)
                if self.filter_ctr:
                    center_x, center_y = self.filter[name].predict(np.array([center_x, center_y]), self.te)
                start_x = center_x - int((size[0] / 2))
                start_y = center_y - int((size[1] / 2))
                _img = cropImage(img, (start_x, start_y), size)
                _eye_dict[eyeName[idx]]['img'] = _img
                _eye_dict[eyeName[idx]]['start_xy'] = (start_x, start_y)
                _eye_dict = eyeOffset(_eye_dict, start_x,  start_y, name)

                #norm
                if h*w != 0:
                    sx = start_x/w
                    sy = start_y/h
                    _eye_dict = eyeOffset_norm(_eye_dict, sx, sy, name)
        else:
            for idx, name in enumerate(eyeName):
                outer = eye_dict[eyeName[idx]]['outer_corner'][0]
                inner = eye_dict[eyeName[idx]]['inner_corner'][0]
                center_x = int((outer[0] + inner[0]) / 2)
                center_y = int((outer[1] + inner[1]) / 2)
                if self.filter_ctr:
                    center_x, center_y = self.filter[name].predict(np.array([center_x, center_y]), self.te)
                if self.size[name] is None:
                    crop_w = abs(int((outer[0] - inner[0]) * 1.2))
                    self.size[name] = crop_w
                else:
                    crop_w = self.size[name]
                # crop_h = int(crop_w * 3/4)
                crop_h = crop_w
                start_x = center_x - int(crop_w / 2)
                start_y = center_y - int(crop_h / 2 + crop_h / 12)

                _img = cropImage(img, (start_x, start_y), (crop_w, crop_h))
                _eye_dict[eyeName[idx]]['img'] = _img
                _eye_dict[eyeName[idx]]['start_xy'] = (start_x, start_y)
                _eye_dict = eyeOffset(_eye_dict, start_x,  start_y, name)

                # norm
                if h * w != 0:
                    sx = start_x / w
                    sy = start_y / h
                    _eye_dict = eyeOffset_norm(_eye_dict, sx, sy, name)
        return _eye_dict


