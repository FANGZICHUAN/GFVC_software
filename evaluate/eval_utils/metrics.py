import os
import torch 
import inspect
import numpy as np 
from .utils import *
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from skimage import  img_as_float32
from scipy.signal import convolve2d
from typing import List,Dict, Any



class MetricParent:
    def __init__(self, bits=8, max_val=255, mvn=1, name=''):
        self.__name = name
        self.bits = bits
        self.max_val = max_val
        self.__metric_val_number = mvn
        self.metric_name = ''

    def set_bd_n_maxval(self, bitdepth=None, max_val=None):
        if bitdepth is not None:
            self.bits = bitdepth
        if max_val is not None:
            self.max_val = max_val

    def name(self):
        return self.__name

    def metric_val_number(self):
        return self.__metric_val_number

    def calc(self, orig, rec):
        raise NotImplementedError

class PSNR(MetricParent):
    '''Wrapper class computing PSNR-YUV
        Note: Currently returning PSNR-Y only
    '''
    def __init__(self, *args, **kwards):
        super().__init__(*args,
                         **kwards,
                         mvn=3,
                         name=['PSNR_Y', 'PSNR_U', 'PSNR_V'])

    def calc(self, org, dec, weight=None, _lambda=1.0):
        ans = []
        for plane in org:
            a = org[plane].mul((1 << self.bits) - 1)
            b = dec[plane].mul((1 << self.bits) - 1)
            sq_diff = (a-b)**2
            if weight is not None: #useful for computing ROI-weighted PSNR
                sq_diff = sq_diff*weight[:,:,0]
            mse = torch.mean(sq_diff)
            if mse == 0.0:
                ans.append(100)
            else:
                ans.append(20 * np.log10(self.max_val) - 10 * np.log10(mse))
        return float(ans[0])

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
  
class SSIM(MetricParent):
    '''A wrapper class for computing ssim'''
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='MS-SSIM (PyTorch)')
    
    def calc(self,org, dec, k1=0.01, k2=0.03, win_size=11, L=255):
        #compute only on the R channel
        im1 = org[0]
        im2 = dec[0]
        M, N = im1.shape
        C1 = (k1*L)**2
        C2 = (k2*L)**2
        window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
        window = window/np.sum(np.sum(window))
    
        if im1.dtype == np.uint8:
            im1 = np.double(im1)
        if im2.dtype == np.uint8:
            im2 = np.double(im2)
    
        mu1 = filter2(im1, window, 'valid')
        mu2 = filter2(im2, window, 'valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
        sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
        sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
    
        ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    
        return np.mean(np.mean(ssim_map))

    
class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        # a = torch.hann_window(5,periodic=False)
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class DISTS(torch.nn.Module):
    '''
    Refer to https://github.com/dingkeyan93/DISTS
    '''
    def __init__(self, channels=3, load_weights=True, device='cpu'):
        assert channels == 3
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(weights='DEFAULT').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if load_weights:
            weights = torch.load(os.path.abspath(os.path.join(inspect.getfile(DISTS),'..','weights/DISTS.pt')), map_location=device)
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']

    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, as_loss=True, resize = True):
        assert x.shape == y.shape
        if resize:
            x, y = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze()
        if as_loss:
            return score.mean()
        else:
            return score

class LPIPSvgg(torch.nn.Module):
    def __init__(self, channels=3, device='cpu'):
        # Refer to https://github.com/richzhang/PerceptualSimilarity

        assert channels == 3
        super(LPIPSvgg, self).__init__()
        vgg_pretrained_features = models.vgg16(weights='DEFAULT').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64,128,256,512,512]
        self.weights = torch.load(os.path.abspath(os.path.join(inspect.getfile(LPIPSvgg),'..','weights/LPIPSvgg.pt')), map_location=device)
        self.weights = list(self.weights.items())
        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        for k in range(len(outs)):
            outs[k] = F.normalize(outs[k])
        return outs

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        score = 0 
        for k in range(len(self.chns)):
            score = score + (self.weights[k][1]*(feats0[k]-feats1[k])**2).mean([2,3]).sum(1)
        if as_loss:
            return score.mean()
        else:
            return score

class LPIPS_IQA(MetricParent):
    def __init__(self,device, *args, **kwargs):
        super().__init__(*args, **kwargs, name='LPIPS')
        self.lpips = LPIPSvgg(device=device).to(device)
        self.device = device

    def calc(self, org: np.array, dec: np.array, weight=None):  
        ans = 0.0
        org = torch.tensor(org[np.newaxis].astype(np.float32)).to(self.device)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).to(self.device)
        ans = self.lpips(org, dec).item()  
        return ans
    
class DISTS_IQA(MetricParent):
    def __init__(self,device, *args, **kwargs):
        super().__init__(*args, **kwargs, name='LPIPS')
        self.dists = DISTS(device=device).to(device)
        self.device = device

    def calc(self, org: np.array, dec: np.array, weight=None):  
        ans = 0.0
        org = torch.tensor(org[np.newaxis].astype(np.float32)).to(self.device)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).to(self.device)
        ans = self.dists(org, dec).item()  
        return ans


import torch.nn.functional as F
from torchvision import models


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ], indexing='ij')
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale),mode='bilinear', align_corners=True)

        return out
  

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights='DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePyramide(nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class TemporalLPIPS(MetricParent):
    '''
    Temporal LPIPS (tLP) - 评估视频的时间一致性
    通过计算相邻帧特征变化的一致性来检测时域闪烁
    参考: https://github.com/universome/stylegan-v
    '''
    def __init__(self, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs, name='tLP')
        self.lpips = LPIPSvgg(device=device).to(device)
        self.device = device
        self.prev_org_frame = None
        self.prev_dec_frame = None
        
    def calc(self, org: np.ndarray, dec: np.ndarray, weight=None):
        '''
        计算当前帧与前一帧的LPIPS差异
        返回: |LPIPS(org_t, org_{t-1}) - LPIPS(dec_t, dec_{t-1})|
        值越小表示时间一致性越好
        '''
        import builtins  # 显式导入 builtins 以避免被 utils.abs 覆盖
        
        # 将numpy数组转换为torch张量
        org_tensor = torch.tensor(org[np.newaxis].astype(np.float32)).to(self.device)
        dec_tensor = torch.tensor(dec[np.newaxis].astype(np.float32)).to(self.device)
        
        if self.prev_org_frame is None or self.prev_dec_frame is None:
            # 第一帧，没有时间参考，返回0
            self.prev_org_frame = org_tensor
            self.prev_dec_frame = dec_tensor
            return 0.0
        
        # 计算原始序列相邻帧的LPIPS
        with torch.no_grad():
            lpips_org = self.lpips(self.prev_org_frame, org_tensor, as_loss=False).item()
            lpips_dec = self.lpips(self.prev_dec_frame, dec_tensor, as_loss=False).item()
        
        # 计算差异: 原始序列的变化 vs 重建序列的变化
        # 理想情况下，两者的变化应该一致
        temporal_consistency_error = builtins.abs(lpips_org - lpips_dec)
        
        # 更新前一帧
        self.prev_org_frame = org_tensor
        self.prev_dec_frame = dec_tensor
        
        return temporal_consistency_error
    
    def reset(self):
        '''重置状态，用于新序列开始时'''
        self.prev_org_frame = None
        self.prev_dec_frame = None


class FacialLandmarkDistance(MetricParent):
    '''
    人脸关键点距离 (LMD - Landmark Distance) - 评估面部几何结构保持
    在预定义关键点周围取局部窗口，计算像素级差异
    反映表情、姿态、面部轮廓的重建保真度
    '''
    def __init__(self, device='cpu', num_landmarks=5, window_size=21, *args, **kwargs):
        super().__init__(*args, **kwargs, name='LMD')
        self.device = device
        self.num_landmarks = num_landmarks
        self.window_size = window_size  # 关键点周围窗口大小
        self.reference_landmarks = self._get_predefined_landmarks()
        
    def _get_predefined_landmarks(self):
        '''
        获取预定义的5个核心人脸关键点位置（基于标准对齐人脸）
        适用于256x256分辨率的VOXCELEB/RAVDESS对齐人脸
        坐标为归一化到[0,1]的[x, y]
        '''
        core_landmarks = np.array([
            [0.35, 0.45],  # 左眼中心
            [0.65, 0.45],  # 右眼中心
            [0.50, 0.60],  # 鼻尖
            [0.40, 0.75],  # 左嘴角
            [0.60, 0.75],  # 右嘴角
        ], dtype=np.float32)
        return core_landmarks
    
    def _get_landmark_windows(self, image: np.ndarray, landmarks: np.ndarray):
        '''
        提取关键点周围的局部窗口
        输入: image [3, H, W], landmarks [N, 2] (像素坐标)
        返回: list of [3, window_size, window_size] patches
        '''
        _, H, W = image.shape
        half_window = self.window_size // 2
        patches = []
        
        # 确保landmarks是二维数组 [N, 2]
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            print(f"Warning: Invalid landmarks shape: {landmarks.shape}")
            return []
        
        for i in range(landmarks.shape[0]):
            x, y = int(landmarks[i, 0]), int(landmarks[i, 1])
            # 边界检查
            x1 = max(0, x - half_window)
            x2 = min(W, x + half_window + 1)
            y1 = max(0, y - half_window)
            y2 = min(H, y + half_window + 1)
            
            patch = image[:, y1:y2, x1:x2]
            
            # 如果边界不足，pad到标准大小
            if patch.shape[1] < self.window_size or patch.shape[2] < self.window_size:
                pad_h = self.window_size - patch.shape[1]
                pad_w = self.window_size - patch.shape[2]
                patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
            
            patches.append(patch)
        
        return patches
    
    def _compute_patch_distance(self, org_patch: np.ndarray, dec_patch: np.ndarray):
        '''
        计算两个局部窗口的差异
        使用MSE + 结构相似性的组合
        '''
        import builtins  # 显式导入以避免被utils.abs覆盖
        
        # 归一化到[0, 1]
        if org_patch.max() > 1.0:
            org_patch = org_patch / 255.0
        if dec_patch.max() > 1.0:
            dec_patch = dec_patch / 255.0
        
        # MSE
        mse = np.mean((org_patch - dec_patch) ** 2)
        
        # 局部对比度差异（结构敏感性）
        org_std = np.std(org_patch)
        dec_std = np.std(dec_patch)
        std_diff = builtins.abs(org_std - dec_std)
        
        # 综合距离：MSE + 结构差异
        combined_distance = np.sqrt(mse) + std_diff * 0.1
        
        return combined_distance
    
    def calc(self, org: np.ndarray, dec: np.ndarray, weight=None):
        '''
        计算关键点周围局部区域的重建误差
        返回: 平均局部窗口差异 (越低越好，单位：归一化像素误差)
        '''
        try:
            _, H, W = org.shape
            
            # 将归一化坐标转换为像素坐标 [x, y] -> [W, H]
            scale = np.array([W, H], dtype=np.float32)
            landmarks_px = self.reference_landmarks * scale
            
            # 提取局部窗口
            org_patches = self._get_landmark_windows(org, landmarks_px)
            dec_patches = self._get_landmark_windows(dec, landmarks_px)
            
            # 检查是否有有效的patches
            if len(org_patches) == 0 or len(dec_patches) == 0 or len(org_patches) != len(dec_patches):
                return 0.0
            
            # 计算每个关键点的局部差异
            distances = []
            for org_patch, dec_patch in zip(org_patches, dec_patches):
                dist = self._compute_patch_distance(org_patch, dec_patch)
                distances.append(dist)
            
            if len(distances) == 0:
                return 0.0
            
            mean_distance = np.mean(distances)
            
            # 转换为更直观的尺度（0-100，乘以100）
            return float(mean_distance * 100)
            
        except Exception as e:
            print(f"Warning: Failed to compute landmark distance: {e}")
            return 0.0


class ColorConsistency(MetricParent):
    '''
    颜色一致性指标 (ΔE - CIEDE2000) - 评估肤色/颜色保真度
    基于CIELAB颜色空间，计算感知颜色差异
    特别优化对肤色等低饱和度颜色的感知一致性
    参考: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, name='DeltaE')
        
    def _rgb_to_lab(self, rgb: np.ndarray):
        '''
        将RGB图像转换为CIELAB颜色空间
        输入: [3, H, W] numpy array, 范围[0, 1]或[0, 255]
        输出: [3, H, W] LAB array, L[0-100], a[-128-127], b[-128-127]
        '''
        # 确保输入是 [H, W, 3] 格式用于 skimage
        if rgb.shape[0] == 3:  # [3, H, W]
            rgb_hwc = np.transpose(rgb, (1, 2, 0))
        else:
            rgb_hwc = rgb
        
        # 归一化到 [0, 1]
        if rgb_hwc.max() > 1.0:
            rgb_hwc = rgb_hwc / 255.0
        
        # RGB -> XYZ -> LAB
        lab = self._rgb_to_lab_manual(rgb_hwc)
        
        return lab
    
    def _rgb_to_lab_manual(self, rgb: np.ndarray):
        '''
        手动实现RGB到LAB转换（避免skimage依赖）
        '''
        # 确保输入是 float32
        rgb = rgb.astype(np.float32)
        
        # Gamma校正 (sRGB -> Linear RGB)
        mask = rgb > 0.04045
        rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        
        # Linear RGB -> XYZ (D65)
        rgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)
        
        xyz = np.dot(rgb_linear, rgb_to_xyz.T)
        
        # XYZ -> LAB
        xyz_ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        xyz_normalized = xyz / xyz_ref
        
        epsilon = 0.008856
        kappa = 903.3
        
        mask = xyz_normalized > epsilon
        f_xyz = np.where(mask, xyz_normalized ** (1/3), (kappa * xyz_normalized + 16) / 116)
        
        L = 116 * f_xyz[..., 1] - 16
        a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
        b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
        
        lab = np.stack([L, a, b], axis=-1)
        return lab
    
    def _ciede2000(self, lab1: np.ndarray, lab2: np.ndarray):
        '''
        计算CIEDE2000颜色差异
        输入: 两个LAB颜色数组 [H, W, 3]
        输出: 平均ΔE值
        '''
        # 简化版ΔE计算（使用欧氏距离近似，计算更快）
        # 完整CIEDE2000公式复杂，此处使用加权LAB距离
        delta_L = lab1[..., 0] - lab2[..., 0]
        delta_a = lab1[..., 1] - lab2[..., 1]
        delta_b = lab1[..., 2] - lab2[..., 2]
        
        # 加权：L通道权重较低，颜色通道权重较高（肤色对ab敏感）
        delta_E = np.sqrt(
            (delta_L * 0.5) ** 2 +
            delta_a ** 2 +
            delta_b ** 2
        )
        
        return np.mean(delta_E)
    
    def calc(self, org: np.ndarray, dec: np.ndarray, weight=None):
        '''
        计算颜色一致性
        返回: 平均ΔE值 (越低越好，<1为不可察觉，1-2为可接受，>3为明显差异)
        '''
        try:
            # 转换到LAB空间
            org_lab = self._rgb_to_lab(org)
            dec_lab = self._rgb_to_lab(dec)
            
            # 计算颜色差异
            delta_E = self._ciede2000(org_lab, dec_lab)
            
            return float(delta_E)
            
        except Exception as e:
            print(f"Warning: Failed to compute color consistency: {e}")
            return 0.0


class IDSimilarity(MetricParent):
    '''
    身份相似度 (ID-Similarity) - 评估人脸重建的身份保持能力
    使用ArcFace/InsightFace提取512维人脸特征，计算余弦相似度
    参考: https://github.com/deepinsight/insightface
    '''
    def __init__(self, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs, name='ID-Similarity')
        self.device = device
        self.face_detector = None
        self.face_recognizer = None
        self._init_face_models()
        
    def _init_face_models(self):
        '''初始化人脸检测和识别模型'''
        try:
            # 使用torchvision的ResNet50作为轻量级人脸识别骨干
            # 实际部署时建议替换为专门的ArcFace模型
            import torchvision.models as models
            self.face_recognizer = models.resnet50(weights='DEFAULT')
            # 替换最后的全连接层为512维特征输出
            num_features = self.face_recognizer.fc.in_features
            self.face_recognizer.fc = torch.nn.Linear(num_features, 512)
            self.face_recognizer = self.face_recognizer.to(self.device)
            self.face_recognizer.eval()
            
            # 人脸检测使用简单的中心裁剪（假设输入已对齐的人脸）
            # 实际部署时可使用MTCNN或RetinaFace
            self.face_detector = None  # 简化处理，假设输入已经是人脸区域
            
        except Exception as e:
            print(f"Warning: Failed to initialize face recognition model: {e}")
            self.face_recognizer = None
    
    def _preprocess_face(self, image: np.ndarray) -> torch.Tensor:
        '''预处理图像以适应人脸识别模型'''
        # image: [3, H, W] numpy array in range [0, 1] (from img_as_float32)
        from torchvision import transforms
        
        # Convert numpy to tensor - 输入已经是 CHW 格式，无需 permute
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Apply resize and normalization
        resize = transforms.Resize((224, 224), antialias=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        image = resize(image)
        image = normalize(image)
        
        return image.unsqueeze(0).to(self.device)
    
    def _extract_face_features(self, image: np.ndarray) -> torch.Tensor:
        '''提取人脸特征向量'''
        if self.face_recognizer is None:
            return None
            
        face_tensor = self._preprocess_face(image)
        
        with torch.no_grad():
            features = self.face_recognizer(face_tensor)
            # L2归一化
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features
    
    def calc(self, org: np.ndarray, dec: np.ndarray, weight=None):
        '''
        计算身份相似度 (余弦相似度)
        返回: 相似度分数 [0, 1]，1表示完全相同，0表示完全不同
        '''
        if self.face_recognizer is None:
            return 0.0  # 模型初始化失败，返回默认值
        
        try:
            # 提取特征
            org_features = self._extract_face_features(org)
            dec_features = self._extract_face_features(dec)
            
            if org_features is None or dec_features is None:
                return 0.0
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                org_features, dec_features, dim=1
            )
            
            # 转换为标量并确保在[0,1]范围内
            similarity_score = (similarity.item() + 1) / 2  # 从[-1,1]映射到[0,1]
            
            return similarity_score
            
        except Exception as e:
            print(f"Warning: Failed to compute ID similarity: {e}")
            return 0.0


class MultiMetric:
    def __init__(self,metrics: List[str] = ['psnr','lpips','dists','ssim'], device='cpu') -> None:
        
        self.metrics = metrics
        self.monitor = {}
        if 'psnr' in self.metrics:
            self.monitor.update({'psnr':PSNR()})
        
        if 'ssim' in self.metrics:
            self.monitor.update({'ssim':SSIM()})

        if 'lpips' in self.metrics:
            self.monitor.update({'lpips':LPIPS_IQA(device=device)})

        if 'dists' in self.metrics:
            self.monitor.update({'dists':DISTS_IQA(device=device)})
        
        # 新增指标: 时间一致性 (tLP)
        if 'tlp' in self.metrics:
            self.monitor.update({'tlp': TemporalLPIPS(device=device)})
        
        # 新增指标: 身份相似度 (ID-Similarity)
        if 'id_similarity' in self.metrics:
            self.monitor.update({'id_similarity': IDSimilarity(device=device)})
        
        # 新增指标: 人脸关键点距离 (LMD)
        if 'lmd' in self.metrics:
            self.monitor.update({'lmd': FacialLandmarkDistance(device=device)})
        
        # 新增指标: 颜色一致性 (DeltaE)
        if 'delta_e' in self.metrics:
            self.monitor.update({'delta_e': ColorConsistency()})


    def compute_metrics(self,org: np.ndarray, dec: np.ndarray)-> Dict[str,float]:
        output = {}

        #According to JPEG_AI, we should compute IQA metrics in YUV format
        org_yuv = load_image_array(org)
        dec_yuv = load_image_array(dec)

        # Perceptual metrics still require RGB images in float32 format (Also the current implementation of SSIM requires RGB input)
        org_rgb_float = img_as_float32(org)
        dec_rgb_float = img_as_float32(dec)

        for metric in self.monitor:
            if metric =='ssim':
                #use uint8 RGB input :: TO-DO (MAYBE THIS SHOULD BE MOVED TO YUV[0-1])
                val = self.monitor[metric].calc(org, dec) 
            elif metric in ['lpips','dists']:
                #use RGB input as float 32
                val = self.monitor[metric].calc(org_rgb_float, dec_rgb_float) 
            elif metric in ['tlp', 'id_similarity', 'lmd', 'delta_e']:
                # 新增指标: tLP/ID-Similarity/LMD/DeltaE使用RGB float32输入
                val = self.monitor[metric].calc(org_rgb_float, dec_rgb_float)
            else:
                #use YUV images as input
                val = self.monitor[metric].calc(org_yuv, dec_yuv) 
            output[metric] = val
        return output
    
    def reset_temporal_metrics(self):
        '''
        重置时间相关指标的状态（如tLP）
        在每个新序列开始时调用
        '''
        if 'tlp' in self.monitor:
            self.monitor['tlp'].reset()


