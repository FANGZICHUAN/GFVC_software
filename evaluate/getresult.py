# +
# get file size in python
import os
import numpy as np


# ================= 配置区域 =================
Inputformat='Iframe_YUV420' # 'RGB444' OR 'YUV420'
testingdata_name='CFVQA' # 'CFVQA' OR 'VOXCELEB'
Model='CFTE'             ## 'FV2V' OR 'FOMM' OR 'CFTE' 
Metric = 'lpips' ## 'psnr' 'dists' 'lpips' 'ssim'

Width = '256'                 # 分辨率宽 (请确认是 256 还是 512)
Height = '256'                # 分辨率高
Mode = 'AE'                   # 编码模式: 'AE' 或 'SEI'
# ===========================================================
#txt_path='../experiment/'+Model+'/'+Inputformat+'/evaluation-YUV420/'+testingdata_name+'_result_'+Metric+'.txt'
#txt_path='../experiment/'+Model+'/'+Inputformat+'/evaluation/'+testingdata_name+'_result_'+Metric + '.txt'
# txt_path = '/home/ysz/GFVC/experiment/FV2V/Iframe_YUV420/resultBit/CFVQA_resultBit.txt'
txt_path = '../experiment/'+Model+'/'+Inputformat+'/'+Width+'x'+Height+'/'+Mode+'/evaluation/'+testingdata_name+'_result_'+Metric+'.txt'


with open(txt_path, 'r') as file:
    # content = file.read()
    for line in file: 
        words=line.split()
        count = len(words)
        if count == 0: continue # 跳过空行
            
        for num in range(count): 
                val = float(words[num])
                if Metric in ['psnr','ssim']:
                    print(f"{val:.5f}")
                elif Metric in ['lpips','dists']:
                    print(f"{1 - val:.5f}")
        #for num in range(4):
            #if Metric in ['psnr','ssim']:
                #print(words[num])
            #elif Metric in ['lpips','dists']:
                #print(1-float(words[num]))

    
    
