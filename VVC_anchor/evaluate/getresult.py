# +
# get file size in python
import os
import numpy as np



Inputformat='YUV420' # 'RGB444' OR 'YUV420'
testingdata_name='CFVQA_NEW' # 'CFVQA' OR 'VOXCELEB'
Metric = 'psnr' ## 'psnr' 'dists' 'lpips' 'ssim'
# txt_path='../experiment/'+Inputformat+'/evaluation/'+testingdata_name+'_result_'+Metric + '.txt'
txt_path='../experiment/'+Inputformat+'/evaluation-YUV420/'+testingdata_name+'_result_'+Metric + '.txt'

# txt_path = '/home/ysz/GFVC_Software_v2_final/VVC_anchor/experiment/YUV420/CFVQA_resultBit.txt'


with open(txt_path, 'r') as file:
    # content = file.read()
    for line in file: 
        words=line.split()
        for num in range(4):
            if Metric in ['psnr','ssim']:
                print(words[num])
            elif Metric in ['lpips','dists']:
                print(1-float(words[num]))

    
    
