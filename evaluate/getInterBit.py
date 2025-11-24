import re
import string
import os
# get file size in python
import os
import numpy as np

def get_all_file(dir_path):
    global files
    for filepath in os.listdir(dir_path):
        tmp_path = os.path.join(dir_path,filepath)
        if os.path.isdir(tmp_path):
            get_all_file(tmp_path)
        else:
            files.append(tmp_path)
    return files

def calc_files_size(files_path):
    files_size = 0
    for f in files_path:
        files_size += os.path.getsize(f)
    return files_size

def searchResult(chunk, matchSyntax, index):
    if re.search(matchSyntax, chunk):
        splitWord = re.split(r'\s+', chunk)
        #print chunk
        #print splitWord[3]
        matchWord = []
        matchWord.append(splitWord[index])
        return matchWord




qplist=[ "22", "32", "42", "52"]

Model='FOMM'             ## 'FV2V' OR 'FOMM' OR 'CFTE'
Iframe_format='YUV420'   ## 'YUV420'  OR 'RGB444'    
height = 512
width = 512
feature_coding_mode = 'SEI'

testingdata_name_list=['CFVQA','CFVQA_NEW']

for testingdata_name in testingdata_name_list:
    
    if testingdata_name=='CFVQA':
        frames=125 
        seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018']
    if testingdata_name=='CFVQA_NEW':
        frames=250 
        seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']


    input_bin_file_path='../experiment/'+str(Model)+'/Iframe_'+str(Iframe_format)+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode
    save_path='../experiment/'+str(Model)+'/Iframe_'+str(Iframe_format)+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode+"/resultBit/"

    totalResult=np.zeros((len(seqlist)+1,len(qplist)))
    
    
    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):  

            ## Overall Bit
            interpath=input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_512x512_25_8bit_444_qp'+str(QP)+'/'+'allframe_org.bin'
            overall_bits=os.path.getsize(interpath)*8     
            
            
            ## Iframe Bit
            Iframepath=input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_512x512_25_8bit_444_qp'+str(QP)+'/'+'allframe_org.log'
            if os.path.isfile(Iframepath):
                f = open(Iframepath, 'r')
                #print f
                while 1:
                    chunk = f.readline()
                    #print(chunk)
                    if chunk == '':
                        break             

                    bitsWord ='bits'
                    findbits = searchResult(chunk, bitsWord, 0)
                    if findbits:
                        splitbits = re.split(r'\s+', chunk)
                        Iframebits=int(splitbits[12])
                        
                        
            totalResult[seqIdx][qpIdx]=(overall_bits-Iframebits)/(frames-1)   


    # summary the bitrate
    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            totalResult[-1][qp]+=totalResult[seq][qp]
        totalResult[-1][qp] /= len(seqlist)

    np.set_printoptions(precision=5)
    # totalResult = totalResult/1000
    # seqlength = frames/25
    # totalResult = totalResult/seqlength
    np.savetxt(save_path+testingdata_name+'_'+'InterBit.txt', totalResult, fmt = '%.5f')                    
