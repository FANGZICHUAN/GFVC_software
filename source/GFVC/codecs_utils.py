import os
from skimage.transform import resize
from typing import  List
from GFVC.CFTE_utils import *
from GFVC.CFTE_Lightweight_utils import *
from GFVC.DAC_utils import *
from GFVC.FOMM_utils import *
from GFVC.FV2V_utils import *
from GFVC.HDAC_utils import *
from GFVC.Translator import Paramtranscoder


def get_inter_coder(model_name):
    if model_name == 'CFTE':
        return CFTE
    elif model_name == 'CFTE_Lightweight':
        return CFTE_Lightweight
    elif model_name == 'DAC':
        return DAC
    elif model_name == 'FOMM':
        return FOMM
    elif model_name == 'FV2V':
        return FV2V
    elif model_name == 'HDAC':
        return HDAC
    elif model_name == 'HDAC_Extension':
        return HDAC_Extension
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

def get_translator(encoder_name, decoder_name, device, trans_channel=256):
    assert encoder_name in ['FOMM', 'CFTE', 'FV2V', 'DAC'], f"Translation with {encoder_name} Encoder not implemented"
    assert decoder_name in ['FOMM', 'CFTE', 'FV2V', 'DAC'], f"Translation with {decoder_name} Decoder not implemented"


    checkpoint_path='./GFVC/Translator/checkpoint/'+encoder_name+'2'+decoder_name+'_checkpoint.pth.tar'  
    trans_enc = getattr(Paramtranscoder,'load_'+encoder_name+'_Enc_translator')(checkpoint_path, trans_channel, cpu=(device=='cpu'))
    trans_dec = getattr(Paramtranscoder,'load_'+decoder_name+'_Dec_translator')(checkpoint_path, trans_channel, cpu=(device=='cpu'))
    if encoder_name == 'FOMM':
        ## FOMM
        FOMM_config_path='./GFVC/FOMM/checkpoint/FOMM.yaml'
        FOMM_checkpoint_path='./GFVC/FOMM/checkpoint/FOMM-checkpoint.pth.tar'         
        FOMM_Analysis_Model, _ = load_fomm_checkpoints(FOMM_config_path, FOMM_checkpoint_path, device=device)
        encoder_analyzer = FOMM_Analysis_Model
        print('--FOMM Model Loaded--')
    elif encoder_name == 'CFTE':
        ## CFTE
        CFTE_config_path='./GFVC/CFTE/checkpoint/CFTE.yaml'
        CFTE_checkpoint_path='./GFVC/CFTE/checkpoint/CFTE-checkpoint.pth.tar'         
        CFTE_Analysis_Model, _ = load_cfte_checkpoints(CFTE_config_path, CFTE_checkpoint_path, device=device)
        encoder_analyzer = CFTE_Analysis_Model
        print('--CFTE Model Loaded--')


    elif encoder_name == 'FV2V':
        FV2V_config_path='./GFVC/FV2V/checkpoint/FV2V.yaml'
        FV2V_checkpoint_path='./GFVC/FV2V/checkpoint/FV2V-checkpoint.pth.tar'         
        _, FV2V_Analysis_Model_Estimator, _ = load_fv2v_checkpoints(FV2V_config_path, FV2V_checkpoint_path, device=device)
        encoder_analyzer = FV2V_Analysis_Model_Estimator
        
        print('--FV2V Model Loaded--')
    
    elif encoder_name == 'DAC':
        DAC_config_path='./GFVC/DAC/checkpoint/DAC.yaml'
        DAC_checkpoint_path='./GFVC/DAC/checkpoint/DAC-checkpoint.pth.tar'         
        DAC_Analysis_Model, _ = load_dac_checkpoints(DAC_config_path, DAC_checkpoint_path, device=device)
        encoder_analyzer = DAC_Analysis_Model
        print('--DAC Model Loaded--')
    
   

    return trans_enc, trans_dec, encoder_analyzer

####################################################################################################
####################################################################################################


class ReferenceImageCoder:
    def __init__(self,img_output_dir:str,qp:int, iframe_format:str, width:int=256, height:int=256,
                 codec_name:str='vtm', device:str='cpu'):
        self.iframe_format = iframe_format
        self.qp = qp
        self.width=width
        self.height=height
        self.img_output_dir = img_output_dir
        self.codec_name = codec_name
        self.device = device


    def vtm_yuv_compress(self, frame_idx_str:str)->tuple:
        # wtite ref and cur (rgb444) to file (yuv420) 
              
        os.system("./image_codecs/vtm/encode.sh "+self.img_output_dir+'frame'+frame_idx_str+" "+self.qp+" "+str(self.width)+" "+str(self.height))   ########################

        bin_file=self.img_output_dir+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8
        
        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_rgb=yuv420_to_rgb444(self.img_output_dir+'frame'+frame_idx_str+'_rec.yuv', self.width, self.height, 0, 1, False, False) 
        img_rec = rec_ref_rgb[0]
        img_rec = img_rec
        return img_rec, bits               
    
    def vtm_rgb_compress(self, frame_idx_str)->tuple:
        # wtite ref and cur (rgb444) 
        os.system("./image_codecs/vtm/encode_rgb444.sh "+self.img_output_dir+'frame'+frame_idx_str+" "+self.qp+" "+str(self.width)+" "+str(self.height))   ########################
        
        bin_file=self.img_output_dir+'frame'+frame_idx_str+'.bin'
        bits=os.path.getsize(bin_file)*8
        
        f_temp=open(self.img_output_dir+'frame'+frame_idx_str+'_rec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*self.height*self.width).reshape((3,self.height,self.width))   # 3xHxW RGB         
        img_rec = resize(img_rec, (3, self.height, self.width))    # normlize to 0-1                  
        return img_rec, bits  
    
    def compress(self, reference_frame: List[np.ndarray], frame_idx:int)->tuple:
        frame_idx_str= str(frame_idx).zfill(4) 
        if self.iframe_format=='YUV420':
            f_temp=open(self.img_output_dir+'frame'+frame_idx_str+'_org.yuv','w')
            img_input_rgb = cv2.merge(reference_frame)
            img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
            img_input_yuv.tofile(f_temp)
            f_temp.close()   

            img_rec, bits = self.vtm_yuv_compress(frame_idx_str)
        elif self.iframe_format=='RGB444':
            f_temp=open(self.img_output_dir+'frame'+frame_idx_str+'_org.rgb','w')
            img_input_rgb = cv2.merge(reference_frame)
            img_input_rgb = img_input_rgb.transpose(2, 0, 1)   # 3xHxW
            img_input_rgb.tofile(f_temp)
            f_temp.close()

            img_rec, bits = self.vtm_rgb_compress(frame_idx_str)
        else:
            raise NotImplementedError(f"Coding in format {self.frame_format} not implemented")
  
        return img_rec, bits
  

class RefereceImageDecoder:
    '''Wrapper for Image decoders [VTM]'''
    def __init__(self,img_input_dir:str,qp:int=4,iframe_format:str='YUV420', width:int=256,height:int=256, dec_name='vtm',device='cpu'):
        self.img_input_dir = img_input_dir
        self.iframe_format = iframe_format
        self.width = width
        self.height = height
        self.qp = qp
        self.device = device
  
        self.dec_name = dec_name

    def vtm_yuv_decompress(self, frame_idx:int):
        os.system("./image_codecs/vtm/decode.sh "+self.img_input_dir+'frame'+str(frame_idx))
        bin_file=self.img_input_dir+'frame'+str(frame_idx)+'.bin'
        bits=os.path.getsize(bin_file)*8

        #  read the rec frame (yuv420) and convert to rgb444
        rec_ref_yuv=yuv420_to_rgb444(self.img_input_dir+'frame'+str(frame_idx)+'_dec.yuv', self.width, self.height, 0, 1, False, False)                  
        return rec_ref_yuv[0]  , bits                                    

    def vtm_rgb_decompress(self, frame_idx:int):
        os.system("./image_codecs/vtm/decode_rgb444.sh "+self.img_input_dir+'frame'+str(frame_idx))
        bin_file=self.img_input_dir+'frame'+str(frame_idx)+'.bin'
        bits=os.path.getsize(bin_file)*8

        f_temp=open(self.img_input_dir+'frame'+str(frame_idx)+'_dec.rgb','rb')
        img_rec=np.fromfile(f_temp,np.uint8,3*self.height*self.width).reshape((3,self.height,self.width))   # 3xHxW RGB            
        return img_rec, bits
    
    def decompress(self, frame_idx:int):
        frame_idx_str = str(frame_idx).zfill(4)
        if self.iframe_format == 'YUV420':
            img_rec, bits = self.vtm_yuv_decompress(frame_idx_str)
        elif self.iframe_format == 'RGB444':
            img_rec, bits = self.vtm_rgb_decompress(frame_idx_str)
        else:
            raise NotImplementedError(f"Frame format '{self.iframe_format}' not implemented!")
        return img_rec, bits

        
####################################################################################################
class SEIEncoder:

    def __init__(self, model, coordinate_Qstep, matrix_Qstep, frames, device='cpu'):
        self.model = model

        self.gfv_id=[]
        self.gfv_cnt=[]
        self.gfv_DrivePicFusionFlag=[]
        self.gfv_LowConfidenceFaceParameterFlag=[]
        self.gfv_CoordinatePresentFlag=[]
        self.gfv_CoordinateQuantizationFactor=[]
        self.gfv_CoordinatePredFlag=[]
        self.gfv_oordinatePointNum=[]
        self.gfv_3DCoordinateFlag=[]
        self.gfv_MatrixElement = []
        self.gfv_XCoordinate=[]
        self.gfv_YCoordinate=[]
        self.gfv_ZCoordinateMaxValue=[]
        self.gfv_ZCoordinate=[]
        
        self.gfv_MatrixPresentFlag=[]
        self.gfv_MatrixElementPrecisionFactor=[]
        self.gfv_MatrixPredFlag=[]
        self.gfv_NumMatrixType=[]
        self.gfv_MatrixTypeIdx=[]
        self.gfv_NumMatricestoNumKpsFlag=[]
        self.gfv_NumMatricesInfo=[]
        self.gfv_Matrix3DSpaceFlag=[]
        self.gfv_NumMatrices=[]
        self.gfv_MatrixWidth=[]
        self.gfv_MatrixHeight=[]
        self.cfgcontent_basicinfo = ["SEIGenerativeFaceVideoEnabled: 1" + '\n' 
                                        "SEIGenerativeFaceVideoNumber: " +str(frames)+ '\n'

                                        "SEIGenerativeFaceVideoBasePicFlag: 1" + '\n' 
                                        "SEIGenerativeFaceVideoNNPresentFlag: 1" + '\n' 
                                        "SEIGenerativeFaceVideoNNBaseFlag: 1" + '\n'  
                                        "SEIGenerativeFaceVideoNNModeIdc: 1" + '\n' 
                                        "SEIGenerativeFaceVideoNNTagURI: gfv_uri_tag" + '\n' 
                                        "SEIGenerativeFaceVideoNNURI: gfv_uri" + '\n'    
                                        "SEIGenerativeFaceVideoPayloadFilename: test.nnr"+ '\n'
                                         
                                         "SEIGenerativeFaceVideoChromaKeyInfoPresentFlag: 0" + '\n'
                                         "SEIGenerativeFaceVideoChromaKeyValuePresentFlag: 0 0 0" + '\n'
                                         "SEIGenerativeFaceVideoChromaKeyValue: 22 144 255" + '\n'
                                         "SEIGenerativeFaceVideoChromaKeyThrPresentFlag: 1 1" + '\n'
                                         "SEIGenerativeFaceVideoChromaKeyThrValue: 21 44" + '\n'
                                    ]


        if self.model == 'CFTE' or self.model == 'CFTE_Lightweight':
            self.gfv_cnt = [i for i in range(frames)]
            self.gfv_id = [1] * frames
            self.gfv_DrivePicFusionFlag = [0] * frames #x
            self.gfv_LowConfidenceFaceParameterFlag = [0] * frames #x
            self.gfv_CoordinatePresentFlag = [0] * frames
            self.gfv_CoordinateQuantizationFactor = [0] * frames
            self.gfv_CoordinatePredFlag = [0] * frames
            self.gfv_oordinatePointNum = [0] * frames
            self.gfv_3DCoordinateFlag = [0] * frames
            self.gfv_XCoordinate = [0] * frames
            self.gfv_YCoordinate = [0] * frames
            self.gfv_ZCoordinateMaxValue = [0] * frames
            self.gfv_ZCoordinate = [0] * frames
            self.gfv_MatrixPresentFlag = [1] * frames
            self.gfv_MatrixElementPrecisionFactor= [matrix_Qstep] * frames
            self.gfv_MatrixPredFlag = [0] + [1]* (frames-1)
            self.gfv_NumMatrixType = [1] * frames
            self.gfv_MatrixTypeIdx = [7] * frames
            self.gfv_NumMatricestoNumKpsFlag = [0] * frames
            self.gfv_NumMatricesInfo = [0] * frames
            self.gfv_Matrix3DSpaceFlag = [0] * frames
            self.gfv_NumMatrices = [1] * frames
            self.gfv_MatrixWidth = [4] * frames
            self.gfv_MatrixHeight = [4] * frames
            
            
        elif self.model == 'FOMM':
            self.gfv_cnt = [i for i in range(frames)]
            self.gfv_id = [1] * frames
            self.gfv_DrivePicFusionFlag = [0] * frames
            self.gfv_LowConfidenceFaceParameterFlag = [0] * frames
            self.gfv_CoordinatePresentFlag = [1] * frames
            self.gfv_CoordinateQuantizationFactor = [coordinate_Qstep] * frames
            self.gfv_CoordinatePredFlag = [0] + [1]* (frames-1)
            self.gfv_oordinatePointNum = [10] * frames
            self.gfv_3DCoordinateFlag = [0] * frames
            self.gfv_ZCoordinateMaxValue=[0] * frames
            self.gfv_ZCoordinate=[0] * frames
            self.gfv_MatrixPresentFlag = [1] * frames
            self.gfv_MatrixElementPrecisionFactor= [matrix_Qstep] * frames
            self.gfv_MatrixPredFlag = [0] + [1]* (frames-1)
            self.gfv_NumMatrixType = [1] * frames
            self.gfv_MatrixTypeIdx = [0] * frames
            self.gfv_NumMatricestoNumKpsFlag = [1] * frames
            self.gfv_NumMatricesInfo = [0] * frames
            self.gfv_Matrix3DSpaceFlag = [0] * frames
            self.gfv_NumMatrices = [0] * frames
            self.gfv_MatrixWidth = [0] * frames
            self.gfv_MatrixHeight = [0] * frames
           
            
        elif self.model == 'DAC' or self.model == 'HDAC':
            self.gfv_cnt = [i for i in range(frames)]
            self.gfv_id = [1] * frames
            self.gfv_DrivePicFusionFlag = [0] * frames
            self.gfv_LowConfidenceFaceParameterFlag = [0] * frames
            self.gfv_CoordinatePresentFlag = [1] * frames
            self.gfv_CoordinateQuantizationFactor = [coordinate_Qstep] * frames
            self.gfv_CoordinatePredFlag = [0] + [1]* (frames-1)
            self.gfv_oordinatePointNum = [10] * frames
            self.gfv_3DCoordinateFlag = [0] * frames
            self.gfv_ZCoordinateMaxValue=[0] * frames
            self.gfv_ZCoordinate=[0] * frames
            self.gfv_MatrixPresentFlag = [0] * frames
            self.gfv_MatrixElementPrecisionFactor= [0] * frames
            self.gfv_MatrixPredFlag = [0] * frames
            self.gfv_NumMatrixType = [0] * frames
            self.gfv_MatrixTypeIdx = [0] * frames
            self.gfv_NumMatricestoNumKpsFlag = [0] * frames
            self.gfv_NumMatricesInfo = [0] * frames
            self.gfv_Matrix3DSpaceFlag = [0] * frames
            self.gfv_NumMatrices = [0] * frames
            self.gfv_MatrixWidth = [0] * frames
            self.gfv_MatrixHeight = [0] * frames
            self.gfv_MatrixElement = [0] * frames

        elif self.model == 'FV2V':
            self.gfv_cnt = [i for i in range(frames)]
            self.gfv_id = [1] * frames
            self.gfv_DrivePicFusionFlag = [0] * frames
            self.gfv_LowConfidenceFaceParameterFlag = [0] * frames
            self.gfv_CoordinatePresentFlag = [1] * frames
            self.gfv_CoordinateQuantizationFactor = [coordinate_Qstep] * frames
            self.gfv_CoordinatePredFlag = [0] + [1]* (frames-1)
            self.gfv_oordinatePointNum = [15] * frames
            self.gfv_3DCoordinateFlag = [1] * frames
            self.gfv_ZCoordinateMaxValue=[1] * frames
            self.gfv_MatrixPresentFlag = [1] * frames
            self.gfv_MatrixElementPrecisionFactor= [matrix_Qstep] * frames
            self.gfv_MatrixPredFlag = [0] + [1]* (frames-1)
            self.gfv_NumMatrixType = [1] * frames
            self.gfv_MatrixTypeIdx = [8] * frames
            self.gfv_NumMatricestoNumKpsFlag = [0] * frames
            self.gfv_NumMatricesInfo = [0] * frames
            self.gfv_Matrix3DSpaceFlag = [0] * frames
            self.gfv_NumMatrices = [1] * frames
            self.gfv_MatrixWidth = [1] * frames
            self.gfv_MatrixHeight = [12] * frames

    def update(self,kp_value):
        if self.model == 'CFTE' or self.model == 'CFTE_Lightweight':
            self.gfv_MatrixElement.append(kp_value)    
        elif self.model == 'FOMM':
            kp_value_frame = kp_value[:20]
            kp_jacobian_frame = kp_value[20:]
            self.gfv_XCoordinate.append(kp_value_frame[::2])
            self.gfv_YCoordinate.append(kp_value_frame[1::2])
            self.gfv_MatrixElement.append(kp_jacobian_frame)
        elif self.model == 'DAC' or self.model == 'HDAC':
            self.gfv_XCoordinate.append(kp_value[::2])
            self.gfv_YCoordinate.append(kp_value[1::2])
        elif self.model=='FV2V':
            rot_t_frame = kp_value[:12]
            exp_frame = kp_value[12:]
            self.gfv_XCoordinate.append(exp_frame[::3])
            self.gfv_YCoordinate.append(exp_frame[1::3])
            self.gfv_ZCoordinate.append(exp_frame[2::3])
            self.gfv_MatrixElement.append(rot_t_frame)


        
    def prepare_to_code(self):
        self.cfgcontent_coordinate_matrix_info = ["SEIGenerativeFaceVideoId: " +str(self.list_to_cmd(self.gfv_id)) + '\n'
                                                    "SEIGenerativeFaceVideoCnt: " + str(self.list_to_cmd(self.gfv_cnt)) + '\n'
                                                    "SEIGenerativeFaceVideoDrivePicFusionFlag: "+str(self.list_to_cmd(self.gfv_DrivePicFusionFlag))  + '\n'
                                                    "SEIGenerativeFaceVideoLowConfidenceFaceParameterFlag: "+ str(self.list_to_cmd(self.gfv_LowConfidenceFaceParameterFlag))  + '\n'

                                                    "SEIGenerativeFaceVideoCoordinatePresentFlag: "+ str(self.list_to_cmd(self.gfv_CoordinatePresentFlag)) +'\n'
                                                    "SEIGenerativeFaceVideoCoordinateQuantizationFactor: "+ str(self.list_to_cmd(self.gfv_CoordinateQuantizationFactor)) +'\n'
                                                    "SEIGenerativeFaceVideoCoordinatePredFlag: " + str(self.list_to_cmd(self.gfv_CoordinatePredFlag)) +'\n'
                                                    "SEIGenerativeFaceVideo3DCoordinateFlag: "+ str(self.list_to_cmd(self.gfv_3DCoordinateFlag)) +'\n'
                                                    "SEIGenerativeFaceVideoCoordinatePointNum: "+ str(self.list_to_cmd(self.gfv_oordinatePointNum)) +'\n'
                                                    "SEIGenerativeFaceVideoXCoordinate: "+ str(self.list_to_cmd(self.gfv_XCoordinate)) +'\n'
                                                    "SEIGenerativeFaceVideoYCoordinate: "+ str(self.list_to_cmd(self.gfv_YCoordinate)) +'\n'
                                                    "SEIGenerativeFaceVideoZCoordinateMaxValue: "+ str(self.list_to_cmd(self.gfv_ZCoordinateMaxValue)) +'\n'
                                                    "SEIGenerativeFaceVideoZCoordinate: "+ str(self.list_to_cmd(self.gfv_ZCoordinate)) +'\n'

                                                    "SEIGenerativeFaceVideoMatrixPresentFlag: "+ str(self.list_to_cmd(self.gfv_MatrixPresentFlag))  + "\n"
                                                    "SEIGenerativeFaceVideoMatrixElementPrecisionFactor: " + str(self.list_to_cmd(self.gfv_MatrixElementPrecisionFactor)) + "\n"
                                                    "SEIGenerativeFaceVideoMatrixPredFlag: " + str(self.list_to_cmd(self.gfv_MatrixPredFlag))  + "\n"
                                                    "SEIGenerativeFaceVideoNumMatrixType: " + str(self.list_to_cmd(self.gfv_NumMatrixType)) + "\n"
                                                    "SEIGenerativeFaceVideoMatrixTypeIdx: " + str(self.list_to_cmd(self.gfv_MatrixTypeIdx)) + "\n"
                                                    "SEIGenerativeFaceVideoNumMatricestoNumKpsFlag: " + str(self.list_to_cmd(self.gfv_NumMatricestoNumKpsFlag)) + "\n"
                                                    "SEIGenerativeFaceVideoNumMatricesInfo: " + str(self.list_to_cmd(self.gfv_NumMatricesInfo)) + "\n"
                                                    "SEIGenerativeFaceVideoMatrix3DSpaceFlag: " + str(self.list_to_cmd(self.gfv_Matrix3DSpaceFlag)) + "\n"
                                                    "SEIGenerativeFaceVideoNumMatrices: "+ str(self.list_to_cmd(self.gfv_NumMatrices)) + "\n"
                                                    "SEIGenerativeFaceVideoMatrixWidth: " + str(self.list_to_cmd(self.gfv_MatrixWidth)) + "\n"
                                                    "SEIGenerativeFaceVideoMatrixHeight: " + str(self.list_to_cmd(self.gfv_MatrixHeight)) + "\n" 
                                                    'SEIGenerativeFaceVideoMatrixElement: '+ str(self.list_to_cmd(self.gfv_MatrixElement))+ "\n"]     



    def list_to_cmd(self,input):
        return eval('%s'%repr(str(input)).replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '').replace(',', ' ')  )                  

    
def split_text_file(input_file, output_dir):
    with open(input_file, 'r') as file:
        data = file.read()
    
    delimiter = '---------------------------------\n'  # 分割符
    
    parts = data.split(delimiter)
    for i, part in enumerate(parts):
        if part.strip():  # 忽略空字符串
            with open(f"{output_dir}/gfv_{i-1}.txt", 'w') as out_file:
                out_file.write(part)


def listformat_CFTE(true_ae_number, num_channel,height, width):    
    listnum=int(height)  ###########
    slidelist=int(width)
    length=listnum*slidelist

    true_ae_number=eval('[%s]'%repr(true_ae_number).replace('[', '').replace(']', '').replace("'", ""))  
    reallatentvalue=(np.array(true_ae_number)).tolist()
    
    latentformat=[]
    for channel_size in range(int(num_channel)):
        
        latentformat_channel=[]
        for slideformat in range(0,slidelist):
            latentformatslide=reallatentvalue[slideformat*listnum:(slideformat+1)*listnum]
            latentformat_channel.extend([latentformatslide])
        del reallatentvalue[0:length]
        latentformat.extend([latentformat_channel])
    latentformat=[latentformat]
    latentformat=str(latentformat)
    latentformat="".join(latentformat.split())
        
    return latentformat  



def listformat_FOMM(true_ae_number):
    
    group=60
    ##按照模型所需格式进行重组的相关参数设置
    listsplit=int(group/(1+2))
    listnum=2
    slidelistnum=int((group/(1+2)*1)/listnum)
    ##对恢复得到的dataRec进行切片，每16个组成一组，并进行帧间补偿    

    ### 第二帧（帧间重建第一帧 based on VVC frame）    
    reallatentvalue=(np.array(true_ae_number)).tolist() #(np.array(refframetensor)+np.array(true_ae_number)).tolist()

    #按照模型所需格式重组
    ###kp
    reallatentvalue_split1=reallatentvalue[0*listsplit:1*listsplit]
    latentformat1=[]
    for slideformat1 in range(0,slidelistnum):
        latentformatslide1=reallatentvalue_split1[slideformat1*listnum:(slideformat1+1)*listnum]
        latentformat1.extend([latentformatslide1])
    latentformat1=[latentformat1]
    #print(latentformat1)        

    ###jocobi
    reallatentvalue_split2=reallatentvalue[1*listsplit:3*listsplit]
    latentformat2=[]
    for slideformat2 in range(0,int((group/(1+2)*2)/listnum)):
        latentformatslide2=reallatentvalue_split2[slideformat2*listnum:(slideformat2+1)*listnum]
        latentformat2.extend([latentformatslide2])
    
    latentformat2_final=[]
    for sub in range(int(len(latentformat2)/2)):
        latentformatslide2_final= [latentformat2[sub*2],latentformat2[sub*2+1]]
        latentformat2_final.append(latentformatslide2_final)
    
    latentformat2_final=[latentformat2_final]

    ###保存恢复的真实的数据格式
    latentformat1=str(latentformat1)
    latentformat1="".join(latentformat1.split())
    latentformat2_final=str(latentformat2_final)
    latentformat2_final="".join(latentformat2_final.split())        
    return latentformat1,latentformat2_final


def listformat_FV2V(true_ae_number):
    
    group=57
    ##按照模型所需格式进行重组的相关参数设置
    listnum=3
    ##对恢复得到的dataRec进行切片，每16个组成一组，并进行帧间补偿    

    ### 第二帧（帧间重建第一帧 based on VVC frame）    
    # reallatentvalue=(np.array(refframetensor)+np.array(true_ae_number)).tolist()

    reallatentvalue=(np.array(true_ae_number)).tolist()
    
    #按照模型所需格式重组
    
    ###kp_mat 3*3 matrix, e.g. [[[63,25,28],[39,63,30],[36,33,64]]] 
    reallatentvalue_split1=reallatentvalue[0:9]
    latentformat1=[]
    for slideformat1 in range(0,3):
        latentformatslide1=reallatentvalue_split1[slideformat1*listnum:(slideformat1+1)*listnum]
        latentformat1.extend([latentformatslide1])
    kp_mat_value=[latentformat1]
    
    ###kp_t_value 1*3 matrix e.g. [[30,25,38]]
    reallatentvalue_split2=reallatentvalue[9:12]
    kp_t_value=[reallatentvalue_split2]
    
    ###kp_exp_value 1*45 matrix e.g. [[30,25,38]]
    reallatentvalue_split3=reallatentvalue[12:57]
    kp_exp_value=[reallatentvalue_split3] 

    ###保存恢复的真实的数据格式
    kp_mat_value=str(kp_mat_value)
    kp_mat_value="".join(kp_mat_value.split())
    kp_t_value=str(kp_t_value)
    kp_t_value="".join(kp_t_value.split())    
    kp_exp_value=str(kp_exp_value)
    kp_exp_value="".join(kp_exp_value.split())        
    

    return kp_mat_value,kp_t_value,kp_exp_value


def listformat_DAC(true_ae_number):
    
    group=60
    ##按照模型所需格式进行重组的相关参数设置
    listsplit=int(group/(1+2))
    listnum=2
    slidelistnum=int((group/(1+2)*1)/listnum)
    ##对恢复得到的dataRec进行切片，每16个组成一组，并进行帧间补偿    

    ### 第二帧（帧间重建第一帧 based on VVC frame）    
    reallatentvalue=(np.array(true_ae_number)).tolist() #(np.array(refframetensor)+np.array(true_ae_number)).tolist()

    #按照模型所需格式重组
    ###kp
    reallatentvalue_split1=reallatentvalue[0*listsplit:1*listsplit]
    latentformat1=[]
    for slideformat1 in range(0,slidelistnum):
        latentformatslide1=reallatentvalue_split1[slideformat1*listnum:(slideformat1+1)*listnum]
        latentformat1.extend([latentformatslide1])
    latentformat1=[latentformat1]
    #print(latentformat1)        


    ###保存恢复的真实的数据格式
    latentformat1=str(latentformat1)
    latentformat1="".join(latentformat1.split())      
    return latentformat1

def sei_decode(f, model, device):

   
    if model=='CFTE' or model=='CFTE_Lightweight':
        current_driving_string=[]
        lines = f.readlines()  
        for line in lines:
            line = line.strip()

#             if 'gfv_id:' in line:
#                 frame=int(line.strip('gfv_id: '))

#             if 'gfv_num_matrix_type:' in line:
#                 gfv_num_matrix_type=int(line.strip('gfv_num_matrix_type: '))

#             if 'gfv_num_matrices_minus1:' in line:
#                 gfv_num_matrices=int(line.strip('gfv_num_matrices_minus1: ')) +1 #/ frame

#             if 'gfv_matrix_width_minus1:' in line:
#                 gfv_matrix_width=int(line.strip('gfv_matrix_width_minus1: ')) +1 

#             if 'gfv_matrix_height_minus1:' in line:
#                 gfv_matrix_height=int(line.strip('gfv_matrix_height_minus1: ')) +1

            if 'gfv_MatrixElement:' in line:
                current_driving_vector=line.strip('gfv_MatrixElement: ')

                current_driving_string.append(current_driving_vector.replace(" ",","))
                
        
        matrix_vector=str(current_driving_string).replace("'",'')
        matrix_vector=eval('[%s]'%repr(matrix_vector).replace('[', '').replace(']', '').replace("'", ""))  
        matrix_vector=(np.array(matrix_vector)).tolist()      
        f.close()
        kp_integer=listformat_CFTE(matrix_vector, 1, 4, 4)               
        kp_integer=json.loads(str(kp_integer))
        kp_value=torch.Tensor(kp_integer).to(device)          
        dict={}
        dict['value']=kp_value  
        kp_value=dict  
        


#     if model=='CFTE' or model=='CFTE_Lightweight':
#         current_driving_string=[]
#         lines = f.readlines()  
#         for line in lines:
#             line = line.strip()

#             if 'gfv_id:' in line:
#                 frame=int(line.strip('gfv_id: '))

#             if 'gfv_num_matrix_type:' in line:
#                 gfv_num_matrix_type=int(line.strip('gfv_num_matrix_type: '))

#             if 'gfv_num_matrices_minus1:' in line:
#                 gfv_num_matrices=int(line.strip('gfv_num_matrices_minus1: ')) +1 #/ frame

#             if 'gfv_matrix_width_minus1:' in line:
#                 gfv_matrix_width=int(line.strip('gfv_matrix_width_minus1: ')) +1 

#             if 'gfv_matrix_height_minus1:' in line:
#                 gfv_matrix_height=int(line.strip('gfv_matrix_height_minus1: ')) +1

#             if 'gfv_MatrixElement:' in line:
#                 current_driving_vector=line.strip('gfv_MatrixElement: ')

#                 current_driving_string.append(current_driving_vector.replace(" ",","))
                
        
#         matrix_vector=str(current_driving_string).replace("'",'')
#         matrix_vector=eval('[%s]'%repr(matrix_vector).replace('[', '').replace(']', '').replace("'", ""))  
#         matrix_vector=(np.array(matrix_vector)).tolist()      
#         f.close()
#         kp_integer=listformat_CFTE(matrix_vector, gfv_num_matrices, gfv_matrix_height, gfv_matrix_width)               
#         kp_integer=json.loads(str(kp_integer))
#         kp_value=torch.Tensor(kp_integer).to(device)          
#         dict={}
#         dict['value']=kp_value  
#         kp_value=dict  
        
        
        
        
    elif model == 'FOMM':
        current_driving_string=[]
        current_driving_coordinate_x=[]
        current_driving_coordinate_y=[]
        lines = f.readlines()  
        for line in lines:
            line = line.strip()

                
            if 'gfv_coordinate_Xtensor:' in line:
                current_driving_coordinate_vector_x=line.strip('gfv_coordinate_Xtensor: ')  
                current_driving_coordinate_x.append(current_driving_coordinate_vector_x.replace(" ",","))

            if 'gfv_coordinate_Ytensor:' in line:
                current_driving_coordinate_vector_y=line.strip('gfv_coordinate_Ytensor: ')                          
                current_driving_coordinate_y.append(current_driving_coordinate_vector_y.replace(" ",","))


            if 'gfv_MatrixElement:' in line:
                current_driving_matrix_vector=line.strip('gfv_MatrixElement: ')
                current_driving_string.append(current_driving_matrix_vector.replace(" ",","))
                

        matrix_vector=str(current_driving_string).replace("'",'')
        matrix_vector=eval('[%s]'%repr(matrix_vector).replace('[', '').replace(']', '').replace("'", ""))  
        matrix_vector=(np.array(matrix_vector)).tolist()   

        coordinate_vector_x=str(current_driving_coordinate_x).replace("'",'')
        coordinate_vector_x=eval('[%s]'%repr(coordinate_vector_x).replace('[', '').replace(']', '').replace("'", ""))  
        coordinate_vector_x=(np.array(coordinate_vector_x)).tolist()               

        coordinate_vector_y=str(current_driving_coordinate_y).replace("'",'')
        coordinate_vector_y=eval('[%s]'%repr(coordinate_vector_y).replace('[', '').replace(']', '').replace("'", ""))  
        coordinate_vector_y=(np.array(coordinate_vector_y)).tolist()                

        coordinate_vector = [item for pair in zip(coordinate_vector_x, coordinate_vector_y) for item in pair]

        kp_value_jocobi=coordinate_vector+matrix_vector ###20+40
        f.close()

        
        coordinate_vector,matrix_vector=listformat_FOMM(kp_value_jocobi)              

        kp_coordinate_driving=json.loads(str(coordinate_vector))
        # print(kp_coordinate_driving)
        kp_coordinate_driving=torch.Tensor(kp_coordinate_driving).to(device)  
        
        kp_matrix_driving=json.loads(str(matrix_vector))
        kp_matrix_driving=torch.Tensor(kp_matrix_driving).to(device)                      
        dict={}
        dict['value']=kp_coordinate_driving  
        dict['jacobian']=kp_matrix_driving                      
        kp_value=dict      
    
    elif model == 'DAC' or model == 'HDAC':
        reference_driving_string=[]
        reference_driving_coordinate_x=[]
        reference_driving_coordinate_y=[]
        
        lines = f.readlines()  
        for line in lines:
            line = line.strip()

                
            if 'gfv_coordinate_Xtensor:' in line:
                reference_driving_coordinate_vector_x=line.strip('gfv_coordinate_Xtensor: ')  
                reference_driving_coordinate_x.append(reference_driving_coordinate_vector_x.replace(" ",","))

            if 'gfv_coordinate_Ytensor:' in line:
                reference_driving_coordinate_vector_y=line.strip('gfv_coordinate_Ytensor: ')                          
                reference_driving_coordinate_y.append(reference_driving_coordinate_vector_y.replace(" ",","))


        coordinate_vector_x=str(reference_driving_coordinate_x).replace("'",'')
        coordinate_vector_x=eval('[%s]'%repr(coordinate_vector_x).replace('[', '').replace(']', '').replace("'", ""))  
        coordinate_vector_x=(np.array(coordinate_vector_x)).tolist()               

        coordinate_vector_y=str(reference_driving_coordinate_y).replace("'",'')
        coordinate_vector_y=eval('[%s]'%repr(coordinate_vector_y).replace('[', '').replace(']', '').replace("'", ""))  
        coordinate_vector_y=(np.array(coordinate_vector_y)).tolist()                

        coordinate_vector = [item for pair in zip(coordinate_vector_x, coordinate_vector_y) for item in pair]

        kp_value=coordinate_vector ###20+40
        f.close()

        
        coordinate_vector=listformat_DAC(kp_value)            

        kp_coordinate_reference=json.loads(str(coordinate_vector))
        kp_coordinate_reference=torch.Tensor(kp_coordinate_reference).to(device)  
        
            
        dict={}
        dict['value']=kp_coordinate_reference   
        
        kp_value=dict    

    elif model == 'FV2V':
       
        reference_driving_string=[]
        reference_driving_coordinate_x=[]
        reference_driving_coordinate_y=[]
        reference_driving_coordinate_z=[]
        
        lines = f.readlines()  
        for line in lines:
            line = line.strip()

                
            if 'gfv_coordinate_Xtensor:' in line:
                reference_driving_coordinate_vector_x=line.strip('gfv_coordinate_Xtensor: ')  
                reference_driving_coordinate_x.append(reference_driving_coordinate_vector_x.replace(" ",","))

            if 'gfv_coordinate_Ytensor:' in line:
                reference_driving_coordinate_vector_y=line.strip('gfv_coordinate_Ytensor: ')                          
                reference_driving_coordinate_y.append(reference_driving_coordinate_vector_y.replace(" ",","))


            if 'gfv_coordinate_Ztensor:' in line:
                reference_driving_coordinate_vector_z=line.strip('gfv_coordinate_Ztensor: ')                          
                reference_driving_coordinate_z.append(reference_driving_coordinate_vector_z.replace(" ",","))                            
                
                
            if 'gfv_MatrixElement:' in line:
                reference_driving_matrix_vector=line.strip('gfv_MatrixElement: ')
                reference_driving_string.append(reference_driving_matrix_vector.replace(" ",","))
                

        matrix_vector=str(reference_driving_string).replace("'",'')
        matrix_vector=eval('[%s]'%repr(matrix_vector).replace('[', '').replace(']', '').replace("'", ""))  
        matrix_vector=(np.array(matrix_vector)).tolist()   

        coordinate_vector_x=str(reference_driving_coordinate_x).replace("'",'')
        coordinate_vector_x=eval('[%s]'%repr(coordinate_vector_x).replace('[', '').replace(']', '').replace("'", ""))  
        coordinate_vector_x=(np.array(coordinate_vector_x)).tolist()               

        coordinate_vector_y=str(reference_driving_coordinate_y).replace("'",'')
        coordinate_vector_y=eval('[%s]'%repr(coordinate_vector_y).replace('[', '').replace(']', '').replace("'", ""))  
        coordinate_vector_y=(np.array(coordinate_vector_y)).tolist()     
        
        coordinate_vector_z=str(reference_driving_coordinate_z).replace("'",'')
        coordinate_vector_z=eval('[%s]'%repr(coordinate_vector_z).replace('[', '').replace(']', '').replace("'", ""))  
        coordinate_vector_z=(np.array(coordinate_vector_z)).tolist()                         

        coordinate_vector = [item for pair in zip(coordinate_vector_x, coordinate_vector_y, coordinate_vector_z) for item in pair]

        kp_value_jocobi=matrix_vector+coordinate_vector ###20+40
        f.close()
        kp_mat_value,kp_t_value,kp_exp_value=listformat_FV2V(kp_value_jocobi)  

        kp_mat_value_current=json.loads(str(kp_mat_value))
        kp_mat_value_current=torch.Tensor(kp_mat_value_current).to(device)  
        
        kp_t_value_current=json.loads(str(kp_t_value))
        kp_t_value_current=torch.Tensor(kp_t_value_current).to(device)  
        
        kp_exp_value_current=json.loads(str(kp_exp_value))
        kp_exp_value_current=torch.Tensor(kp_exp_value_current).to(device)                      
        
        dict={}
        dict['rot_mat']=kp_mat_value_current  
        dict['t']=kp_t_value_current  
        dict['exp']=kp_exp_value_current  
        kp_value=dict    



    return kp_value
