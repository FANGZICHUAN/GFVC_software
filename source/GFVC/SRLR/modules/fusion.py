import torch
from torch import nn
import torch.nn.functional as F
from GFVC.SRLR.modules.util import *
from GFVC.SRLR.modules.flow_util import *


class GeneEncoder(nn.Module):
    """
    Hourglass Encoder
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(GeneEncoder, self).__init__()        
        
        down_blocks = []
        for i in range(num_blocks):
            
            down_blocks.append(DownBlock2d(min(max_features, block_expansion * (2 ** (i + 1))),
                                           min(max_features, block_expansion * (2 ** (i + 2))),
                                           kernel_size=3, padding=1))           
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            out=down_block(outs[-1])
            outs.append(out)      
            
        return outs


class GeneDecoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(GeneDecoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 2)))
            out_filters = min(max_features, block_expansion * (2 ** (i + 1)))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
            
        self.up_blocks = nn.ModuleList(up_blocks)        
        self.out_filters = (block_expansion + in_features)*2
        
        self.down_1_8 = AntiAliasInterpolation2d(1, 0.125).cuda()   #0.125
        self.down_2_8 = AntiAliasInterpolation2d(2, 0.125).cuda()           
        
    def forward(self, x, flow, occlusion):

        flow_blocks = []  
        occlusion_blocks = []
        for i in range(3): 
            self.downflow=AntiAliasInterpolation2d(2, 1/2**(i)).cuda() #1/2**(i)
            self.occlusion=AntiAliasInterpolation2d(1, 1/2**(i)).cuda()            
            denseflow=self.downflow(flow.permute(0,3,1,2)).permute(0,2,3,1)
            denseocclusion=self.occlusion(occlusion)
            flow_blocks.append(denseflow)  
            occlusion_blocks.append(denseocclusion)
        
        out = x.pop()
        denseflow_8=self.down_2_8(flow.permute(0,3,1,2)).permute(0,2,3,1)
        denseocclusion_8=self.down_1_8(occlusion)  
        out_deform_1 =  F.grid_sample(out, denseflow_8) * denseocclusion_8  

        for i, up_block in enumerate(self.up_blocks):
            out = up_block(out)
            skip = x.pop()
            flow=flow_blocks.pop()
            occlusion=occlusion_blocks.pop()
            out_deform =  F.grid_sample(skip, flow) * occlusion       
            out = torch.cat([out, out_deform], dim=1)
        return out    


class GeneHourglass2d(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion=32, in_features=32, num_blocks=3, max_features=256):
        super(GeneHourglass2d, self).__init__()
        self.geneencoder = GeneEncoder(block_expansion, in_features, num_blocks, max_features)
        self.genedecoder = GeneDecoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.genedecoder.out_filters

        self.final = nn.Conv2d(self.out_filters, 3, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, x, flow, occlusion):
        out=self.geneencoder(x)
        out=self.genedecoder(out, flow, occlusion)
        out = self.final(out)
        out = F.sigmoid(out)
        return out    

    
    
class SPADE_Flow(nn.Module):
    def __init__(self):
        super(SPADE_Flow, self).__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256
        
        self.first = SameBlock2d(6, 32, kernel_size=(7, 7), padding=(3, 3))
        
        meshmotiondown_blocks = []
        num_down_blocks=3
        max_features=512
        block_expansion=32
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            meshmotiondown_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.motiondown_blocks = nn.ModuleList(meshmotiondown_blocks)        
        
        self.seg = DownBlock2d(ic, ic, kernel_size=(3, 3), padding=(1, 1))
        # self.seg = DownBlock2d(ic+2, ic, kernel_size=(3, 3), padding=(1, 1))
        
        #self.seg = DownBlock2d(2, ic, kernel_size=(3, 3), padding=(1, 1))
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # self.up_0 = SPADEResnetBlock(2 * ic+3, ic, norm_G, label_nc)
        # self.up_1 = SPADEResnetBlock(ic+2, oc, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)        
        
        self.up = nn.Upsample(scale_factor=2)

        
        self.down_2_2 = AntiAliasInterpolation2d(2, 0.5).cuda()  
        self.down_3_4 = AntiAliasInterpolation2d(3, 0.25).cuda()  
        self.down_2_8 = AntiAliasInterpolation2d(2, 0.125).cuda()  
        
        self.meshflow = nn.Conv2d(oc, 2, kernel_size=(7, 7), padding=(3, 3))          
        self.down_sample_flow = AntiAliasInterpolation2d(2, 0.25).cuda()  
        
        self.meshocclusion = nn.Conv2d(oc, 1, kernel_size=(7, 7), padding=(3, 3))            
        self.down_sample_occlusion = AntiAliasInterpolation2d(1, 0.25).cuda()          
        self.sigmoid = nn.Sigmoid()       
    
    def forward(self, source_image, deformed_image):
        
        
        input=torch.cat([source_image, deformed_image], dim=1) #torch.Size([bs, 6, 256, 256]        
        
        out=self.first(input)
        #print(out.shape)
        
        for i in range(3):
            out = self.motiondown_blocks[i](out)         
        # print(out.shape) #torch.Size([bs, 256, 32, 32])
        
        seg = self.seg(out)
                
        x = self.fc(out)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)

        x = self.up(x)    #torch.Size([5, 512, 64, 64])
    ################################
        x = self.up_0(x, seg)   # torch.Size([5, 256, 64, 64])
        x = self.up(x)      #torch.Size([5, 64, 128, 128])
        x = self.up_1(x, seg)    #torch.Size([5, 64, 128, 128])
        x = self.up(x)   #torch.Size([5, 64, 256, 256])       
        flow=convert_flow_to_deformation(self.meshflow(x))
        occlusion=self.sigmoid(self.meshocclusion(x))
        
        return occlusion, flow        

    
    
class DepthAwareAttention_scale64(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(DepthAwareAttention_scale64,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        
        self.query_conv = nn.Conv2d(in_channels = in_dim*4 , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim*4 , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim*4 , out_channels = in_dim *4, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,source,feat):
        """
            inputs :
                source : input feature maps( B X C X W X H) 256,64,64
                driving : input feature maps( B X C X W X H) 256,64,64
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = source.size()
        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        # energy =  torch.bmm(proj_query,proj_key) # transpose check
        energy =  torch.matmul(proj_query,proj_key) # transpose check        
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.matmul(proj_value,attention.permute(0,2,1) )

        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + feat

        return out,attention         
    
class DepthAwareAttention_scale32(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(DepthAwareAttention_scale32,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim *8, out_channels = in_dim//4 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim*8 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,source,feat):
        """
            inputs :
                source : input feature maps( B X C X W X H) 256,64,64   512,16,16
                driving : input feature maps( B X C X W X H) 256,64,64  512,16,16
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = source.size()
        
        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.matmul(proj_query,proj_key) # transpose check #bmm
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.matmul(proj_value,attention.permute(0,2,1) ) ##bmm
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + feat

        return out,attention     
    
    
class DepthAwareAttention_scale16(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(DepthAwareAttention_scale16,self).__init__()
        self.chanel_in = in_dim ##64
        self.activation = activation 

        self.query_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim *8, out_channels = in_dim//4 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim*8 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,source,feat):
        """
            inputs :
                source_img : input feature maps( B X C X W X H)   512,16,16
                residual : input feature maps( B X C X W X H) 512,16,16
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = source.size()

        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.matmul(proj_query,proj_key) # transpose check bmm
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.matmul(proj_value,attention.permute(0,2,1) )  #bmm
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + feat

        return out,attention     
    
    
class DepthAwareAttention_scale8(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(DepthAwareAttention_scale8,self).__init__()
        self.chanel_in = in_dim ##64
        self.activation = activation 

        self.query_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim *8, out_channels = in_dim//4 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim*8 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,source,feat):
        """
            inputs :
                source_img : input feature maps( B X C X W X H)   512,16,16
                residual : input feature maps( B X C X W X H) 512,16,16
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = source.size()

        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.matmul(proj_query,proj_key) # transpose check bmm
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.matmul(proj_value,attention.permute(0,2,1) )  #bmm
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + feat

        return out,attention         
    
    
class DepthAwareAttention_scale4(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(DepthAwareAttention_scale4,self).__init__()
        self.chanel_in = in_dim ##64
        self.activation = activation 

        self.query_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim *8, out_channels = in_dim//4 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim*8 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,source,feat):
        """
            inputs :
                source_img : input feature maps( B X C X W X H)   512,16,16
                residual : input feature maps( B X C X W X H) 512,16,16
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = source.size()

        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.matmul(proj_query,proj_key) # transpose check bmm
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.matmul(proj_value,attention.permute(0,2,1) )  #bmm
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + feat

        return out,attention  
    
    
    
    
class DepthAwareAttention_scale24(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(DepthAwareAttention_scale24,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim *8, out_channels = in_dim//4 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim*8 , out_channels = in_dim*8 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
        self.conv = nn.Conv2d(in_channels=in_dim *8,out_channels=in_dim *8, kernel_size=(9, 9), stride=1, padding=0)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_dim *8, out_channels=in_dim *8, kernel_size=(5, 5), stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=in_dim *8, out_channels=in_dim *8, kernel_size=(5, 5), stride=1, padding=0)

            
    def forward(self,source,feat):
        """
            inputs :
                source : input feature maps( B X C X W X H) 256,64,64   512,16,16
                driving : input feature maps( B X C X W X H) 256,64,64  512,16,16
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        feat=self.conv(feat)
        # print(feat.shape) #torch.Size([bs, 512, 24, 24])
        
        m_batchsize,C,width ,height = source.size()
        
        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.matmul(proj_query,proj_key) # transpose check #bmm
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.matmul(proj_value,attention.permute(0,2,1) ) ##bmm
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + feat
        # print(out.shape) #torch.Size([bs, 512, 24, 24])
        out=self.deconv1(out)
        out=self.deconv2(out)

        # print(out.shape) #torch.Size([bs, 512, 32, 32])
        return out,attention     
    
    
    
    
class DepthAwareAttention_scale48(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(DepthAwareAttention_scale48,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        
        self.query_conv = nn.Conv2d(in_channels = in_dim*4 , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim*4 , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim*4 , out_channels = in_dim *4, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
        self.conv = nn.Conv2d(in_channels=in_dim*4,out_channels=in_dim*4, kernel_size=(17, 17), stride=1, padding=0)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_dim *4, out_channels=in_dim *4, kernel_size=(5, 5), stride=1, padding=0)        
        self.deconv2 = nn.ConvTranspose2d(in_channels=in_dim *4, out_channels=in_dim *4, kernel_size=(5, 5), stride=1, padding=0)        
        self.deconv3 = nn.ConvTranspose2d(in_channels=in_dim *4, out_channels=in_dim *4, kernel_size=(5, 5), stride=1, padding=0)        
        self.deconv4 = nn.ConvTranspose2d(in_channels=in_dim *4, out_channels=in_dim *4, kernel_size=(5, 5), stride=1, padding=0)        

        
    def forward(self,source,feat):
        """
            inputs :
                source : input feature maps( B X C X W X H) 256,64,64   512,16,16
                driving : input feature maps( B X C X W X H) 256,64,64  512,16,16
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        feat=self.conv(feat)
        m_batchsize,C,width ,height = source.size()
        
        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.matmul(proj_query,proj_key) # transpose check #bmm
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.matmul(proj_value,attention.permute(0,2,1) ) ##bmm
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + feat
        out=self.deconv4(self.deconv3(self.deconv2(self.deconv1(out))))
        
        
        
        return out,attention     
        
    
    
    
    
    
    
    

class FusionGenerator(nn.Module):
    """
    Generator follows NVIDIA architecture.
    """

        
    def __init__(self, num_channels, num_kp, scale_factor, num_ref, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks, codec_mode):
        
        super(FusionGenerator, self).__init__()


        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        
        self.flowprediction=SPADE_Flow()
        self.GeneHourglass2d=GeneHourglass2d()
        self.scale_factor = scale_factor 
        self.block_expansion=block_expansion
        
        if self.scale_factor==1/4:
            
            down_blocks = []
            for i in range(2):            
                in_features = min(max_features, block_expansion * (2 ** i))
                out_features = min(max_features, block_expansion * (2 ** (i + 1)))
                down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.down_blocks = nn.ModuleList(down_blocks)

            self.src_first = SameBlock2d(1, block_expansion*4, kernel_size=(7, 7), padding=(3, 3))

            self.bottleneck = torch.nn.Sequential()
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks-1)))
            for i in range(num_bottleneck_blocks): 
                self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

            self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

            up_blocks = []
            num_up_blocks=2
            for i in range(num_up_blocks):
                in_features = min(max_features, block_expansion * (2 ** (num_up_blocks - i)))
                out_features = min(max_features, block_expansion * (2 ** (num_up_blocks - i - 1)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.up_blocks = nn.ModuleList(up_blocks)     

            self.AttnModule = DepthAwareAttention_scale64(out_features,nn.ReLU())
            
            
        if self.scale_factor==1/8:
            
            down_blocks = []
            for i in range(3):            
                in_features = min(max_features, block_expansion * (2 ** i))
                out_features = min(max_features, block_expansion * (2 ** (i + 1)))
                down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.down_blocks = nn.ModuleList(down_blocks)

            #source depth
            self.src_first = SameBlock2d(1, block_expansion*8, kernel_size=(7, 7), padding=(3, 3))

            self.bottleneck = torch.nn.Sequential()
            in_features = min(max_features, block_expansion * (4 ** num_down_blocks))
            for i in range(num_bottleneck_blocks): 
                self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

            self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

            up_blocks = []
            num_up_blocks=3
            for i in range(num_up_blocks):
                in_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i)))
                out_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i - 1)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.up_blocks = nn.ModuleList(up_blocks)     

            self.AttnModule = DepthAwareAttention_scale32(out_features,nn.ReLU())
            

        if self.scale_factor==1/16:
            
            down_blocks = []
            for i in range(4):            
                in_features = min(max_features, block_expansion * (2 ** i))
                out_features = min(max_features, block_expansion * (2 ** (i + 1)))
                down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.down_blocks = nn.ModuleList(down_blocks)

            #source depth
            self.src_first = SameBlock2d(1, block_expansion*8, kernel_size=(7, 7), padding=(3, 3))

            self.bottleneck = torch.nn.Sequential()
            in_features = min(max_features, block_expansion * (4 ** num_down_blocks))
            for i in range(num_bottleneck_blocks): 
                self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

            self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

            up_blocks = []
            num_up_blocks=4
            for i in range(num_up_blocks):
                in_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i)))
                out_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i - 1)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.up_blocks = nn.ModuleList(up_blocks)                    

            self.AttnModule = DepthAwareAttention_scale16(out_features,nn.ReLU())

        if self.scale_factor==1/32:
            
            down_blocks = []
            for i in range(5):            
                in_features = min(max_features, block_expansion * (2 ** i))
                out_features = min(max_features, block_expansion * (2 ** (i + 1)))
                down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.down_blocks = nn.ModuleList(down_blocks)

            #source depth
            self.src_first = SameBlock2d(1, block_expansion*8, kernel_size=(7, 7), padding=(3, 3))

            self.bottleneck = torch.nn.Sequential()
            in_features = min(max_features, block_expansion * (4 ** num_down_blocks))
            for i in range(num_bottleneck_blocks): 
                self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

            self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

            up_blocks = []
            num_up_blocks=5
            for i in range(num_up_blocks):
                in_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i)))
                out_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i - 1)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.up_blocks = nn.ModuleList(up_blocks)                    

            self.AttnModule = DepthAwareAttention_scale8(out_features,nn.ReLU())    
    
    
        if self.scale_factor==1/64:
            
            down_blocks = []
            for i in range(6):            
                in_features = min(max_features, block_expansion * (2 ** i))
                out_features = min(max_features, block_expansion * (2 ** (i + 1)))
                down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.down_blocks = nn.ModuleList(down_blocks)

            #source depth
            self.src_first = SameBlock2d(1, block_expansion*8, kernel_size=(7, 7), padding=(3, 3))

            self.bottleneck = torch.nn.Sequential()
            in_features = min(max_features, block_expansion * (4 ** num_down_blocks))
            for i in range(num_bottleneck_blocks): 
                self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

            self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

            up_blocks = []
            num_up_blocks=6
            for i in range(num_up_blocks):
                in_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i)))
                out_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i - 1)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.up_blocks = nn.ModuleList(up_blocks)                    

            self.AttnModule = DepthAwareAttention_scale4(out_features,nn.ReLU())    
            
            
            
        if self.scale_factor ==3/32:
            
            down_blocks = []
            for i in range(3):            
                in_features = min(max_features, block_expansion * (2 ** i))
                out_features = min(max_features, block_expansion * (2 ** (i + 1)))
                down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.down_blocks = nn.ModuleList(down_blocks)

            #source depth
            self.src_first = SameBlock2d(1, block_expansion*8, kernel_size=(7, 7), padding=(3, 3))

            self.bottleneck = torch.nn.Sequential()
            in_features = min(max_features, block_expansion * (4 ** num_down_blocks))
            for i in range(num_bottleneck_blocks): 
                self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

            self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

            up_blocks = []
            num_up_blocks=3
            for i in range(num_up_blocks):
                in_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i)))
                out_features = min(max_features, block_expansion * (4 ** (num_up_blocks - i - 1)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.up_blocks = nn.ModuleList(up_blocks)     

            self.AttnModule = DepthAwareAttention_scale24(out_features,nn.ReLU())    
            
            
            
        if self.scale_factor==3/16:
            
            down_blocks = []
            for i in range(2):            
                in_features = min(max_features, block_expansion * (2 ** i))
                out_features = min(max_features, block_expansion * (2 ** (i + 1)))
                down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.down_blocks = nn.ModuleList(down_blocks)

            self.src_first = SameBlock2d(1, block_expansion*4, kernel_size=(7, 7), padding=(3, 3))

            self.bottleneck = torch.nn.Sequential()
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks-1)))
            for i in range(num_bottleneck_blocks): 
                self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

            self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

            up_blocks = []
            num_up_blocks=2
            for i in range(num_up_blocks):
                in_features = min(max_features, block_expansion * (2 ** (num_up_blocks - i)))
                out_features = min(max_features, block_expansion * (2 ** (num_up_blocks - i - 1)))
                up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            self.up_blocks = nn.ModuleList(up_blocks)     

            self.AttnModule = DepthAwareAttention_scale48(out_features,nn.ReLU())            
            
            
            
            
            
            
            
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)            

    
    def forward(self, source, generated, faceresidual):

        ############# Attention for modifying the generation
        # Encoding (downsampling) part
        out_generated = self.first(generated)
        for i in range(len(self.down_blocks)):
            out_generated = self.down_blocks[i](out_generated)  
        # print(out.shape)
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        residual_out = self.src_first(faceresidual)  
        # print(residual_out.shape)
        # print(out_generated.shape)

        out_transformed,attention  = self.AttnModule(residual_out,out_generated) # out: #torch.Size([bs, 512, widith*scale, height*scale])
        
        output_dict["attention"] = attention  
        output_dict["out_transformed"] = out_transformed          
        
        out_attentioned = self.bottleneck(out_transformed) 
        
        for i in range(len(self.up_blocks)):
            out_attentioned = self.up_blocks[i](out_attentioned)
        
        out_attentioned = self.final(out_attentioned)
        out_attentioned = torch.sigmoid(out_attentioned)
        output_dict["attentioned_prediction"] = out_attentioned   
        
        ################## Predict the accurate flow for better fidelity/texture        
        occlusion, denseflow =self.flowprediction(source, out_attentioned)
        output_dict['denseflow'] = denseflow #torch.Size([bs, 256, 256, 2])
        output_dict['occlusion'] = occlusion #torch.Size([bs, 1, 256, 256])                               

        # # Encoding (downsampling) part
        out=self.GeneHourglass2d(self.first(source), denseflow, occlusion)
        output_dict["fused"] = out      
        # print("Fusion: ",out.shape)        
        
        return output_dict    