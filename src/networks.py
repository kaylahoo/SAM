import torch
import torch.nn as nn
from src.partialconv2d import PartialConv2d  # 加
from src.partialconv2d import PConvBNActiv  # 加
from src.depconv2d import DepConvBNActiv
import torch.nn.functional as F  # 加
from src.stageone import InpaintGenerator1
import torch.nn.functional as F  # 加
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.codecs.base_codec import BaseCodec
from image_synthesis.modeling.modules.vqgan_loss.vqperceptual import VQLPIPSWithDiscriminator
from image_synthesis.modeling.utils.misc import distributed_sinkhorn, get_token_type, get_gaussian_weight, \
    st_gumbel_softmax
from image_synthesis.distributed.distributed import all_reduce, get_world_size
from image_synthesis.modeling.modules.edge_connect.losses import EdgeConnectLoss
from image_synthesis.modeling.utils.position_encoding import build_position_encoding


# from timm.models.layers import DropPath


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)  # 初始化正态分布N ( 0 , std=1 )
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)  # 初始化为 正态分布~ N ( 0 , std )

                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # kaiming针对relu函数提出的初始化方法
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)  # 初始化为常数

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    # class Generator(nn.Module):

    #     def __init__(self, image_in_channels=4, edge_in_channels=4, out_channels=4, init_weights=True):
    #         super(InpaintGenerator, self).__init__()
    #
    #         self.freeze_ec_bn = False
    #         # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    #
    #         # -----------------------
    #         # small encoder-decoder
    #         # -----------------------
    #         self.ec_texture_1 = PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7')
    #         self.ec_texture_2 = PConvBNActiv(64, 128, sample='down-5', )
    #         self.ec_texture_3 = PConvBNActiv(128, 256, sample='down-5')
    #         self.ec_texture_4 = PConvBNActiv(256, 512, sample='down-3')
    #         self.ec_texture_5 = PConvBNActiv(512, 512, sample='down-3')
    #         self.ec_texture_6 = PConvBNActiv(512, 512, sample='down-3')
    #         self.ec_texture_7 = PConvBNActiv(512, 512, sample='down-3')
    #
    #         self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
    #         self.dc_texture_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
    #         self.dc_texture_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
    #         self.dc_texture_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
    #         self.dc_texture_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
    #         self.dc_texture_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
    #         self.dc_texture_1 = PConvBNActiv(64 + out_channels, 64, activ='leaky')
    #
    #         # -------------------------
    #         # large encoder-decoder
    #         # -------------------------
    #         self.ec_structure_1 = DepConvBNActiv(edge_in_channels, 64, bn=False, sample='down-31', groups=edge_in_channels)
    #         self.ec_structure_2 = DepConvBNActiv(64, 128, sample='down-29', groups=64)
    #         self.ec_structure_3 = DepConvBNActiv(128, 256, sample='down-27', groups=128)
    #         self.ec_structure_4 = DepConvBNActiv(256, 512, sample='down-13', groups=256)
    #         self.ec_structure_5 = DepConvBNActiv(512, 512, sample='down-13', groups=512)
    #         self.ec_structure_6 = DepConvBNActiv(512, 512, sample='down-13', groups=512)
    #         self.ec_structure_7 = DepConvBNActiv(512, 512, sample='down-13', groups=512)
    #
    #         self.dc_structure_7 = DepConvBNActiv(512 + 512, 512, groups=512,activ='leaky')
    #         self.dc_structure_6 = DepConvBNActiv(512 + 512, 512, groups=512,activ='leaky')
    #         self.dc_structure_5 = DepConvBNActiv(512 + 512, 512, groups=512,activ='leaky')
    #         self.dc_structure_4 = DepConvBNActiv(512 + 256, 256, groups=256,activ='leaky')
    #         self.dc_structure_3 = DepConvBNActiv(256 + 128, 128, groups=128,activ='leaky')
    #         self.dc_structure_2 = DepConvBNActiv(128 + 64, 64, groups=64,activ='leaky')
    #         self.dc_structure_1 = DepConvBNActiv(64+out_channels, 64, groups=4,activ='leaky')
    #
    #         self.fusion_layer1 = nn.Sequential(
    #             nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
    #             nn.LeakyReLU(negative_slope=0.2)
    #         )
    #         self.fusion_layer2 = nn.Sequential(
    #             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    #             nn.LeakyReLU(negative_slope=0.2),
    #         )
    #         self.out_layer = nn.Sequential(
    #             nn.Conv2d(64, 3, kernel_size=1),
    #             nn.Tanh()
    #         )
    #
    #         if init_weights:
    #             self.init_weights()
    #
    #     def forward(self, images_masks, masks):
    # #texture
    #         ec_textures = {}
    #         ec_structures = {}
    #         input_image = images_masks
    #
    #         input_texture_mask = torch.cat((masks, masks, masks,masks), dim=1)
    #         #print('116',input_texture_mask.shape)#[8,4,256,256]
    #
    #         ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'] = input_image, input_texture_mask
    #         # print('t00', ec_textures['ec_t_0'].shape)  # [2,4,256,256]
    #         # print('t00', ec_textures['ec_t_masks_0'].shape)  # [2,4,256,256]
    #         ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'] = self.ec_texture_1(ec_textures['ec_t_0'],ec_textures['ec_t_masks_0'])
    #         #print('t11', ec_textures['ec_t_1'].shape)#[2,64,128,128]
    #         #print('t11', ec_textures['ec_t_masks_1'].shape)#[2,64,128,128]
    #         ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'] = self.ec_texture_2(ec_textures['ec_t_1'],ec_textures['ec_t_masks_1'])
    #         #print('t22', ec_textures['ec_t_2'].shape)#[2,128,64,64]
    #         #print('t22', ec_textures['ec_t_masks_2'].shape)#[2,128,64,64]
    #         ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'] = self.ec_texture_3(ec_textures['ec_t_2'],ec_textures['ec_t_masks_2'])
    #         #print('t33', ec_textures['ec_t_3'].shape)#[2,256,32,32]
    #         #print('t33', ec_textures['ec_t_masks_3'].shape)#[2,256,32,32]
    #         ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'] = self.ec_texture_4(ec_textures['ec_t_3'],ec_textures['ec_t_masks_3'])
    #         #print('t44',ec_textures['ec_t_4'].shape)#[2,512,16,16]
    #         #print('t44', ec_textures['ec_t_masks_4'].shape)#[2,512,16,16]
    #         ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'] = self.ec_texture_5(ec_textures['ec_t_4'],ec_textures['ec_t_masks_4'])
    #         #print('t55', ec_textures['ec_t_5'].shape)#[2,512,8,8]
    #         #print('t55', ec_textures['ec_t_masks_5'].shape)#[2,512,8,8]
    #         ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'] = self.ec_texture_6(ec_textures['ec_t_5'],ec_textures['ec_t_masks_5'])
    #         # print('t66', ec_textures['ec_t_6'].shape)#[2,512,4,4]
    #         # print('t66', ec_textures['ec_t_masks_6'].shape)#[2,512,4,4]
    #         ec_textures['ec_t_7'], ec_textures['ec_t_masks_7'] = self.ec_texture_7(ec_textures['ec_t_6'],ec_textures['ec_t_masks_6'])
    #         # print('t7', ec_textures['ec_t_7'].shape)#[2,512,2,2]
    #         # print('t7m', ec_textures['ec_t_masks_7'].shape)#[2,512,2,2]
    #
    # #structure
    #         input_structure_mask = torch.cat((masks, masks, masks,masks), dim=1)
    #         ec_structures['ec_s_0'], ec_structures['ec_s_masks_0'] = input_image, input_structure_mask
    #         # print('s000', ec_structures['ec_s_0'].shape)#[2,4,256,256]
    #         # print('s000', ec_structures['ec_s_masks_0'].shape)#[2,4,256,256]
    #         ec_structures['ec_s_1'], ec_structures['ec_s_masks_1'] = self.ec_structure_1(ec_structures['ec_s_0'],ec_structures['ec_s_masks_0'])
    #         # print('s111', ec_structures['ec_s_1'].shape)#[2,64,128,128]
    #         # print('s111', ec_structures['ec_s_masks_1'].shape)#[2,64,128,128]
    #         ec_structures['ec_s_2'], ec_structures['ec_s_masks_2'] = self.ec_structure_2(ec_structures['ec_s_1'], ec_structures['ec_s_masks_1'])
    #         # print('s222', ec_structures['ec_s_2'].shape)#[2,128,64,64]
    #         # print('s222', ec_structures['ec_s_masks_2'].shape)#[2,128,64,64]
    #
    #         ec_structures['ec_s_3'], ec_structures['ec_s_masks_3'] = self.ec_structure_3(ec_structures['ec_s_2'],ec_structures['ec_s_masks_2'])
    #         # print('s33', ec_structures['ec_s_3'].shape)#[2,256,32,32]
    #         # print('s333', ec_structures['ec_s_masks_3'].shape)#[2,256,32,32]
    #
    #         ec_structures['ec_s_4'], ec_structures['ec_s_masks_4'] = self.ec_structure_4(ec_structures['ec_s_3'],ec_structures['ec_s_masks_3'])
    #         # print('4s44', ec_structures['ec_s_4'].shape)#[2,512,16,16]
    #         # print('s444', ec_structures['ec_s_masks_4'].shape)#[2,512,16,16]
    #         ec_structures['ec_s_5'], ec_structures['ec_s_masks_5'] = self.ec_structure_5(ec_structures['ec_s_4'],ec_structures['ec_s_masks_4'])
    #         # print('s555', ec_structures['ec_s_5'].shape)#[2,512,8,8]
    #         # print('s555', ec_structures['ec_s_masks_5'].shape)#[2,512,8,8]
    #         ec_structures['ec_s_6'], ec_structures['ec_s_masks_6'] = self.ec_structure_6(ec_structures['ec_s_5'],ec_structures['ec_s_masks_5'])
    #         # print('s666', ec_structures['ec_s_6'].shape)#[2,512,4,4]
    #         # print('s666', ec_structures['ec_s_masks_6'].shape)#[2,512,4,4]
    #         ec_structures['ec_s_7'], ec_structures['ec_s_masks_7'] = self.ec_structure_7(ec_structures['ec_s_6'],ec_structures['ec_s_masks_6'])
    #         # print('s777', ec_structures['ec_s_7'].shape)#[2,512,2,2]
    #         # print('s777', ec_structures['ec_s_masks_7'].shape)#[2,512,2,2]
    #
    #         dc_texture, dc_tecture_mask = ec_structures['ec_s_7'], ec_structures['ec_s_masks_7']  # 2x2
    #         # print('#', dc_texture.shape)#[2,512,2,2]
    #         # print('#', dc_tecture_mask.shape)#[2,512,2,2]
    #         for _ in range(7, 0, -1):
    #             ec_texture_skip = 'ec_t_{:d}'.format(_ - 1)  # ec_t_6
    #             ec_texture_masks_skip = 'ec_t_masks_{:d}'.format(_ - 1)  # ec_t_masks_6
    #             dc_conv = 'dc_texture_{:d}'.format(_)  # dc_texture_7
    #
    #             dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')  # dc_texture 4x4
    #             dc_tecture_mask = F.interpolate(dc_tecture_mask, scale_factor=2, mode='nearest')  # dc_tecture_mask 4x4
    #             #print('!!!',dc_texture.shape)
    #
    #             #print(ec_textures[ec_texture_skip].shape)
    #             dc_texture = torch.cat((dc_texture, ec_textures[ec_texture_skip]), dim=1)  # dc_texture 4x4 ec_textures['ec_t_6'] 4x4
    #            # print('186',dc_texture.shape)
    #             dc_tecture_mask = torch.cat((dc_tecture_mask, ec_textures[ec_texture_masks_skip]),
    #                                         dim=1)  # dc_tecture_mask 4x4 ec_textures['ec_t_masks_6'] 4x4
    #
    #             dc_texture, dc_tecture_mask = getattr(self, dc_conv)(dc_texture,dc_tecture_mask)
    #             #print('191',dc_texture.shape,dc_tecture_mask.shape)
    #             # self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ='leaky')  self.dc_texture_7(dc_texture, dc_tecture_mask)
    #
    #         dc_structure, dc_structure_masks = ec_textures['ec_t_7'], ec_textures['ec_t_masks_7']
    #         #print('195',dc_structure.shape)
    #         for _ in range(7, 0, -1):
    #             ec_structure_skip = 'ec_s_{:d}'.format(_ - 1)
    #             ec_structure_masks_skip = 'ec_s_masks_{:d}'.format(_ - 1)
    #             dc_conv = 'dc_structure_{:d}'.format(_)
    #             #print('200', ec_structure_skip,ec_structure_masks_skip,dc_conv)
    #             dc_structure = F.interpolate(dc_structure, scale_factor=2, mode='bilinear')
    #             #print('202', dc_structure.shape)
    #             dc_structure_masks = F.interpolate(dc_structure_masks, scale_factor=2, mode='nearest')
    #             #print('204', dc_structure_masks.shape)
    #             dc_structure = torch.cat((dc_structure, ec_structures[ec_structure_skip]), dim=1)
    #             dc_structure_masks = torch.cat((dc_structure_masks, ec_structures[ec_structure_masks_skip]), dim=1)
    #             #print('207', dc_structure.shape,dc_structure_masks.shape)
    #             dc_structure, dc_structure_masks = getattr(self, dc_conv)(dc_structure, dc_structure_masks)
    #         #print('210')
    #         output1 = torch.cat((dc_texture, dc_structure),dim=1)
    #         output2 = self.fusion_layer1(output1)
    #         output3 = self.fusion_layer2(output2)
    #         output4 = self.out_layer(output3)
    #
    #         return output4

    # def __init__(self, residual_blocks=8, init_weights=True):
    #     super(InpaintGenerator, self).__init__() #在子类中调用父类方法
    #
    #     self.encoder = nn.Sequential(
    #         nn.ReflectionPad2d(3),#镜像填充
    #         nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
    #         nn.InstanceNorm2d(64, track_running_stats=False), #一个channel内做归一化，算H*W的均值
    #         nn.ReLU(True),
    #
    #         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(128, track_running_stats=False),
    #         nn.ReLU(True),
    #
    #         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(256, track_running_stats=False),
    #         nn.ReLU(True)
    #     )
    #
    #     blocks = []
    #     for _ in range(residual_blocks):
    #         block = ResnetBlock(256, 2)
    #         blocks.append(block)
    #
    #     self.middle = nn.Sequential(*blocks)
    #
    #     self.decoder = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(128, track_running_stats=False),
    #         nn.ReLU(True),
    #
    #         nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(64, track_running_stats=False),
    #         nn.ReLU(True),
    #
    #         nn.ReflectionPad2d(3),
    #         nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
    #     )
    #
    #     if init_weights:
    #         self.init_weights()
    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = self.middle(x)
    #     x = self.decoder(x)
    #     x = (torch.tanh(x) + 1) / 2
    #
    #     return x

    # csy
    # def __init__(self, residual_blocks=8, init_weights=True):
    #     super(InpaintGenerator, self).__init__()
    #
    #     n = 32
    #     # rough
    #     self.enc_r1 = nn.Sequential(
    #         nn.ReflectionPad2d(3),
    #         nn.Conv2d(in_channels=7, out_channels=n, kernel_size=7, stride=1, padding=0),
    #         nn.InstanceNorm2d(n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 256
    #
    #     self.Pconv_r2 = PartialConv2d(in_channels=n, out_channels=2*n, kernel_size=7, stride=2, padding=3,
    #                                   return_mask=True)
    #     self.enc_r2 = nn.Sequential(
    #         nn.InstanceNorm2d(2*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 128
    #
    #     self.Pconv_r3 = PartialConv2d(in_channels=2*n, out_channels=4*n, kernel_size=3, stride=2, padding=1,
    #                                   return_mask=True)
    #     self.enc_r3 = nn.Sequential(
    #         nn.InstanceNorm2d(4*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 64
    #
    #     self.enc_r4 = nn.Sequential(
    #         nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=3, stride=2, padding=1),
    #         nn.InstanceNorm2d(8*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 32
    #
    #
    #     # fine
    #     self.enc_f1 = nn.Sequential(
    #         nn.ReflectionPad2d(2),
    #         nn.Conv2d(in_channels=7, out_channels=n, kernel_size=5, stride=1, padding=0),
    #         nn.InstanceNorm2d(n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 256
    #
    #     self.Pconv_f2 = PartialConv2d(in_channels=n, out_channels=2*n, kernel_size=5, stride=2, padding=2, return_mask=True)
    #     self.enc_f2 = nn.Sequential(
    #         nn.InstanceNorm2d(2*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 128
    #
    #     self.Pconv_f3 = PartialConv2d(in_channels=2*n, out_channels=4*n, kernel_size=3, stride=2, padding=1, return_mask=True)
    #     self.enc_f3 = nn.Sequential(
    #         nn.InstanceNorm2d(4*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 64
    #
    #     self.enc_f4 = nn.Sequential(
    #         nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=3, stride=2, padding=1),
    #         nn.InstanceNorm2d(8*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 32
    #
    #     # bottleneck
    #     blocks = []
    #     for i in range(residual_blocks-1):
    #         block = ResnetBlock(dim=8*n, dilation=2, use_attention_norm=False)
    #         blocks.append(block)
    #     self.middle = nn.Sequential(*blocks)
    #     self.chan_att_norm = ResnetBlock(dim=8*n, dilation=2, use_attention_norm=True)
    #
    #     # decoder
    #     self.dec_1 = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels=8*n, out_channels=4*n, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(4*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 64
    #
    #     self.dec_2 = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels=4*n, out_channels=2*n, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(2*n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 128
    #
    #     self.dec_3 = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels=2*n+3, out_channels=n, kernel_size=4, stride=2, padding=1),
    #         nn.InstanceNorm2d(n, track_running_stats=False),
    #         nn.ReLU(True)
    #     )  # 256
    #
    #     self.dec_4 = nn.Sequential(
    #         nn.Conv2d(in_channels=n, out_channels=3, kernel_size=1, padding=0)
    #     )  # 256
    #
    #
    #
    #     if init_weights:
    #         self.init_weights()
    #
    # def forward(self, x, mask, stage):
    #
    #     if stage is 0:
    #         structure = x[:, -3:, ...]
    #         structure = F.interpolate(structure, scale_factor=0.5, mode='bilinear')
    #
    #         x = self.enc_r1(x)
    #         x, m = self.Pconv_r2(x, mask)
    #         x = self.enc_r2(x)
    #         f1 = x.detach()
    #         x, _ = self.Pconv_r3(x, m)
    #         x = self.enc_r3(x)
    #         x = self.enc_r4(x)
    #         x = self.middle(x)
    #         x, l_c1 = self.chan_att_norm(x)
    #         x = self.dec_1(x)
    #         x = self.dec_2(x)
    #         x, l_p1 = self.pos_attention_norm1(x, f1, m)
    #         att_structure = self.pos_attention_norm1(structure, structure, m, reuse=True)
    #         att_structure = att_structure.detach()
    #         x = torch.cat((x, att_structure), dim=1)
    #         x = self.dec_3(x)
    #         x = self.dec_4(x)
    #         x_rough = (torch.tanh(x) + 1) / 2
    #         orth_loss = (l_c1 + l_p1) / 2
    #         return x_rough, orth_loss
    #
    #     if stage is 1:
    #         residual = x[:, -3:, ...]
    #         residual = F.interpolate(residual, scale_factor=0.5, mode='bilinear')
    #
    #         x = self.enc_f1(x)
    #         x, m = self.Pconv_f2(x, mask)
    #         x = self.enc_f2(x)
    #         f2 = x.detach()
    #         x, _ = self.Pconv_f3(x, m)
    #         x = self.enc_f3(x)
    #         x = self.enc_f4(x)
    #         x = self.middle(x)
    #         x, l_c2 = self.chan_att_norm(x)
    #         x = self.dec_1(x)
    #         x = self.dec_2(x)
    #         x, l_p2 = self.pos_attention_norm2(x, f2, m)
    #         att_residual = self.pos_attention_norm2(residual, residual, m, reuse=True)
    #         att_residual = att_residual.detach()
    #         x = torch.cat((x, att_residual), dim=1)
    #         x = self.dec_3(x)
    #         x = self.dec_4(x)
    #         x_fine = (torch.tanh(x) + 1) / 2
    #         orth_loss = (l_c2 + l_p2) / 2
    #         return x_fine, orth_loss
    def __init__(self, image_in_channels=4, out_channels=4, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.freeze_ec_bn = False
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # -----------------------
        # small encoder-decoder
        # -----------------------
        self.ec_texture_1 = PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7')
        self.ec_texture_2 = PConvBNActiv(64, 128, sample='down-5', )
        self.ec_texture_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_texture_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_texture_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_texture_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_texture_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_texture_1 = PConvBNActiv(64 + out_channels, 64, activ='leaky')

        self.fusion_layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fusion_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()
        )

        if init_weights:
            self.init_weights()

        path = "/home/lab265/8T/liulu/SAA/checkpoints/ffhq/InpaintingModel1_gen.pth"
        # "/home/lab265/lab265/lab508_8T/liulu/SAA/checkpoints/ffhq/InpaintingModel_gen.pth"
        self.content_codec = InpaintGenerator1(ckpt_path=path, trainable=False)
        self.kv = self.content_codec.quantize.get_codebook()['default']['code']
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.up_dim = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down_dim = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, images_masked, masks):
        # x = images_masked
        # x = self.encoder(x)
        # x = self.middle1(x)
        # b, c, h, w = x.shape
        # tgt = x.reshape(b, c, h * w).permute(2, 0, 1).contiguous()
        # # # #print(tgt.shape) [1024,12,512]
        # mem = self.kv.unsqueeze(dim=1).repeat(1, x.shape[0], 1).to(tgt.device)
        # attn_out, _ = self.attn(tgt, mem, tgt
        #                             )
        # attn_out = attn_out.permute(1, 2, 0).reshape(x.shape)
        # x = x + attn_out
        # x = self.middle2(x)
        # x = self.decoder(x)
        # x = (torch.tanh(x) + 1) / 2
        # #x = torch.sigmoid(x) #edge
        # return x #, emb_loss

        #     self.encoder = nn.Sequential(
        #         nn.ReflectionPad2d(3),
        #         nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
        #         nn.InstanceNorm2d(64, track_running_stats=False),
        #         nn.ReLU(True),
        #
        #         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
        #         nn.InstanceNorm2d(128, track_running_stats=False),
        #         nn.ReLU(True),
        #
        #         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
        #         nn.InstanceNorm2d(256, track_running_stats=False),
        #         nn.ReLU(True)
        #     )
        #
        #     blocks = []
        #     for _ in range(residual_blocks):
        #         block = ResnetBlock(256, 2)
        #         blocks.append(block)
        #
        #     self.middle = nn.Sequential(*blocks)
        #
        #     self.decoder = nn.Sequential(
        #         nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
        #         nn.InstanceNorm2d(128, track_running_stats=False),
        #         nn.ReLU(True),
        #
        #         nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
        #         nn.InstanceNorm2d(64, track_running_stats=False),
        #         nn.ReLU(True),
        #
        #         nn.ReflectionPad2d(3),
        #         nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        #     )
        #
        #     if init_weights:
        #         self.init_weights()
        #
        # def forward(self, x):
        #     x = self.encoder(x)
        #     x = self.middle(x)
        #     x = self.decoder(x)
        #     x = (torch.tanh(x) + 1) / 2
        #
        #     return x

        ec_textures = {}
        ec_structures = {}
        input_image = images_masked

        input_texture_mask = torch.cat((masks, masks, masks, masks), dim=1)
        # print('116',input_texture_mask.shape)#[8,4,256,256]

        ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'] = input_image, input_texture_mask
        # print('t00', ec_textures['ec_t_0'].shape)  # [2,4,256,256]
        # print('t00', ec_textures['ec_t_masks_0'].shape)  # [2,4,256,256]
        ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'] = self.ec_texture_1(ec_textures['ec_t_0'],
                                                                               ec_textures['ec_t_masks_0'])
        # print('t11', ec_textures['ec_t_1'].shape)#[2,64,128,128]
        # print('t11', ec_textures['ec_t_masks_1'].shape)#[2,64,128,128]
        ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'] = self.ec_texture_2(ec_textures['ec_t_1'],
                                                                               ec_textures['ec_t_masks_1'])
        # print('t22', ec_textures['ec_t_2'].shape)#[2,128,64,64]
        # print('t22', ec_textures['ec_t_masks_2'].shape)#[2,128,64,64]
        ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'] = self.ec_texture_3(ec_textures['ec_t_2'],
                                                                               ec_textures['ec_t_masks_2'])
        # print('t33', ec_textures['ec_t_3'].shape)#[2,256,32,32]
        # print('t33', ec_textures['ec_t_masks_3'].shape)#[2,256,32,32]
        ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'] = self.ec_texture_4(ec_textures['ec_t_3'],
                                                                               ec_textures['ec_t_masks_3'])
        # print('t44',ec_textures['ec_t_4'].shape)#[2,512,16,16]
        # print('t44', ec_textures['ec_t_masks_4'].shape)#[2,512,16,16]
        ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'] = self.ec_texture_5(ec_textures['ec_t_4'],
                                                                               ec_textures['ec_t_masks_4'])
        # print('t55', ec_textures['ec_t_5'].shape)#[2,512,8,8]
        # print('t55', ec_textures['ec_t_masks_5'].shape)#[2,512,8,8]
        ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'] = self.ec_texture_6(ec_textures['ec_t_5'],
                                                                               ec_textures['ec_t_masks_5'])
        # print('t66', ec_textures['ec_t_6'].shape)#[2,512,4,4]
        # print('t66', ec_textures['ec_t_masks_6'].shape)#[2,512,4,4]
        ec_textures['ec_t_7'], ec_textures['ec_t_masks_7'] = self.ec_texture_7(ec_textures['ec_t_6'],
                                                                               ec_textures['ec_t_masks_6'])
        # print('t7', ec_textures['ec_t_7'].shape)#[2,512,2,2]
        # print('t7m', ec_textures['ec_t_masks_7'].shape)#[2,512,2,2]
        dc_texture, dc_tecture_mask = ec_textures['ec_t_7'], ec_textures['ec_t_masks_7']

        for _ in range(7, 0, -1):

            ec_texture_skip = 'ec_t_{:d}'.format(_ - 1)  # ec_t_6
            ec_texture_masks_skip = 'ec_t_masks_{:d}'.format(_ - 1)  # ec_t_masks_6
            dc_conv = 'dc_texture_{:d}'.format(_)  # dc_texture_7

            if _ == 3:
                dc_texture_512 = self.up_dim(dc_texture)
                b, c, h, w = dc_texture_512.shape  # [12, 512, 32, 32]
                tgt = dc_texture_512.reshape(b, c, h * w).permute(2, 0, 1).contiguous()
                # print(tgt.shape) [1024,12,512]
                mem = self.kv.unsqueeze(dim=1).repeat(1, dc_texture.shape[0], 1).to(tgt.device)
                # print(mem.shape) [1024,12,512]
                attn_out, _ = self.attn(tgt, mem, mem)
                # print(attn_out.shape) [1024,12,512] --> permute(1, 2, 0) --> [12,512,1024]
                attn_out = attn_out.permute(1, 2, 0).reshape(dc_texture_512.shape)
                attn_out_256 = self.down_dim(attn_out)
                dc_texture = dc_texture + attn_out_256

            dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')  # dc_texture 4x4
            dc_tecture_mask = F.interpolate(dc_tecture_mask, scale_factor=2, mode='nearest')  # dc_tecture_mask 4x4

            # print(ec_textures[ec_texture_skip].shape)
            dc_texture = torch.cat((dc_texture, ec_textures[ec_texture_skip]),
                                   dim=1)  # dc_texture 4x4 ec_textures['ec_t_6'] 4x4
            # print('186',dc_texture.shape)
            dc_tecture_mask = torch.cat((dc_tecture_mask, ec_textures[ec_texture_masks_skip]),
                                        dim=1)  # dc_tecture_mask 4x4 ec_textures['ec_t_masks_6'] 4x4

            dc_texture, dc_tecture_mask = getattr(self, dc_conv)(dc_texture, dc_tecture_mask)

        output1 = dc_texture
        output2 = self.fusion_layer1(output1)
        output3 = self.fusion_layer2(output2)
        output4 = self.out_layer(output3)

        return output4


def value_scheduler(init_value, dest_value, step, step_range, total_steps, scheduler_type='cosine'):
    assert scheduler_type in ['cosine', 'step'], 'scheduler {} not implemented!'.format(scheduler_type)

    step_start, step_end = tuple(step_range)
    if step_end <= 0:
        step_end = total_steps

    if step < step_start:
        return init_value
    if step > step_end:
        return dest_value
    factor = float(step - step_start) / float(max(1, step_end - step_start))
    if scheduler_type == 'cosine':
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * factor)))
    elif scheduler_type == 'step':
        factor = 1 - factor
    else:
        raise NotImplementedError('scheduler type {} not implemented!'.format(scheduler_type))
    if init_value >= dest_value:  # decrease
        value = dest_value + (init_value - dest_value) * factor
    else:  # increase
        factor = 1 - factor
        value = init_value + (dest_value - init_value) * factor
    return value


def gumbel_softmax(logits, temperature=1.0, gumbel_scale=1.0, dim=-1, hard=True):
    # gumbels = torch.distributions.gumbel.Gumbel(0,1).sample(logits.shape).to(logits)
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    # adjust the scale of gumbel noise
    gumbels = gumbels * gumbel_scale

    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


# class for quantization
def calculate_kl_loss(q, p, lamb=0.1):
    if q.dim() > 1 and q.shape[0] > 1:
        q = q.view(-1, q.shape[-1])
        q = q.mean(dim=0).to(p)
    loss = torch.sum(p * torch.log((p / q)))
    return loss * lamb


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self,
                 n_e,  # 嵌入向量的数量
                 e_dim,  # 嵌入向量的维度
                 beta=0.25,  # 损失项中使用的代价因子
                 n_cluster=0,
                 masked_embed_start=None,
                 embed_init_scale=1.0,  # 嵌入向量的初试大小
                 embed_ema=False,  # 是否使用指数加权平均EMA嵌入向量？
                 get_embed_type='matmul',  # 获取嵌入向量的方式，有'matmul'和'dist'两种
                 distance_type='euclidean',  # 离度量的类型。默认为欧氏距离

                 gumbel_sample=False,  # 是否使用Gumbel-Softmax技巧
                 adjust_logits_for_gumbel='sqrt',  # 用于调整交叉熵损失函数中对数概率的项
                 gumbel_sample_stop_step=None,  # 循环多少次停止使用Gumbel-Softmax
                 temperature_step_range=(0, 15000),  # 温度参数的变化范围
                 temperature_scheduler_type='cosine',  # 温度调度策略，默认为余弦退火
                 temperature_init=1.0,  # 初始温度
                 temperature_dest=1 / 16.0,  # 最终温度
                 gumbel_scale_init=1.0,  # Gumbel分布的初始比例系数
                 gumbel_scale_dest=1.0,  # Gumbel分布的最终比例系数
                 gumbel_scale_step_range=(0, 1),  # 比例系数的变化范围
                 gumbel_scale_scheduler_type='cosine',  # 比例系数的调度策略，默认为余弦退火
                 ):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.n_cluster = n_cluster
        self.semantic_label = None
        self.embed_ema = embed_ema
        self.gumbel_sample = gumbel_sample
        self.adjust_logits_for_gumbel = adjust_logits_for_gumbel
        self.temperature_step_range = temperature_step_range
        self.temperature_init = temperature_init
        self.temperature_dest = temperature_dest
        self.temperature_scheduler_type = temperature_scheduler_type
        self.gumbel_scale_init = gumbel_scale_init
        self.gumbel_scale_dest = gumbel_scale_dest
        self.gumbel_scale_step_range = gumbel_scale_step_range
        self.gumbel_sample_stop_step = gumbel_sample_stop_step
        self.gumbel_scale_scheduler_type = gumbel_scale_scheduler_type
        if self.gumbel_sample_stop_step is None:
            # 如果没有指定 gumbel_sample_stop_step，那么设置其值为 temperature_step_range 的最大值
            self.gumbel_sample_stop_step = max(self.temperature_step_range[-1], self.temperature_step_range[-1])

        self.get_embed_type = get_embed_type  # 获取嵌入向量的方式，有'matmul'和'dist'两种
        self.distance_type = distance_type  # 距离度量的类型。默认为欧氏距离

        if self.embed_ema:  # 如果使用 EMA 嵌入向量？？？
            self.decay = 0.99  # EMA的衰减系数
            self.eps = 1.0e-5  # 防止分母为0的小常数
            embed = torch.randn(n_e, e_dim)  # 初始化嵌入向量
            # embed = torch.zeros(n_e, e_dim)
            # embed.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)
            self.register_buffer("embedding", embed)  # 将嵌入向量保存到模型的缓存中
            self.register_buffer("cluster_size", torch.zeros(n_e))
            self.register_buffer("embedding_avg", embed.clone())
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)  # 定义一个 nn.Embedding 层
            self.embedding.weight.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)  # 初始化嵌入向量

        self.masked_embed_start = masked_embed_start  # 定义一个变量masked_embed_start，初始值为None
        if self.masked_embed_start is None:
            # 如果 masked_embed_start 的值为 None，那么设置 masked_embed_start = n_e（嵌入向量的数量）
            self.masked_embed_start = self.n_e

        if self.distance_type == 'learned':
            # 如果使用的是学习得到的距离权重，那么定义一个nn.Linear层
            self.distance_fc = nn.Linear(self.e_dim, self.n_e)

        self.pos_enc = build_position_encoding(hidden_dim=self.e_dim)
        self.semantic_pos = nn.Sequential(*[
            nn.Linear(self.n_cluster, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.n_cluster),
        ])
        # self.semantic_classifier = nn.Sequential(*[
        #     nn.Linear(self.e_dim, self.e_dim * 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.e_dim * 2, 1024),
        #     # nn.Softmax(dim=1)
        # ])
        # self.alpha = nn.Parameter(torch.ones(size=(1, self.n_cluster)) * 0.1, requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')

    @property
    def device(self):
        # 返回嵌入向量保存的设备类型
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight.device
        return self.embedding.device

    @property
    def norm_feat(self):
        # 如果距离度量类型为'cosine'或'sinkhorn'，则返回True；否则返回False
        return self.distance_type in ['cosine', 'sinkhorn']

    @property
    def embed_weight(self):
        # 如果嵌入向量是通过 nn.Embedding 定义的，那么返回 nn.Embedding 的权重
        # 否则，返回嵌入向量本身
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight
        else:
            return self.embedding

    """
    code_book
        -- default
            -- code: self.embedding
            -- label: default_label
        --unmasked
            -- code: self.embedding[:self.n_cluster]
        --masked
            -- code: self.embedding[self.n_cluster:]
    """

    def get_codebook(self):
        codes = {
            'default': {
                'code': self.embed_weight
            }
        }
        default_label = torch.ones((self.n_e)).to(self.device)
        default_label[:self.n_cluster] = 0
        codes['default']['label'] = default_label

        return codes

    def get_semantic_label(self):
        # 返回已有的语义标签
        return self.semantic_label

    @torch.no_grad()
    def update_semantic_label(self):
        # 开始更新语义标签，使用torch.no_grad()确保不会计算梯度
        d = torch.ones(self.n_e, self.n_cluster).to(self.device)  # 定义张量d，形状为(n_e, n_cluster)，初始值为1，数据类型为float32
        if self.n_cluster > 0:
            if self.semantic_label is None:
                # 如果之前没有语义标签
                l_c = torch.arange(end=self.n_cluster)  # 前self.n_cluster个嵌入向量的语义标签为0到self.n_cluster-1
                l_f = torch.randint(0, self.n_cluster, size=(
                    self.n_e - self.n_cluster, 1)).squeeze()  # 后 n_e-n_cluster 个嵌入向量的语义标签为[0,self.n_cluster-1]之间的随机整数。
                self.semantic_label = torch.cat((l_c, l_f), dim=0).to(
                    self.device)  # 将所有语义标签拼接在一起，保存到类变量self.semantic_label中
            else:
                # with torch.no_grad:
                d = self.get_distance(
                    self.embed_weight[self.n_cluster:])  # 计算后(n_e-n_cluster)个嵌入向量与所有n_cluster个聚类中心之间的距离矩阵
                label_ = torch.argmax(0 - d[:, :self.n_cluster],
                                      dim=1)  # 取出每个后(n_e-n_cluster)个嵌入向量与聚类中心距离最近的那个聚类中心的编号，保存在label_中
                self.semantic_label[self.n_cluster:] = label_  # 将这些后(n_e-n_cluster)个嵌入向量的语义标签更新为对应的聚类中心编号
        else:
            # 如果 n_cluster = 0（即没有聚类中心），则将所有嵌入向量的语义标签设置为1
            self.semantic_label = torch.ones(self.n_e).to(self.device)
        return d  # 返回距离矩阵d

    def norm_embedding(self):
        if self.training:
            with torch.no_grad():
                w = self.embed_weight.data.clone()  # 复制self.embed_weight张量中的数据
                w = F.normalize(w, dim=1, p=2)  # 对每一行嵌入向量进行L2归一化处理
                if isinstance(self.embedding, nn.Embedding):
                    # # 如果embedding是nn.Embedding类型，则将它的权重设置为w
                    self.embedding.weight.copy_(w)
                else:
                    # # 否则，将embedding的值设置为w
                    self.embedding.copy_(w)

    def get_index(self, logits, topk=1, step=None, total_steps=None):
        """
        logits: BHW x N
        topk: the topk similar codes to be sampled from

        return:
            indices: BHW
        """

        if self.gumbel_sample:  # 判断是否使用 Gumbel Softmax 方法进行采样
            gumbel = True
            if self.training:
                # 如果step达到了设定的停止步数，就不再使用 Gumbel Softmax 方法采样
                if step > self.gumbel_sample_stop_step and self.gumbel_sample_stop_step > 0:
                    gumbel = False
            else:
                gumbel = False
        else:
            gumbel = False

        if gumbel:
            # 计算温度值和缩放因子???
            temp = value_scheduler(init_value=self.temperature_init,
                                   dest_value=self.temperature_dest,
                                   step=step,
                                   step_range=self.temperature_step_range,
                                   total_steps=total_steps,
                                   scheduler_type=self.temperature_scheduler_type
                                   )
            scale = value_scheduler(init_value=self.gumbel_scale_init,
                                    dest_value=self.gumbel_scale_dest,
                                    step=step,
                                    step_range=self.gumbel_scale_step_range,
                                    total_steps=total_steps,
                                    scheduler_type=self.gumbel_scale_scheduler_type
                                    )
            if self.adjust_logits_for_gumbel == 'none':
                pass
            elif self.adjust_logits_for_gumbel == 'sqrt':
                logits = torch.sqrt(logits)
            elif self.adjust_logits_for_gumbel == 'log':
                logits = torch.log(logits)
            else:
                raise NotImplementedError

            # for logits, the larger the value is, the corresponding code shoule not be sampled, so we need to negative it
            logits = -logits
            # one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=True) # BHW x N
            logits = gumbel_softmax(logits, temperature=temp, gumbel_scale=scale, dim=1, hard=True)
        else:
            logits = -logits

        # now, the larger value should be sampled
        if topk == 1:
            indices = torch.argmax(logits, dim=1)
        else:
            # 如果需要输出多个码本索引，则先选出概率值最大的K个码本索引，再从中随机选出一个作为所需输出
            assert not gumbel, 'For gumbel sample, topk may introduce some random choices of codes!'
            topk = min(logits.shape[1], topk)

            _, indices = torch.topk(logits, dim=1, k=topk)  # N x K
            chose = torch.randint(0, topk, (indices.shape[0],)).to(indices.device)  # N
            chose = torch.zeros_like(indices).scatter_(1, chose.unsqueeze(dim=1), 1.0)  # N x K
            indices = (indices * chose).sum(dim=1, keepdim=False)

            # filtered_logits = logits_top_k(logits, filter_ratio=topk, minimum=1, filter_type='count')
            # probs = F.softmax(filtered_logits * 1, dim=1)
            # indices = torch.multinomial(probs, 1).squeeze(dim=1) # BHW

        return indices

    def get_distance(self, z, code_type='all'):
        """
        z: L x D, the provided features    # 计算输入特征向量与码本向量之间的距离

        return:
            d: L x N, where N is the number of tokens, the smaller distance is, the more similar it is
        """
        if self.distance_type == 'euclidean':  # 当距离类型为欧几里得距离时
            d = torch.sum(z ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embed_weight ** 2, dim=1) - 2 * \
                torch.matmul(z, self.embed_weight.t())  # 计算欧氏距离
        elif self.distance_type == 'learned':  # 当距离类型为学习距离时
            d = 0 - self.distance_fc(z)  # BHW x N   # 调用预先定义好的学习距离函数进行计算
        elif self.distance_type == 'sinkhorn':  # 当距离类型为 Sinkhorn 距离时
            s = torch.einsum('ld,nd->ln', z, self.embed_weight)  # BHW x N  # 计算矩阵积(张量乘积)，即将特征向量z与码本向量的转置相乘
            d = 0 - distributed_sinkhorn(s.detach())  # 对计算得到的矩阵进行深度迭代优化（多轮 Sinkhorn 迭代），获取最终的距离结果
            # import pdb; pdb.set_trace()
        elif self.distance_type == 'cosine':  # 当距离类型为余弦距离时
            d = 0 - torch.einsum('ld,nd->ln', z, self.embed_weight)  # BHW x N # 计算余弦距离
        else:
            raise NotImplementedError(
                'distance not implemented for {}'.format(self.distance_type))  # 抛出异常，提示尚未实现该距离度量类型

        if code_type == 'masked':
            d = d[:, self.masked_embed_start:]
        elif code_type == 'unmasked':
            d = d[:, :self.masked_embed_start]

        return d  # 返回计算出的距离矩阵

    def orth_loss(self, lamb=1e-3):
        w = self.embed_weight[:self.n_cluster]
        loss = w @ w.t() * (torch.ones(self.n_cluster, self.n_cluster) - torch.eye(self.n_cluster)).to(w.device)
        loss = torch.sum(loss ** 2)
        return loss * lamb

    def _quantize(self, z, token_type=None, topk=1, step=None, total_steps=None):
        """
        实现了对特征向量进行量化操作的函数 _quantize。

            :param z: 被量化的特征向量，大小为 L*D 的张量
            :param token_type: token 的类型。如果 token 未被加密，则为 1；否则为其他值。
            :param topk: 选择 top-k 个最相似的码本向量进行替换，其中 k = topk * step / total_steps
            :param step: 在 update_embed 函数中调用时使用，表示当前更新 embed 的步数
            :param total_steps: 在 update_embed 函数中调用时使用，表示更新 embed 的总步数
            :return:
                - z_q：量化后的向量，大小仍为 L*D 的张量
                - min_encoding_indices: 元素为最相似的码本向量索引的一维张量
                - cls_loss: 分类误差损失
                - orth_loss: 正交约束损失
                - used_cluster: 被使用到的码本簇的集合
        """
        d = self.get_distance(z)  # 获取特征向量 z 与码本向量之间的距离

        # find closest encodings
        # import pdb; pdb.set_trace()
        if self.n_cluster == 0:  # 如果码本聚类数量为 0
            # min_encoding_indices = torch.argmin(d, dim=1) # L
            min_encoding_indices = self.get_index(d, topk=topk, step=step, total_steps=total_steps)  # 直接返回最相似的码本向量
        else:  # 如果码本聚类数量不为 0
            # ？ min_encoding_indices = torch.zeros(z.shape[0]).long().to(z.device)   # 创建一个全部元素为 0 的一维张量
            self.update_semantic_label()  # 更新语义标签

            embed_semantic_label = F.one_hot(self.semantic_label).to(torch.float32)  # 对语义标签进行one_hot编码
            d_from_cluster = (d @ embed_semantic_label) / torch.sum(embed_semantic_label, dim=0)  # 计算距离码本簇平均值的距离
            likelihood_value = d_from_cluster  # 1024 x 16   # 计算概率值

            pos_emb = self.pos_enc(z).view(-1, self.e_dim)  # 1024 x 512 # 对特征向量进行位置编码
            likelihood_pos = pos_emb @ self.embed_weight[:self.n_cluster, :].t()  # 对特征向量与码本权重进行相乘
            # likelihood_pos = likelihood_pos / torch.sqrt(torch.sum((pos_emb ** 2), dim=1))  # 3.16 投影
            likelihood_pos = self.semantic_pos(likelihood_pos)  # 进行正则化

            # 使用 Gumbel softmax 函数计算权重，并使用投影进行约束
            likelihood = F.gumbel_softmax(likelihood_value + likelihood_pos, dim=1, tau=0.5, hard=False)
            # 等价于
            # likelihood_value = F.gumbel_softmax(likelihood_value, dim=0, tau=0.5, hard=False)
            # likelihood_pos = F.gumbel_softmax(likelihood_pos, dim=1, tau=0.5, hard=False)
            # likelihood = likelihood_value * likelihood_pos

            token_semantic_type = likelihood * torch.mean(embed_semantic_label,
                                                          dim=0)  # P(k|z) = P(z|k) * P(k) # 计算 token 的语义类型
            selector = token_semantic_type @ embed_semantic_label.t()  # 选择适当的编码簇
            d_ = d * selector  # 根据所选编码簇计算新的距离值
            min_encoding_indices = self.get_index(d_, topk=topk, step=step, total_steps=total_steps)  # 获取最相似的码本向量

            used_cluster = torch.argmax(token_semantic_type, dim=1).unique()  # 获取被使用到的码本簇
            # sem_num = torch.sum(F.one_hot(torch.argmax(token_semantic_type, dim=1)).to(torch.float32), dim=0)

            dis_p = get_gaussian_weight(self.n_cluster)
            dis_p = torch.softmax(dis_p / 0.5, dim=0)  # 计算高斯分布权重
            cls_loss = calculate_kl_loss(token_semantic_type, dis_p)  # 计算分类误差损失
            orth_loss = self.orth_loss()  # 计算正交约束损失

        # 根据获取到的码本向量替换特征向量 z
        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)  # 创建一个大小为 L*n_e 的零张量
            min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)  # 将最相似的码本向量所在位置置为 1
            # import pdb; pdb.set_trace()
            z_q = torch.matmul(min_encodings, self.embed_weight)  # .view(z.shape)  # 将全零矩阵与码本权重矩阵相乘，得到最终的量化向量 z_q
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(min_encoding_indices,
                              self.embed_weight)  # .view(z.shape)  # 将最相似的码本向量索引传入 F.embedding 函数，然后得到最终的量化向量 z_q
        else:
            raise NotImplementedError

        return z_q, min_encoding_indices  # , cls_loss, orth_loss, used_cluster  # 返回量化后的向量 z_q，最相似的码本向量索引，分类误差损失，正交约束损失以及使用到的码本簇

    def forward(self, z, token_type=None, topk=1, step=None, total_steps=None):
        """
            z: B x C x H x W
            token_type: B x 1 x H x W
        """
        # 判断距离类型是否是sinkhorn或cosine，若是，则在执行下一步前需要标准化输入向量
        if self.distance_type in ['sinkhorn', 'cosine']:
            # need to norm feat and weight embedding
            self.norm_embedding()
            z = F.normalize(z, dim=1, p=2)

        # reshape z -> (batch, height, width, channel) and flatten
        # 将z的形状变为(batch_size, height, width, channel)后展平为(BHW x C)的形状，并将token_type展平
        batch_size, _, height, width = z.shape
        # import pdb; pdb.set_trace()
        z = z.permute(0, 2, 3, 1).contiguous()  # B x H x W x C
        z_flattened = z.view(-1, self.e_dim)  # BHW x C

        token_type_flattened = None
        # 获取量化结果和最小编码索引
        z_q, min_encoding_indices = self._quantize(z_flattened, token_type=token_type_flattened, topk=1, step=None,
                                                   total_steps=None)
        # 将 z_q 从形状 (batch_size * height * width, D) 的张量转换为 (batch_size, height, width, D) 的张量
        z_q = z_q.view(batch_size, height, width, -1)  # .permute(0, 2, 3, 1).contiguous()

        if self.training and self.embed_ema:  # 判断是否为训练模式和是否使用指数滑动平均
            # import pdb; pdb.set_trace()
            assert self.distance_type in ['euclidean', 'cosine']
            # 将最小编码索引转换成 one-hot 向量
            indices_onehot = F.one_hot(min_encoding_indices, self.n_e).to(z_flattened.dtype)  # L x n_e
            # 计算每个聚类中包含的样本数量总和和样本嵌入向量的总和
            indices_onehot_sum = indices_onehot.sum(0)  # n_e
            z_sum = (z_flattened.transpose(0, 1) @ indices_onehot).transpose(0, 1)  # n_e x D

            # 执行全局归约操作，计算所有进程的 indices_onehot_sum 和 z_sum 的和
            all_reduce(indices_onehot_sum)
            all_reduce(z_sum)

            # 更新聚类中包含的样本数量总和和样本嵌入向量的总和的指数滑动平均值
            self.cluster_size.data.mul_(self.decay).add_(indices_onehot_sum, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(z_sum, alpha=1 - self.decay)
            # 计算嵌入向量的新的均值，并将其赋值为 self.embedding
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

        # 根据是否使用指数滑动平均计算损失
        if self.embed_ema:
            loss = (z_q.detach() - z).pow(2).mean()
        else:
            # compute loss for embedding
            loss = torch.mean((z_q.detach() - z).pow(2)) + self.beta * torch.mean((z_q - z.detach()).pow(2))

        # preserve gradients # 计算量化后的 z_q，并将其与原始向量 z 做差，再加上 z，得到最终的 z_q
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape # 将形状从 BxCxHxW 转换为 BxHxWxC
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        # 获取最小编码索引的唯一值
        unique_idx = min_encoding_indices.unique()
        # 返回结果字典
        output = {
            'quantize': z_q,
            'used_unmasked_quantize_embed': torch.zeros_like(loss) + (unique_idx < self.masked_embed_start).sum(),
            'used_masked_quantize_embed': torch.zeros_like(loss) + (unique_idx >= self.masked_embed_start).sum(),
            'quantize_loss': loss,
            'index': min_encoding_indices.view(batch_size, height, width)
        }
        return output

    def get_codebook_entry(self, indices, shape):
        # import pdb; pdb.set_trace()

        # shape specifying (batch, height, width)
        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
            min_encodings.scatter_(1, indices[:, None], 1)
            # get quantized latent vectors
            z_q = torch.matmul(min_encodings.float(), self.embed_weight)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(indices, self.embed_weight)
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()
        if shape is not None:
            z_q = z_q.view(*shape, -1)  # B x H x W x C

            if len(z_q.shape) == 4:
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        inplace = True
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),  # nn.LeakyReLU给非负值赋予一个非零斜率
        )  # spectral_norm利用pytorch自带的频谱归一化函数，给设定好的网络进行频谱归一化，主要用于生成对抗网络的鉴别器

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            #  谱归一化，为了约束GAN的鉴别器映射函数
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
