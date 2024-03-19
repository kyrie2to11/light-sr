import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    if (args.model_size == 'small'):
        return FAN(scale=args.scale[0], num_blocks=4)
    elif (args.model_size == 'medium'):
        return FAN(scale=args.scale[0], num_blocks=8)
    elif (args.model_size == 'large'):
        return FAN(scale=args.scale[0], num_blocks=12)
    else:
        raise NotImplementedError(f'{args.model} is not implemented.')

class BSConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 dilation=1,
                 bias=True):
        super(BSConv, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 padding='same',
                                 bias=bias) # fc layer
        
        self.conv_dw = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=out_channels,
                                 bias=bias) # depthwise conv


    def forward(self, inputs):
        x = self.conv1x1(inputs)
        x = self.conv_dw(x)
        return x

class PartialConv(nn.Module):
    def __init__(self,
                 in_channels,
                 bias=True,
                 stride=1,
                 extract_dim_fraction=0.5):
        super(PartialConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim_extract = int(in_channels * extract_dim_fraction)

        self.aggregator = nn.Conv2d(self.in_channels,
                                    self.out_channels,
                                    kernel_size=1,
                                    stride=stride,
                                    padding='same',
                                    bias=bias) # fc layer

        self.extractor_fraction_1 = nn.Conv2d(self.dim_extract,
                                              self.dim_extract,
                                              kernel_size=5,
                                              stride=stride,
                                              padding='same',
                                              bias=bias)

        self.extractor_fraction_2 = nn.Conv2d(self.dim_extract,
                                              self.dim_extract,
                                              kernel_size=5,
                                              stride=stride,
                                              padding='same',
                                              dilation=3,
                                              bias=bias)


    def forward(self, inputs):
        x = self.aggregator(inputs)
        # x1, x2 = x[:, :self.dim_extract], x[:, self.dim_extract:]
        # above numpy style slice operation may lead to trace fault, detailed error is as follows:
        '''
        torch._dynamo.exc.TorchRuntimeError: Failed running call_module L__self___mpfd_blocks_0_out_1_partial_conv_extractor_fraction_1(*(FakeTensor(..., size=(32, 16, 348), grad_fn=<SliceBackward0>),), **{}):
        Given groups=1, weight of size [16, 16, 5, 5], expected input[1, 32, 16, 348] to have 16 channels, but got 32 channels instead
        '''
        # so substitute it using torch.narrow() function -> same error sticks
        x1 = torch.narrow(x, 1, 0, self.dim_extract)
        x2 = torch.narrow(x, 1, self.dim_extract, self.in_channels - self.dim_extract)
        x1 = self.extractor_fraction_1(x1)
        x1 = self.extractor_fraction_2(x1)
        x = torch.cat([x1, x2], dim=1) # concat along channel dimension
        return x

class PFE(nn.Module):
    def __init__(self,
                 in_channels,
                 extract_dim_fraction=0.5,
                 mix_ratio=2,
                 bias=True):
        super(PFE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.partial_conv = PartialConv(self.in_channels, extract_dim_fraction=extract_dim_fraction)
        self.conv_mix = nn.Conv2d(self.in_channels,
                                  self.in_channels*mix_ratio,
                                  kernel_size=1,
                                  padding='same',
                                  bias=bias) # fc layer
        self.batch_norm = nn.BatchNorm2d(self.in_channels*mix_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(self.in_channels*mix_ratio,
                                 in_channels,
                                 kernel_size=1,
                                 padding='same',
                                 bias=bias) # fc layer

    def forward(self, inputs):
        x = self.partial_conv(inputs)
        x = self.conv_mix(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        return x

class PPA(nn.Module):
    def __init__(self,
                 in_channels,
                 extract_dim_fraction=0.5,
                 padding='same',
                 bias=True):
        super(PPA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim_extract = int(self.in_channels * extract_dim_fraction)

        self.conv1x1 = nn.Conv2d(self.in_channels,
                                 self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=padding,
                                 bias=bias) # fc layer
        
        self.relu = nn.ReLU(inplace=True)
        
        self.attn_conv = nn.Conv2d(self.dim_extract,
                                   self.dim_extract,
                                   kernel_size=3,
                                   stride=1,
                                   padding='same',
                                   bias=bias)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv1x1(inputs)
        x1, x2 = x[:, :self.dim_extract], x[:, self.dim_extract:]
        x1 = self.relu(x1)
        attn_score = self.attn_conv(x1)
        attn_weight = self.sigmoid(attn_score)
        x1 = x1 * attn_weight
        x = torch.cat([x1, x2], dim=1) # concat along channel dimension
        return x

class MPFD(nn.Module):
    def __init__(self, in_channels=16):
        super(MPFD, self).__init__()
        self.out_1 = PFE(in_channels)
        self.out_2 = PFE(in_channels)
        self.out_3 = PFE(in_channels)

        self.refine_1 = nn.Conv2d(in_channels,
                                  in_channels//4,
                                  kernel_size=1,
                                  stride=1,
                                  padding='same') # fc layer
        self.refine_2 = nn.Conv2d(in_channels,
                                  in_channels//4,
                                  kernel_size=1,
                                  stride=1,
                                  padding='same') # fc layer
        self.refine_3 = nn.Conv2d(in_channels,
                                  in_channels//4,
                                  kernel_size=1,
                                  stride=1,
                                  padding='same') # fc layer
        self.refine_inputs = nn.Conv2d(in_channels,
                                       in_channels//4,
                                       kernel_size=1,
                                       stride=1,
                                       padding='same') # fc layer
        self.ppa = PPA(in_channels=in_channels)

    def forward(self, inputs):
        out_1 = self.out_1(inputs)
        out_2 = self.out_2(out_1)
        out_3 = self.out_3(out_2)

        refine_1 = self.refine_1(out_1)
        refine_2 = self.refine_2(out_2)
        refine_3 = self.refine_3(out_3)
        refine_inputs = self.refine_inputs(inputs)

        out = torch.cat([refine_1, refine_2, refine_3, refine_inputs], dim=1) # concat along channel dimension
        out = self.ppa(out)
        return out

class ConvTail(nn.Module):
    def __init__(self,
                 in_channels,
                 num_out_ch,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 bias=False,
                 dilation=1,
                 scale=4):

        super(ConvTail, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              num_out_ch*scale*scale,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, inputs):
        return self.conv(inputs)

class FAN(nn.Module):
    def __init__(self, num_out_ch=3, scale=4, num_feat=32, num_blocks=4):
        super(FAN, self).__init__()
        self.num_out_ch = num_out_ch
        self.scale = scale
        self.num_feat = num_feat
        self.num_blocks = num_blocks

        self.conv_head = nn.Conv2d(in_channels=3,
                                   out_channels=self.num_feat,
                                   kernel_size=3,
                                   padding='same',
                                   bias=True)
        self.bsconv = BSConv(in_channels=self.num_feat, out_channels=self.num_feat)
        self.mpfd_blocks = nn.ModuleList([MPFD(in_channels=self.num_feat) for _ in range(self.num_blocks)])
        self.conv_tail = ConvTail(in_channels=self.num_feat, num_out_ch=self.num_out_ch, scale=self.scale)

    def forward(self, inputs):
        x_forward = self.conv_head(inputs)
        x_forward = self.bsconv(x_forward)

        for mpfd_block in self.mpfd_blocks:
            x_forward = mpfd_block(x_forward)

        out = self.conv_tail(x_forward)
        out = F.pixel_shuffle(out, self.scale)

        bilinear = F.interpolate(inputs, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + bilinear
        out = torch.clamp(out, 0, 255)

        return out

