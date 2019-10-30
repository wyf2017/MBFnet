#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseModule import BaseModule, SequentialEx

import traceback
import logging
logger = logging.getLogger(__name__)


visualize_refine = False # True # 
visualize_disps = False # True # 
if(visualize_refine or visualize_disps):
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt


ActiveFun = nn.ReLU(inplace=True) # nn.LeakyReLU(negative_slope=0.1, inplace=True) # 
NormFun2d = nn.BatchNorm2d # nn.InstanceNorm2d # 
NormFun3d = nn.BatchNorm3d # nn.InstanceNorm3d # 


def conv3x3(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding
    >>> module = conv3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1, 
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
    kargs['padding'] = kargs['dilation']
    return nn.Conv2d(in_channel, out_channel, 3, **kargs)


def conv3x3_bn(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
                conv3x3(in_channel, out_channel, **kargs), 
                NormFun2d(out_channel), ActiveFun, )


def conv3x3x3(in_channel, out_channel, **kargs):
    """
    3x3x3 Conv3d with padding
    >>> module = conv3x3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1, 
    >>> x = torch.rand(1, 1, 5, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3, 3]
    """
    
    kargs.setdefault('dilation', 1)
    kargs['padding'] = kargs['dilation']
    return nn.Conv3d(in_channel, out_channel, 3, **kargs)


def conv3x3x3_bn(in_channel, out_channel, **kargs):
    """
    3x3x3 Conv3d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
                conv3x3x3(in_channel, out_channel, **kargs), 
                NormFun3d(out_channel), ActiveFun, )


class SimpleResidual3d(nn.Module):
    """
    3D SimpleResidual
    >>> module = SimpleResidual3d(1, dilation=2)
    >>> x = torch.rand(1, 1, 5, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 5, 5, 5]
    """

    def __init__(self, planes, dilation=1):
        super(SimpleResidual3d, self).__init__()
        
        self.planes = planes
        self.conv1 = conv3x3x3(planes, planes, dilation=dilation, bias=False)
        self.bn1 = NormFun3d(planes)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation, bias=False)
        self.bn2 = NormFun3d(planes)
        self.ActiveFun = ActiveFun


    def forward(self, x):

        out = self.bn1(self.conv1(x))
        out = self.ActiveFun(out)
        out = self.bn2(self.conv2(out))
        out += x
        out = self.ActiveFun(out)

        return out


class SpatialPyramidPooling(nn.Module):
    """Spatial pyramid pooling
    >>> feature = torch.rand(2, 16, 5, 5)
    >>> msf = SpatialPyramidPooling(16, 4)
    >>> out = msf(feature)
    >>> list(out.shape)
    [2, 16, 5, 5]
    """

    def __init__(self, planes, kernel_size_first=4):
        super(SpatialPyramidPooling, self).__init__()

        self.planes = planes
        
        ks = kernel_size_first
        self.avg_pool1 = nn.AvgPool2d(ks, ceil_mode=True)
        self.avg_pool2 = nn.AvgPool2d(2, ceil_mode=True)
        self.avg_pool3 = nn.AvgPool2d(2, ceil_mode=True)
        self.avg_pool4 = nn.AvgPool2d((2, 1), ceil_mode=True)
        self.branch1 = conv3x3_bn(planes, planes)
        self.branch2 = conv3x3_bn(planes, planes)
        self.branch3 = conv3x3_bn(planes, planes)
        self.branch4 = conv3x3_bn(planes, planes)

        self.lastconv = SequentialEx(
                conv3x3_bn(planes*5, planes*3), 
                conv3x3(planes*3, self.planes, bias=False),
                )


    def forward(self, x):
        
        branchs = [x]
        favg1 = self.avg_pool1(x)
        favg2 = self.avg_pool2(favg1)
        favg3 = self.avg_pool3(favg2)
        favg4 = self.avg_pool4(favg3)
        
        h, w = x.shape[-2:]
        upfun = lambda feat: F.interpolate(feat, (h, w), mode='bilinear', align_corners=True)
        
        branchs.extend(list(map(lambda branch, feat: upfun(branch(feat)), 
                                [self.branch1, self.branch2, self.branch3, self.branch4], 
                                [favg1, favg2, favg3, favg4])))

        output = torch.cat(branchs, 1)
        output = self.lastconv(output)

        return output


class MultiBranchRefine(nn.Module):
    def __init__(self, in_planes, planes=32, branch=1):
        super(MultiBranchRefine, self).__init__()
        
        self.branch = max(1, branch)
        self.in_planes = in_planes
        self.planes = planes
        self.planes_b = planes*self.branch
        self.planes_o = self.branch*2 if self.branch>1 else 1
        
        self.conv = SequentialEx(
                conv3x3_bn(self.in_planes, self.planes, dilation=1), 
                conv3x3_bn(self.planes, self.planes, dilation=2), 
                conv3x3_bn(self.planes, self.planes, dilation=4), 
                conv3x3_bn(self.planes, self.planes, dilation=8), 
                conv3x3_bn(self.planes, self.planes, dilation=16), 
                conv3x3_bn(self.planes, self.planes_b, dilation=1), 
                conv3x3_bn(self.planes_b, self.planes_b, groups=self.branch), 
                conv3x3(self.planes_b, self.planes_o, groups=self.branch, bias=False),
                )


    def forward(self, input):

        out = self.conv(input)
        if(1 < self.branch):
            out_b = out[:, ::2]
            weight_b = out[:, 1::2]
            weight_b = F.softmax(weight_b, dim=1)
            out = (weight_b*out_b).sum(dim=1, keepdim=True)
            
            # visualize weight of branch
            if(visualize_refine):
                datas = [weight_b[:1], out_b[:1], weight_b[:1]*out_b[:1], out[:1], ]
                h, w = datas[0].shape[-2:]
                pad = max(1, max(h, w)//100)
                mw = -datas[0].view(-1, h*w).mean(dim=1)
                _, idxs = torch.sort(mw, descending=False)
                plt.subplot(111)
                for idx, imgs in enumerate(datas):
                    shape_view = [-1, 1] + list(imgs.shape[-2:])
                    imgs = imgs.view(shape_view).detach()
                    imgs = imgs.transpose(0, 1).contiguous().view(shape_view).clamp(-6, 6)
                    if(len(idxs)==len(imgs)):
                        imgs = imgs[idxs]
                    imgs = make_grid(imgs, nrow=out_b.size(1), padding=pad, normalize=False)
                    timg = imgs[0].data.cpu().numpy()
                    path_save = 'z_branch_%d_%d_%d.png' % (h//10, w//10, idx)
                    plt.imsave(path_save, timg)
                    plt.subplot(4, 1, idx+1); plt.imshow(timg)
                plt.show()

        else:
            # visualize refine
            if(visualize_refine):
                imgs = out[:1].detach().clamp(-6, 6)
                h, w = imgs.shape[-2:]
                pad = max(1, max(h, w)//100)
                imgs = make_grid(imgs, nrow=1, padding=pad, normalize=False)
                timg = imgs[0].data.cpu().numpy()
                path_save = 'z_refine_%d_%d.png' % (h//10, w//10)
                plt.imsave(path_save, timg)
                plt.subplot(111); plt.imshow(timg)
                plt.show()

        return out


#--------zLAPnet and two variant(zLAPnetF, zLAPnetR)---------#
class zMBFnet(BaseModule):
    '''Stereo Matching based on Multi-branch Fusion'''

    def __init__(self, maxdisp, str_kargs='S4B1W'):
        super(zMBFnet, self).__init__(maxdisp)
        
        kargs = self._parse_kargs(str_kargs)
        self.nScale = min(7, max(4, kargs[0]))
        self.nBranch = kargs[1]
        self.flag_wrap = kargs[2]
        
        k = 2**(self.nScale)
        self.shift = int(self.maxdisp//k) + 1
        self.disp_step = float(k)
        
        # feature extration for cost
        fn1s = [3, 32, 32, 32, 64, 64, 64][:self.nScale]
        fn2s = (fn1s[1:] + fn1s[-1:])
        SPPs = ([False]*2 + [True]*5)[:self.nScale]
        fks = ([4]*3 + [2]*4)[:self.nScale]
        self.convs = nn.ModuleList(map(self._conv_down2_SPP, fn1s, fn2s, SPPs, fks))
        self.modules_weight_decay = [self.convs]

        # feature fuse for refine
        fn1s_r = [n1 + n2 for n1, n2 in zip(fn1s, fn2s)]
        fn2s_r = [16] + fn1s[1:]
        self.convs_r = nn.ModuleList(map(conv3x3_bn, fn1s_r, fn2s_r))
        self.modules_conv = [self.convs_r]
        
        # cost_compute for intializing disparity
        self.cost_compute = self._estimator(fn1s[-1]*2, fn1s[-1])
        self.modules_conv += [self.cost_compute ]

        # refines
        if(self.flag_wrap):
            fn1s_rf = [fn2s_r[0]+1] + [n1*2 + 1 for n1 in fn2s_r[1:]]
        else:
            fn1s_rf = [n1 + 1 for n1 in fn2s_r]
        fn2s_rf = fn2s_r
        branchs = [self.nBranch]*(len(fn1s_rf)-1)
        refines_b = list(map(MultiBranchRefine, fn1s_rf[1:], fn2s_rf[1:], branchs))
        refine0 = self._refine_simple(fn1s_rf[0], fn2s_rf[0])
        self.refines = nn.ModuleList([refine0] + refines_b)
        self.modules_conv += [self.refines]
        
        # init weight
        self.modules_init_()
        

    @property
    def name(self):
        tname = '%s_S%dB%d' % (self._get_name(), self.nScale, self.nBranch)
        return tname+'W' if self.flag_wrap else tname


    def _parse_kargs(self, str_kargs):
        nScale, nBranch = 5, 1
        regex_args = re.compile(r's(\d+)b(\d+)')
        res = regex_args.findall(str_kargs.lower())
        assert 1 == len(res), str(res)
        nScale, nBranch = int(res[0][0]), int(res[0][1])
        flag_wrap = 'w' in str_kargs.lower()
        return nScale, nBranch, flag_wrap


    def _conv_down2_SPP(self, in_planes, planes, SPP=False, fks=4):

        if SPP:
            return SequentialEx(
                        conv3x3_bn(in_planes, planes, stride=2), 
                        conv3x3_bn(planes   , planes, stride=1), 
                        SpatialPyramidPooling(planes, fks)
                        )
        else:
            return SequentialEx(
                        conv3x3_bn(in_planes, planes, stride=2), 
                        conv3x3_bn(planes   , planes, stride=1), 
                        )


    def _estimator(self, in_planes, planes):

        return SequentialEx(
                conv3x3x3_bn(in_planes, planes), 
                SimpleResidual3d(planes), 
                SimpleResidual3d(planes), 
                conv3x3x3(planes, 1, bias=False), 
                )


    def _refine_simple(self, in_planes, planes):

        return SequentialEx(
                conv3x3_bn(in_planes, planes, dilation=1), 
                conv3x3_bn(planes, planes, dilation=2), 
                conv3x3(planes, 1, bias=False),
                )


    def forward(self, imL, imR):

        bn = imL.size(0)

        # feature extration---forward
        x = torch.cat([imL, imR], dim=0)
        convs = [x]
        for i in range(self.nScale):
            x = self.convs[i](x)
            convs.append(x)
        
        # cost and disp
        shift = min(convs[-1].size(-1), self.shift) # conv5.size(-1)*3//5 # 
        cost = self.disp_volume_gen(convs[-1][:bn], convs[-1][bn:], shift, 1)
        cost = self.cost_compute(cost).squeeze(1)
        disp = self.disp_regression(cost, 1.0)

        disps = [disp]

        # conv_merge, refine_simple, refine_wrap
        def conv_merge(mConv, conv, conv_last):
            conv_last = self._upsample_as_bilinear(conv_last, conv)
            conv = torch.cat([conv, conv_last], dim=1)
            conv = mConv(conv)
            return conv

        def refine_simple(mRefine, fL, disp_last):
            disp = self._upsample_as_bilinear(disp_last.detach()*2.0, fL)
            input = torch.cat([fL, disp], dim=1)
            disp1 = disp + mRefine(input)
            return disp1

        def refine_wrap(mRefine, fL, fR, disp_last):
            disp = self._upsample_as_bilinear(disp_last.detach()*2.0, fL)
            factor = 2.0/(disp.size(-1) - 1)
            fL_wrap = self.imwrap(fR.detach(), disp*factor)
            input = torch.cat([fL, fL_wrap, disp], dim=1)
            disp1 = disp + mRefine(input)
            return disp1

        # hierarchical refine
        x = convs[-1]
        for i in range(self.nScale-1, 0, -1):
            
            x = conv_merge(self.convs_r[i], convs[i], x) # feature fusion---inverse
            convs[i] = x

            if(self.flag_wrap):

                # refine with wraped left feature
                m_bRf, fL, fR, disp_last = self.refines[i], x[:bn], x[bn:], disps[0]
                disp = refine_wrap(m_bRf, fL, fR, disp_last)
                disps.insert(0, disp)
            
            else:

                # refine with wraped left feature
                m_bRf, fL, disp_last = self.refines[i], x[:bn], disps[0]
                disp = refine_simple(m_bRf, fL, disp_last)
                disps.insert(0, disp)

        x = conv_merge(self.convs_r[0], imL, x[:bn])
        m_bRf, fL, disp_last = self.refines[0], x, disps[0]
        disp = refine_simple(m_bRf, fL, disp_last)
        disps.insert(0, disp)

        # visualize_disps
        if(visualize_disps):
            plt.subplot(111)
            col = 2
            row = (len(disps)+col-1)//col
            for idx, disp in enumerate(disps):
                timg = disp[0, 0].data.cpu().numpy()
                plt.imsave('z_disp%d.png'%idx, timg)
                plt.subplot(row, col, idx+1); plt.imshow(timg);
            plt.show()
        
        # return
        if(self.training):
            loss_ex = torch.zeros(1).type_as(imL)
            for disp, x in zip(disps[1:], convs[1:]):
                x, factor = x.detach(), 2.0/(x.size(-1) - 1)
                loss_ex += 0.1*self._loss_feature(disp*factor, x[:bn], x[bn:])
            return loss_ex[None], disps
        else:
            return disps[0].clamp(0)


    def get_parameters(self, lr=1e-3,  weight_decay=0):
        ms_weight_decay = self.modules_weight_decay
        ms_conv = self.modules_conv
        return self._get_parameters_group(ms_weight_decay, ms_conv, lr,  weight_decay)
    

    def compute_loss(self, disps, disp_true):
        return self._loss_disps(disps, disp_true)


def get_model_by_name(name, maxdisp):
    
    tmp = name.split('_')
    name_class = tmp[0]
    assert 2==len(tmp)
    str_kargs = tmp[1]
    try:
        return eval(name_class)(maxdisp, str_kargs)
    except:
        raise Exception(traceback.format_exc())


if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
    list_name = ['zMBFnet_S5B2W', 'zMBFnet_S5B2', ]
    for name in list_name:
        model = get_model_by_name(name, 192)
        logger.info('%s passed!\n ' % model.name)

