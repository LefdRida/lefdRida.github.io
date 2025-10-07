---
layout: distill
title: Depth Estimation
description:
tags: Technical-post, ML
giscus_comments: false
date: 2025-03-18 11:12:00-0400
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Rida Lefdali
    url: "www.linkedin.com/in/rlefdali/"
    affiliations:
      name: Independent
toc:
  - name: Monocular 3D Mapping
  - name: NUY Depth V2 dataset
  - name: Approach and Architecure
  - name: First Stage: Depth Estimation
  - name: focal length and depth shift estimation
  - name: Loss and Metrics
  - name: Data Preparation
  - name: Implementation
  - name: Depth shift and focal scale network
  - name: Quantitative Results
  - name: Qualitative Results

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Monocular 3D Mapping

Depth estimation is a task consisting of estimating the distance of each pixel realtive to the camera to result in a depth map. Using the depth map with some camera parameters, explained later, we can map an image $ I \in\mathbb{R}^{M\times N \times 3}$ to its 3D point cloud $X \in\mathbb{R}^{M \times N \times 3}$. This means, we can map each pixel in the 2D image to its corresponding point in the 3D space. 

A camera projects a 3D point in the real worl to a 2D point on the image. This transformation can be modeled using matrix multiplication using the intersic camera parameters and extrinsic camera parameters. 
The camera extrinsic matrix (parameters), denoted $E$, is used to map a 3D point in the real world reference to a 3D point cloud in the camera reference.
The camera intrinsic matrix (parameters), denoted $K$ is, used to map a 3D camera centered point to a 2D point on the image plane.

The matrix $K$ and $E$ are defined as follows: 


$$ \left\lbrace
\begin{aligned}
& K = 
\begin{pmatrix}
&f_x  &0   & c_x  \\
&0   &f_y   & c_y  \\
&0   &0   & 1  \\
\end{pmatrix}
\substack{\text{such that $(f_{x}, f_{y})$ are the focal lengths of the camera} \\ \text{and $(c_{x}, c_{y})$ is the camera optical center.}} \\
\text{and}\\
& E = [R|T] =
\begin{pmatrix}
&r_{11}   &r_{12}   & r_{13} & t_{1} \\
&r_{21}   &r_{22}   & r_{23} & t_{2} \\
&r_{31}   &r_{32}   & r_{33} & t_{3} \\
\end{pmatrix} 
\substack{\text{which contains the rotation matrix} \\ \text{and translation vector}} \\
\end{aligned}
\right. $$


So, given the intrinsic matrix $K$ and the depth map $D$, and using the pinhole camera model, we can map each pixel $p_{i}=(x_{i}, y_{i})^{T}$ in the 2D image plane to its corresponding the 3D point in the camera referential $P_{c, i} = (X_{c, i}, Y_{c, i}, Z_{c, i})^{T}$
using the schema below: 

$$ \left\lbrace
\begin{aligned}
& X_{c, i} = \frac{(x_{i} - c_x)Z_{c, i}}{f_x}\\
& Y_{c, i} = \frac{(y_{i} - c_y)Z_{c, i}}{f_y}\\
& Z_{c, i} = D(x_{i}, y_{i}) \quad \text{(the depth value corresponding to the pixel $p_{i}$)}
\end{aligned}
\right. $$

We will be interested in only on mapping the image pixel to its 3D point in the camera referential.\

To map to the 3D real world space, we can use the relationshape between a point in the camera referential $P_{c}$ and its corresponding point in the real world referential $P_{r}$, which is:

$$ \left\lbrace
\begin{aligned}
& P_{c} = RP_{r} + T  \quad \text{where $P_{r} = (X_{r}, Y_{r}, Z_{r})^{T}$,  $T$ and $R$ are the translation vector and rotation matrices}\\
& RP_{r} = P_{c} - T \\
& P_{r} = R^{T}(P_{c} - T) \quad \text{The rotation matrix is always invertible. And as it is orthogonal ans its inverse is its transpose.} \\
\end{aligned}
\right. $$

The following python function implements the pinhole model to generate 3D point in the camera referential using depth map and the intrinsic matrix.

```python 
def generate_point_cloud(depth_map, intrinsic_matrix, depth_info, rgb_image=None):
    # Image dimensions
    H, W = depth_map.shape
    # Intrinsic parameters
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    depth_max = depth_info[3] # If depth info are provided
    depth_min = depth_info[0]
    
    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # Convert to normalized camera coordinates
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    # Back-project to 3D camera coordinates
    z_cam = depth_map
    x_cam = z_cam * x_norm
    y_cam = z_cam * y_norm
    # Stack into a 3D array (camera coordinates)
    points_cam = np.stack((x_cam, z_cam, -1*y_cam), axis=-1).reshape(-1, 3)
    if rgb_image is not None:
      colors = rgb_image[v, u] / 255.0
    return points_cam, colors
```

## NUY Depth V2 dataset

NUY Depth V2 is a large dataset that contains indoor scenes. The following presents some scene examples from the data and different views of the constructed 3D point of these scenes.


{% include figure.liquid loading="eager" path="assets/img/NUY_depth_data.png" class="img-fluid rounded z-depth-1" zoomable=true %}

## Approach and Architecure

The objective is to map the image $ I \in\mathbb{R}^{M\times N \times 3}$ to its 3D point cloud $X \in\mathbb{R}^{M\times N \times 3}$ . we can think about a naive or simple autoencoder decoder architecture such as UNET. Where the econder learn the image representation and the decoder map the representation to 3D point cloud. UNET architecture has been tested but it did not give satisfying results.

Our focus will be focus on the camera referential. If we consider the Pinhole camera model, we have the cordinaates $X$, $Y$ and $Z$ depend on the depth map as shown in equation (1).

The idea is instead to learn a mapping function of the image to the 3D point cloud, we will think of a design that estimates the depth map and the camera parameters.

We will follow an approach proposed by Yin et al [6] that has two stages:

1. The first stage is a training a neural network to learn estimating the depth map that will be used to reconstruct a 3D point cloud using standard camera parameters. Not using the correct parameters, precisely, the focal length will result in a distorded point cloud even the global shape is preserved. 

2. The second stage then is composed from two networks each take as input a distorded point cloud and estimate either a focal scale or depth shift to restore the distorded 3D point. 

## First Stage: Depth Estimation : 
For depth map estimation eigen et al [1] introduced a CNN model levereging on multi scale network.This approach involves training a coarse scale network to predict the depth map at global level which is subsequently refined by a secondary network to refine the local regions. Using a multi-scale architecture proven to be effective, Xian et al [7] proposed a multi scale network that use a feature fusion module to fuse features from the encoder and decoder at different scales to obtain finer prediction. Additionnaly to using a multiscale architecture, in Big to Small model [2], the author proposed, under the assumption of locar planar, a local planar guidance module to guide features to the final depth. Unlike other methods which use only skip connection from encoder stage and upsampling to recover the final depth.

We will use BTS model for this task. BTS has an encoder-decoder architecture. The encoder outputs a dense feature map of $H/8$ resolution. The decoding phase consists of $4$ stages. Each stage $k$, (with $k \in {8, 4, 2, 1}$) takes a dense feature map of $H/k$ resolution and apply two operations:
1. We apply an up-convolution operation to have a feature map of resolution of $2H/k$. 
2. We apply local planar guidance to produce a coarse depth map $\tilde{c}^{k\times k}$ of resolution $H$ which downsampled using linear interpolation to a resolution of $2H/k$
The output of these two operations are element-wise multiplied and passed through a convolution operation to have the input feature map of resolution $2H/k$ for the next stage. 

To have the estimated depth map $d$, all the coersed depth map produced by the local planar guidance are used together as follows:

$$d = f(W_1 \tilde{c}^{1 \times 1} + W_2 \tilde{c}^{2\times 2} + W_3 \tilde{c}^{4\times 4} + W_4\tilde{c}^{8\times 8})$$ 

**the LPG module:**  Given a feature map having a spatial resolution $H/k$, it estimates $4D$ plane coefficient for each spatial cell to reconstruct a coarse depth that fit a locally defined $k\times k$ patch on the full resolution. The LPG uses ray-plane intersection to convert each estimated 4D plane coefficient to $k\times k$ local depth cues on the full resolution:

$$c = \frac{n_{4}}{n_{1}u_{i} + n_{2}v_{i} + n_{3}}$$

where $n = (n_{1}, n_{2}, n_{3}, n_{4})$ are the estimated 4D parameters and $(u_{i}, v_{i})$ \are $k\times k$ patch wise normalized coordinate of pixel $i$

The $n$ parameters are the plane parameters where $(n_{1}, n_{2}, n_{3})$ is the normal vector and $n_{4}$ is the distance from the origine to the plane. To estimate these parameters, they use the fact that a normal vector can be computed using two angles, polars and azimuthal, using the the following formulas: (more details in the Appendix)

$$ \left\lbrace
\begin{aligned}
& n_{1} = sin(\theta)cos(\phi)\\
& n_{2} = sin(\theta)sin(\phi)\\
& n _{3} = cos(\theta)\\
& n_{4} = d
\end{aligned}
\right. $$

To estimates these three parameters at the scale $H/k$, the LPG takes as input the feature of the previous fusion module, i.e feature at scale $H/2k$, and pass them through a series of $1\times1$ convolution to reduce the number of channels by a factor of $2$ until it reaches $3$. So the final convolution layer of the LPG estimates $\theta$, $\phi$ and $d$. More details about the computation are in the implementation of this module.
Thus using the LPG, we have an estimation of depth map at different scales. The lower scales learns the global shapes and the higher scales learns local details. 
The final depth is estimated through a convolution that takes all the depth map at each scale

## focal length and depth shift estimation

The input distorded point cloud are created as follows:
1. for the first network, we shift the depth map by a value drawn from a uniform distribution. The shift will result in a distorded shape as it will affect non uniformly, the X, Y, and Z. So, the goal of the first network is to estimate the depth shift value.
2. For the second network, the distortion is created by scaling the focal length by a coefficient drawn from a uniform distribution. This scaling will affect only X and Y and will results in points far from each other or closer to each other.

to estimate the depth shift and focal length shift. As in [6] we have neural network based on PointNet to predict the depth shift or focal length shift given a input of distorted point cloud: 

$$L_{depth\ shift} = min_{\theta} |N_d(F(u_0, v_0, f^{*}, d^{*} + ∆^{*}_d), \theta) − ∆^{*}_d|$$

where $$\Delta^{*}_d$$ is drawn from a uniform distribution $$\text{Uniform}(-0.25, 0.8)$$ during training

$$L_{focal\ scale} = min_{\theta} |N_d(F(u_0, v_0, \alpha^{*}f^{*}, d^{*}), \theta) − \alpha^{*}|$$

where $$\alpha^{*}$$ is drawn from a uniform distribution $$\text{Uniform}(0.6, 1.25)$$ during training


The network used for focal scale estimation and depth shift estimation is a PoinNet [9] applied for regression task. The network takes as input a 3D point cloud (B, N, 3) and apply an **input transformation block** followed by a 1D conv layer to extract features. A **feature transformation block** is applied on the extracted and followed by a series of 1D conv operation to result in a vector of global feature using a max aggregation. The max pooling is used to have invariance w.t.r point cloud order.
The  **input transformation block** and  **feature transformation block**  contains a series of conv and linear operations and use relu as activation function and max pooling to aggregates information. The transormation block is used to transform the point cloud to a canonical form to have invariance w.r.t to transformation.

## Loss and Metrics

The depth estimation, focal length scale estimation or depth shift estimation are regression task. For the loss will be based of L1 or MSE loss 
In depth estimation many losses function have been proposed in the literature such as Huber Loss, silog loss, ordinal regression loss. In our case, we will use scale invariance log which computes the error between the ground truth and the prediction without taking into account the scale discrepency.So, it consider only the relative error between the values.

$$L(\hat{d}, d) = \frac{1}{n}\sum_{p}{||ln(d^{*}_{p}) - ln(d_{p})||^{2}} - \frac{1}{n^{2}}(\sum_{p}{(ln(d^{*}_{p}) - ln(d_{p}))})^{2}$$

for evaluation we consider the following metrics used in the literature: 

- **Accuracy under a threshold** $\delta$ % of $$p  :  \delta = max(\frac{\hat{d}_p}{d_p}, \frac{d_p}{\hat{d}_p}) < threshold$$

- **Abs. Rel.:** Mean Absolute Value of the Relative Error. 
$$\frac{1}{T} \sum_{p \in T} \left| \frac{d_p - \hat{d}_p}{d_p} \right|$$


## Data Preparation

The NUY depth dataset is handled respectively by `NUYDepth` class that stores samples of images, depth map for training depth map network and point cloud with depth shift or focal scale for PointNet networks. The class contains the method
```python 
def _generate_point_cloud():
``` 
which generate a point cloud using Pinhol camera model. 


The image and depth map are resized to (256, 256) for computational reasons in the case we are training for depth map estimation. For depth shift or focal scale estimation, we keep the full resolution of the depth map to construct the 3D point cloud, but we use only a sample of 4096 points for for computational reasons. 

The images are normalized using mean and std. We use data augmentation, flipping and brightness adjusting as we are using only a subset of 1300 image from the data The original data has 490GB. 


## Implementation

Here the implementation of the models, datasets and training loops. The results discussion are right after the following cells.

```python 
import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math

from collections import namedtuple

# This sets the batch norm layers in pytorch as if {'is_training': False, 'scale': True} in tensorflow
def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))
        
        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels*2, momentum=0.01, affine=True, track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation), dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)
    

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()        
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                                 kernel_size=1, stride=1, padding=0),
                                                                       nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2
    
    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
        
        return net

class local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]
        
        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3))#.cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio
        
        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio))#.cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)

class bts(nn.Module):
    def __init__(self, feat_out_channels, num_features=512):
        super(bts, self).__init__()
        self.max_depth = 1.
        self.upconv5    = upconv(feat_out_channels[4], num_features)
        self.bn5        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        
        self.conv5      = torch.nn.Sequential(nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.upconv4    = upconv(num_features, num_features // 2)
        self.bn4        = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4      = torch.nn.Sequential(nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.bn4_2      = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        
        self.daspp_3    = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6    = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12   = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18   = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24   = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.reduc8x8   = reduction_1x1(num_features // 4, num_features // 4, self.max_depth)
        self.lpg8x8     = local_planar_guidance(8)
        
        self.upconv3    = upconv(num_features // 4, num_features // 4)
        self.bn3        = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3      = torch.nn.Sequential(nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.reduc4x4   = reduction_1x1(num_features // 4, num_features // 8, self.max_depth)
        self.lpg4x4     = local_planar_guidance(4)
        
        self.upconv2    = upconv(num_features // 4, num_features // 8)
        self.bn2        = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2      = torch.nn.Sequential(nn.Conv2d(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias=False),
                                              nn.ELU())
        
        self.reduc2x2   = reduction_1x1(num_features // 8, num_features // 16, self.max_depth)
        self.lpg2x2     = local_planar_guidance(2)
        
        self.upconv1    = upconv(num_features // 8, num_features // 16)
        self.reduc1x1   = reduction_1x1(num_features // 16, num_features // 32, self.max_depth, is_final=True)
        self.conv1      = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.get_depth  = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())

    def forward(self, features, focal):
        skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]
        dense_features = torch.nn.ReLU()(features[4])
        upconv5 = self.upconv5(dense_features) # H/16
        upconv5 = self.bn5(upconv5)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)
        
        upconv4 = self.upconv4(iconv5) # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)
        
        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)
        
        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = torch_nn_func.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8 = self.lpg8x8(plane_eq_8x8, focal)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.max_depth
        depth_8x8_scaled_ds = torch_nn_func.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='nearest')
        
        upconv3 = self.upconv3(daspp_feat) # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)
        
        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = torch_nn_func.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.max_depth
        depth_4x4_scaled_ds = torch_nn_func.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')
        
        upconv2 = self.upconv2(iconv3) # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)
        
        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = torch_nn_func.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.max_depth
        
        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        concat1 = torch.cat([upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
        iconv1 = self.conv1(concat1)
        final_depth = self.max_depth * self.get_depth(iconv1)

        
        return final_depth, depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        import torchvision.models as models
      
        self.base_model = models.resnet50(pretrained=True)
        self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
        self.feat_out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        feature = x
        skip_feat = []
        i = 1
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
            i = i + 1
        return skip_feat
    

class BtsModel(nn.Module):
    def __init__(self):
        super(BtsModel, self).__init__()
        self.encoder = encoder()
        self.decoder = bts(self.encoder.feat_out_channels, 512)

    def forward(self, x, focal=1):
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat, focal)
```

## Depth shift and focal scale network

```python
    
class MEADSTD_TANH_NORM_Loss(nn.Module):
    """
    loss = MAE((d-u)/s - d') + MAE(tanh(0.01*(d-u)/s) - tanh(0.01*d'))
    """
    def __init__(self, valid_threshold=-1e-8, max_threshold=1e8):
        super(MEADSTD_TANH_NORM_Loss, self).__init__()
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold
        #self.thres1 = 0.9
 
    def transform(self, gt):
        # Get mean and standard deviation
        data_mean = []
        data_std_dev = []
        for i in range(gt.shape[0]):
            gt_i = gt[i]
            mask = gt_i > 0
            depth_valid = gt_i[mask]
            if depth_valid.shape[0] < 10:
                data_mean.append(torch.tensor(0).cuda())
                data_std_dev.append(torch.tensor(1).cuda())
                continue
            size = depth_valid.shape[0]
            depth_valid_sort, _ = torch.sort(depth_valid, 0)
            depth_valid_mask = depth_valid_sort[int(size*0.1): -int(size*0.1)]
            data_mean.append(depth_valid_mask.mean())
            data_std_dev.append(depth_valid_mask.std())
        data_mean = torch.stack(data_mean, dim=0).cuda()
        data_std_dev = torch.stack(data_std_dev, dim=0).cuda()

        return data_mean, data_std_dev

    def forward(self, pred, gt):
        """
        Calculate loss.
        """
        #gt = torch.nn.functional.interpolate(gt,
        #                                    size=(
        #                                        pred.size()[2], pred.size()[3]),
        #                                    mode='nearest').to('cuda')
        mask = (gt > self.valid_threshold) & (gt < self.max_threshold)   # [b, c, h, w]
        mask_sum = torch.sum(mask, dim=(1, 2, 3))
        # mask invalid batches
        mask_batch = mask_sum > 100
        if True not in mask_batch:
            return torch.tensor(0.0, dtype=torch.float).cuda()
        mask_maskbatch = mask[mask_batch]
        pred_maskbatch = pred[mask_batch]
        gt_maskbatch = gt[mask_batch]

        gt_mean, gt_std = self.transform(gt_maskbatch)
        gt_trans = (gt_maskbatch - gt_mean[:, None, None, None]) / (gt_std[:, None, None, None] + 1e-8)

        B, C, H, W = gt_maskbatch.shape
        loss = 0
        loss_tanh = 0
        for i in range(B):
            mask_i = mask_maskbatch[i, ...]
            pred_depth_i = pred_maskbatch[i, ...][mask_i]
            gt_trans_i = gt_trans[i, ...][mask_i]

            depth_diff = torch.abs(gt_trans_i - pred_depth_i)
            loss += torch.mean(depth_diff)

            tanh_norm_gt = torch.tanh(0.01*gt_trans_i)
            tanh_norm_pred = torch.tanh(0.01*pred_depth_i)
            loss_tanh += torch.mean(torch.abs(tanh_norm_gt - tanh_norm_pred))
        loss_out = loss/B + loss_tanh/B
        return loss_out.float()

class Shift_Loss(nn.Module):
    def __init__(self):
        super(Shift_Loss, self).__init__()
    def forward(pred, gt):
        return torch.abs(pred - gt)


class MAE_error(nn.Module):
    def __init__(self):
        super(MAE_error, self).__init__()

    def forward(self, prediction, gt):
        # prediction = prediction[:, 0:1]
        abs_err = torch.abs(prediction - gt)
        mask = (gt > 1e-3).detach()
        mae_error = torch.mean(abs_err[mask])
        return mae_error

class RelMAE_error(nn.Module):
    def __init__(self):
        super(RelMAE_error, self).__init__()
    def forward(self, prediction, gt):
        # prediction = prediction[:, 0:1]
        #prediction = torch.clamp(prediction, min=1e-4)
        prediction = torch.abs(prediction)
        abs_err = torch.abs(prediction - gt)
        mask = (gt > 1e-3).detach()
        mae_error = torch.mean(abs_err[mask]/gt[mask])
        return mae_error
class Gamma_Metric(nn.Module):
    def __init__(self):
        super(Gamma_Metric, self).__init__()

    def forward(self, prediction, gt):
        # prediction = prediction[:, 0:1]
        mask = (gt > 1e-3).detach()
        prediction = torch.abs(prediction) #torch.clamp(prediction, min=1e-4)
        max_proportion = torch.max(prediction[mask]/gt[mask], gt[mask]/prediction[mask])
        gamma = torch.mean(1.*(max_proportion < 1.25))
        return gamma
```

## Quantitative Results


We plot the evolution of the training and test loss for both models on both dataset. Both models converge on both dataset except for the model adapt on blended MVS were the test loss is fluctuating along a single value which does not show any decreasing trend and that could be a sign of the inability of the model on that dataset. This could be due to the fact that the model is to small to generalize on that data. 

The table below resume the evaluation metrics on both dataset and we have the adapt model with the local planar shows slight increase in both metrics accuracy under threshold and relative MAE, for both datasets, comparing to the adapt modal alone. For the accuracy under threshold the highest the better and for relative MAE the lower the better.

|         | Accuracy under threshold      | Relative MAE       |  RMSE       |
|----------------|----------------|----------------|----------------|
| NUY Depth  (Adaptive)| 0.530  | 4.633 | 1.353 |
| NUY Depth  (Adaptive with local planar)|0.013 |1.454| 17.2 |

## Qualitative Results

The pipeline for inference starts by predicting the depth map, on which we apply a sigmoid function to have positive values. The depth map prediction model outputs depth maps with negative values. we Have test using sigmoid, ReLU or Softplus activation function to output the final depth map during training but it led to bad results. Then, we estimate the focal length and the depth shift. To do so, we create a 3D point cloud from a standard focal length and camera optical center. Then we predict refine the focal length by estimating a focal scale from the constructed 3D point. We use the new focal length to construct new 3D point cloud and to refine then the depth by estimating a depth shift. Again use the refined depth to refine another time the focal length. Then we have the final depth and the focal length that we use to generate the final 3D point cloud. 


{% include figure.liquid loading="eager" path="assets/img/NUY_depth_data_prediction.png" class="img-fluid rounded z-depth-1" zoomable=true %}

Qualitativly, we can see that the approach can capture the global shapes, but the point cloud are very distorded. For example if we compare the left views and top views between group truth and the prediction, we can see easily the huge difference which could be due to the depth prediction.


