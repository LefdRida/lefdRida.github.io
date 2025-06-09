---
layout: post
title: Depth Estimation
date: 2025-03-20 11:12:00-0400
description: 
tags: 
categories: Technical-post
related_posts: false
---


### 3D Task

The task aims to train a neural network $f$ to map an Image $ I \in\mathbb{R}^{M\times N \times 3}$ to 3D cloud points $X \in\mathbb{R}^{M \times N \times 3}$. This means, the model has to map each pixel in the 2D image to its corresponding point in the 3D space. 


The work on this task is carried on Kaggle which offers 30H/week of usage of GPU with 16GB of VRAM.


### Dataset:

During this task, datasets such as blendedMVS or NUY depth have been considered. The datasets contains scene images with their depth map and camera intrinsic and extrinsic matrices, denoted $D$, $K$ and $E$ respectively.

Where: 

\begin{equation} K = 
\begin{pmatrix}
&f_x  &0   & c_x  \\
&0   &f_y   & c_y  \\
&0   &0   & 1  \\
\end{pmatrix}
\text{such that $(f_{x}, f_{y})$ are the focal lengths of the camera and $(c_{x}, c_{y})$ is the camera optical center.}
\end{equation} 


and 

\begin{equation} E = [R|T] =
\begin{pmatrix}
&r_{11}   &r_{12}   & r_{13} & t_{1} \\
&r_{21}   &r_{22}   & r_{23} & t_{2} \\
&r_{31}   &r_{32}   & r_{33} & t_{3} \\
\end{pmatrix} 
\text{which contains the rotation matrix and translation vector}
\end{equation} 

So, Given the intrinsic matrix $K$ and the depth map $D$, and using the pinhole camera model, we can map each pixel $p_{i}=(x_{i}, y_{i})^{T}$ in the 2D image plane to its corresponding the 3D point $P_{i} = (X_{i}, Y_{i}, Z_{i})^{T}$
in the camera referential using the schema below: 

$$X_{i} = \frac{(x_{i} - cx)Z_{i}}{f_x}$$
$$Y_{i} = \frac{(y_{i} - cy)Z_{i}}{f_y}$$
$$Z_{i} = D(x_{i}, y_{i}) \quad \text{(the depth value corresponding to the pixel $p_{i}$) } \\ \\$$


To map to the 3D real world space, we can use the relationshape between a point in the camera referential $P_{c}$ and its corresponding point in the real world referential $P_{r}$, which is:
$$P_{c} = RP_{r} + T  \quad \text{where $P_{r} = (X_{r}, Y_{r}, Z_{r})^{T}$,  $T$ and $R$ are the translation vector and rotation matrices}$$
$$RP_{r} = P_{c} - T$$
$$P_{r} = R^{T}(P_{c} - T) \quad \text{The rotation matrix is always invertible. And as it is orthogonal ans its inverse is its transpose.}$$


```python 
def generate_point_cloud(depth_map, rgb_image=None,  extrinsic_matrix=None, intrinsic_matrix=None, cam_path=None, fx=None, fy=None, cx=None, cy=None):
    # Image dimensions
    H, W = depth_map.shape
    if cam_path is not None:
      with open(cam_path, 'r') as f:
        lines = list(f.read().splitlines())
        intrinsic_matrix =np.array([l.split() for l in lines[7:10]], dtype=float)
        depth_info = np.array(lines[11].split(), dtype=float)
    # Intrinsic parameters
    depth_min = 1e-4
    depth_max = 10.
    fx, fy = fx, fy
    cx, cy = cx, cy

    if intrinsic_matrix is not None:
      fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
      cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
      depth_max = depth_info[3]
      depth_min = depth_info[0]
    

    if fx is None:
      fx, fy = 5.826e+02, 5.826e+02
    if cx is None:
      cx, cy = 3.130e+02, 2.384e+02
    
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
    
    mask = np.logical_and(depth_map>depth_min, depth_map<depth_max)
    colors = None
    if rgb_image is not None:
      colors = rgb_image[v, u] / 255.0
    return points_cam.reshape((H, W, 3)), colors, mask
```

##### Blended MVS dataset

BlendedMVS is a large scale dataset that contains scenes with multiple views. The following presents some scene examples from the data and different views of the constructed 3D point of these scenes.

```python 
data= create_mvs_data()
pl_data = pl.DataFrame(data)
pl_data = pl_data.filter(pl.col('category_id')=='5c20ca3a0843bc542d94e3e2')
sub_data = pl_data.to_pandas()
gt_blendedmvs_data = {}
all_images = []
mean = np.array([0, 0, 0], dtype=np.float64)
std = np.array([0, 0, 0], dtype=np.float64)
for i, row in sub_data.iterrows():
    depth_map = cv2.imread(row['depth_map_path'], cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.imread(row['img_path'])
    all_images.append(rgb_image)
    cam_path = row['cam_path']
    category_id = row['category_id']
    point_cloud_camera, colors, mask = generate_point_cloud(depth_map, rgb_image, cam_path=cam_path)
    gt_blendedmvs_data[f'image_{i}'] = (point_cloud_camera, colors, rgb_image, depth_map, mask)
    if i==10:
        break

visualize_point(all_point=gt_blendedmvs_data, eval_mode=False)
```

** Figure of examples **

For the blended mvs we have alse the images masked, which mean images containing only the object of interest. as you can see in the example below. Sometimes we have images of walls or containing scenes taking from extreme angles eg top-down view. This kind of images will make the task a little bit harder. We eliminate this type of image by thresholding on the number of black pixel presents in the image. 

##### NUY Depth V2 dataset

NUY Depth V2 is a large dataset that contains indoor scenes. The following presents some scene examples from the data and different views of the constructed 3D point of these scenes.

```python 
import numpy as np
import h5py
import random
f = h5py.File('nyu_depth_v2_labeled.mat','r')
f = h5py.File('nyu_depth_v2_labeled.mat','r')
test_idx = np.load('/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/splits/nyu_depth_test_idx.npy').tolist()
depth =  f.get('depths')
images =  f.get('images')
depth_images = np.array(depth)[test_idx]
images = np.array(images)[test_idx]
gt_nuy_data = {}
all_images = []
for i in range(5):
    depth_map = np.transpose(depth_images[i], axes =(1, 0))
    rgb_image = np.transpose(images[i], axes =(2, 1, 0))
    all_images.append(rgb_image)
    point_cloud_camera, colors, mask = generate_point_cloud(depth_map, rgb_image)
    gt_nuy_data[f'image_{i}'] = (point_cloud_camera, colors, rgb_image, depth_map, mask)

visualize_point(all_point=gt_nuy_data, eval_mode=False)
```

** Figure of examples **

## Approach and Architecure

The objective is to map the image $ I \in\mathbb{R}^{M\times N \times 3}$ to its 3D point cloud $X \in\mathbb{R}^{M\times N \times 3}$ . we can think about a naive or simple autoencoder decoder architecture such as UNET. Where the econder learn the image representation and the decoder map the representation to 3D point cloud. UNET architecture has been tested but it did not give satisfying results.
Our focus will be focus on the camera referential and if we consider the Pinhole camera model, we have the cordinaates X, Y and Z are based on depth map. We have the Z is the depth and X, Y are the the depth map multiplied by coefficient depends on the camera parameters, i.e, the focal length and pricipal camera point and the pixel coordinates. 
The idea is instead to learn a mapping function of the image to the 3D point cloud, we will think of a design that estimates the depth map and the camera parameters.

We will follow an approach proposed by Yin et al [6] that has two stages. The first is a network to estimate the depth map that will be used to reconstruct a 3D point cloud using standard camera parameters. Not using the correct parameters, precisely, the focal length will result in a distorded point cloud even the global shape is preserved. 
The other stage then is composed from two networks each take as input a distorded point cloud and estimate either a focal scale or depth shift to restore the distorded 3D point. The input distorded point cloud are created as follows:
- for the first network, we shift the depth map by a value drawn from a uniform distribution. The shift will result in a distorded shape as it will affect non uniformly, the X, Y, and Z. So, the goal of the first network is to estimate the depth shift value.
- For the second network, the distortion is created by scaling the focal length by a coefficient drawn from a uniform distribution. This scaling will affect only X and Y and will results in points far from each other or closer to each other.

For depth map estimation eigen et al [1] introduced a CNN model levereging on multi scale network.This approach involves training a coarse scale network to predict the depth map at global level which is subsequently refined by a secondary network to refine the local regions. Using a multi-scale architecture proven to be effective, Xian et al [7] proposed a multi scale network that use a feature fusion module to fuse features from the encoder and decoder at different scales to obtain finer prediction. Additionnaly to using a multiscale architecture, in Big to Small model [2], the author proposed, under the assumption of locar planar, a local planar guidance module to guide features to the final depth. Unlike other methods which use only skip connection from encoder stage and upsampling to recover the final depth.

Our design for depth estimation will be then, incorporating the local planar guidance module in the network proposed in [7]
The depth estimation network is a multi scale encoder-decoder network. The encoder consists of a Resnet and outputs the features maps of different scales [1/4, 1/8, 1/16, 1/32].
The decoder begins by applying $3\times3$ up-convolution to the last feature maps of $H/32$ resolution from the Encoder, generating feature maps of $H/16$ resolution. These are then fused with encoder feature maps of the same scale using the feature fusion module. The resulting $H/8$ resolution feature maps are processed through a local planar module to produce the initial coarse depth map. This depth map, along with the $H/8$ feature maps from the decoder and encoder, undergoes another fusion step via the feature fusion module. The process is repeated for resolutions $H/4$ and $H/2$, with the difference at $H/2$ being the replacement of the fusion module with an adaptive module, and no encoder feature maps are used at this stage. The result of the adaptive module is the final depth map.

- **The feature fusion module:**  The module takes as input two feature maps and, if available, a coarse depth map from a previous LPG module. All inputs have the same spatial resolution, $H/k$. The first feature map, from the encoder and contains low-level information, undergoes transformation through a _feature transformation block_. This block transorms the feature representation to the specific task but also adjusts the number of channels. The transformed features are then fused via element-wise summation with the second input feature map, from the preceding fusion module. If a depth map is present, it is concatenated with the aggregated features before being processed by a second _feature transformation block_. Finally, an _upconvolution block_ refines the output for the next module. This _upconvolution block_ consists of an upsampling operation (scaling by a factor of two), a $3\times 3$ convolution, and an ELU activation function.

- **The feature transformation Block** is a residual block that starts by a 3x3 convolution block to adjust the channel number and then a convolution branch with two 3x3 convolution operations, BatchNorm layer and ReLU activation function to learn residual. 

- **the LPG module:**  Given a feature map having a spatial resolution $H/k$, it estimates $4D$ plane coefficient for each spatial cell to reconstruct a coarse depth that fit a locally defined $k\times k$ patch on the full resolution. The LPG uses ray-plane intersection to convert each estimated 4D plane coefficient to $k\times k$ local depth cues on the full resolution:
$$c = \frac{n_{4}}{n_{1}u_{i} + n_{2}v_{i} + n_{3}} \qquad  \text{ where } \quad n = (n_{1}, n_{2}, n_{3}, n_{4}) \text{ where are the estimated 4D parameters}$$
$$\qquad \text{and } (u_{i}, v_{i}) \text{  are $k\times k$ patch wise normalized coordinate of pixel $i$ }$$
The $n$ parameters are the plane parameters where $(n_{1}, n_{2}, n_{3})$ is the normal vector and $n_{4}$ is the distance from the origine to the plane. To estimate these parameters, they use the fact that a normal vector can be computed using two angles, polars and azimuthal, using the the following formulas: (more details in the Appendix)
$$n_{1} = sin(\theta)cos(\phi)$$
$$n_{2} = sin(\theta)sin(\phi)$$
$$n_{3} = cos(\theta)$$
$$n_{4} = d$$ 
To estimates these three parameters at the scale $H/k$, the LPG takes as input the feature of the previous fusion module, i.e feature at scale $H/2k$, and pass them through a series of $1\times1$ convolution to reduce the number of channels by a factor of $2$ until it reaches $3$. So the final convolution layer of the LPG estimates $\theta$, $\phi$ and $d$. More details about the computation are in the implementation of this module.
Thus using the LPG, we have an estimation of depth map at different scales. The lower scales learns the global shapes and the higher scales learns local details. 
The final depth is estimated through a convolution that takes all the depth map at each scale

to estimate the depth shift and focal length shift. As in [6] we have neural network based on PointNet to predict the depth shift or focal length shift given a input of distorted point cloud: 

$L_{depth\ shift} = min_θ |N_d(F(u₀, v₀, f^{*}, d^{*} + ∆^{*}_d), θ) − ∆^{*}_d|$ where $∆^{*}_d$ is drawn from a uniform distribution $\text{Uniform}(-0.25, 0.8)$ during training

$L_{focal\ scale} = min_θ |N_d(F(u₀, v₀, \alpha^{*}f^{*}, d^{*}), θ) − \alpha^{*}|$ where $\alpha^{*}$ is drawn from a uniform distribution $\text{Uniform}(0.6, 1.25)$ during training


The network used for focal scale estimation and depth shift estimation is a PoinNet [9] applied for regression task. The network takes as input a 3D point cloud (B, N, 3) and apply an **input transformation block** followed by a 1D conv layer to extract features. A **feature transformation block** is applied on the extracted and followed by a series of 1D conv operation to result in a vector of global feature using a max aggregation. The max pooling is used to have invariance w.t.r point cloud order.
The  **input transformation block** and  **feature transformation block**  contains a series of conv and linear operations and use relu as activation function and max pooling to aggregates information. The transormation block is used to transform the point cloud to a canonical form to have invariance w.r.t to transformation.

contains feature transformation block, three feature fusion module and adaptive ouput module. The decoder starts by applying the feature transformation block on the last feature maps of ResNet (1/32 scale) and upsampling the feature maps by a factor of 2 to have 1/16 scale. Then it applies three feature fusion modules that each one takes two feature maps of the same scale as input. the First feature maps is an output of ResNet and the second is the output of the previous feature fusion module. 
The idea of feature fusion module is similar to UNET skip connection. Instead of concatenation the encoder block feature maps and decoder block ouput for a given scale, summation is used to fuse feature maps. 

### Data Preparation

The NUY depth dataset and blendedMVS dataset are handled respectively by `NUYDepth` and `BlendedMVSDataset` classes that store samples of images, depth map for training depth map network and point cloud with depth shift or focal scale for PointNet networks. The classes contain both a method 
```python 
def _generate_point_cloud():
``` 
which generate a point cloud using Pinhol camera model. 
The classes take a data containing either the images and depth maps for `NUYDepth` or a dataframe containing the paths, and the train test split and other arguments specifyong if we train for depth map estimation of depth shift of focal scale. 

In both dataset the image and depth map are resized to (256, 256) for computational reasons in the case we are training for depth map estimation. For depth shift or focal scale estimation, we keep the full resolution of the depth map to construct the 3D point cloud, but we use only a sample of 4096 points for for computational reasons. 

The images are normalized using mean and std. For the NUY depth dataset we are using data augmentation, flipping and brightness adjusting as we are using only a subset of 1300 image from the data The original data has 490GB. 
For blendedMVS we are eliminating some objects that corresponds to views of walls or views from bottom to ease the task a little bit.

### Loss and Metrics

The depth estimation, focal length scale estimation or depth shift estimation are regression task. For the loss will be based of L1 or MSE loss 
In depth estimation many losses function have been proposed in the literature such as Huber Loss, silog loss, ordinal regression loss. In our case, we will use scale invariance log which computes the error between the ground truth and the prediction without taking into account the scale discrepency.So, it consider only the relative error between the values.

$L(d^{~}, d) = \frac{1}{n}\sum_{p}{||ln(d^{*}_{p}) - ln(d_{p})||^{2}} - \frac{1}{n^{2}}(\sum_{p}{(ln(d^{*}_{p}) - ln(d_{p}))})^{2}$

for evaluation we consider the following metrics used in the literature: 

- **Accuracy under a threshold** $\delta$ % of $ p  :  \delta = max(\frac{\hat{d}_p}{d_p}, \frac{d_p}{\hat{d}_p}) < threshold $

- **Abs. Rel.:** Mean Absolute Value of the Relative Error. $ \frac{1}{T} \sum_{p \in T} \left| \frac{d_p - \hat{d}_p}{d_p} \right| $ 

#### Implementation

Here the implementation of the models, datasets and training loops. The results discussion are right after the following cells.

```python 
####################### Datasets   #######################

class NUYDepth(Dataset):
    """Custom PyTorch Dataset for NYU Depth Dataset
    
    Creates pairs depth and RGB images for depth estimation
    or shifted point cloud and the value of the focal scale or depth shift 
    for pointNet restoration.
    """
    def __init__(self, data, transform=None, train=True, train_shift=False, train_focal_length=False):
        """
        Args:
            data: A dictionary containing depth and RGB images
            Train: to specify whether the dataset is for training or testing
            train_shift: to train PoinNet for depth shift estimation
            train_focal_length: to train PoinNet for focal scale estimation
        """
        self.train = train
        self.train_shift = train_shift
        self.train_focal_length = train_focal_length
        self.transform = transform
        self.images = data['images']
        self.depth_images = data['depth']
        self.mean = np.array([122.54569849, 104.78370761, 100.02426444])
        self.std = np.array([73.74125527, 75.45537148, 78.87226132])
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        """Get a pair of depth and RGB images
        or shifted point cloud and the value of the focal scale or depth shift 
        for pointNet training.
        
        Returns:
            - image: RGB image, depth_gt: depth image
            - shifted_points: shifted point cloud, depth_shift: depth shift
            - shifted_points: shifted point cloud, focal_scale: focal scale
        """
        
        # Get RGB image and depth map and permute their dimension to valid format 
        # Perfom a cropping as specified in the official website of NYU dataset
        image = np.transpose(self.images[idx], axes =(2, 1, 0))
        image = image[45:471, 41:601, :]
        depth_gt = np.transpose(self.depth_images[idx], axes =(1, 0))       
        depth_gt = depth_gt[45:471, 41:601]
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
    
        depth_gt = np.expand_dims(depth_gt, axis=2) # we add a dimension for the channel --> (H, W, 1)
        
        if self.train_shift:
            # If we train for depth shift estimation, 
            # Generate a random value for the shift 
            depth_shift = np.random.uniform(-0.25, 0.8)
            # Generate a shifted point cloud
            _, shifted_points = self._generate_point_cloud(depth_gt, depth_shift=depth_shift)
            shifted_points = torch.Tensor(shifted_points)
            depth_shift = torch.Tensor([depth_shift])
            return  shifted_points, depth_shift
        if self.train_focal_length:
            # If we train for focal scale estimation, 
            # Generate a random value for the focal scale 
            focal_scale = np.random.uniform(0.8, 1.2)
            # Generate a shifted point cloud
            _, shifted_points = self._generate_point_cloud(depth_gt, focal_scale=focal_scale)
            shifted_points = torch.Tensor(shifted_points)
            focal_scale = torch.Tensor([focal_scale])
            return  shifted_points, focal_scale
         
        if self.train:
            # If we train for depth estimation, we apply a random data augmentation
            # flipping or gamma, brightness, contrast ajustment
            image, depth_gt = self.train_preprocess(image, depth_gt)

        # Resize to 256x256 the depth map and the image for computational reasons
        #if self.train:
        image = (image - self.mean)/self.std
        image = cv2.resize(image, (256, 256))
        depth_gt = cv2.resize(depth_gt, (256, 256)) # We chose interpolation = cv2.INTER_NEAREST for precise value of depth
        image = np.asarray(image, dtype=np.float32)# / 255.0
        depth_gt = depth_gt / 10.
        depth_gt = np.expand_dims(depth_gt, axis=2)
        
        image = torch.Tensor(image)
        depth_gt = torch.Tensor(depth_gt)
        
        return image, depth_gt


    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma
        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness
        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    @staticmethod
    def _generate_point_cloud(depth_map, depth_shift=None, focal_scale=None):
        """Get point cloud using pinhole camera model.
        args:
            depth_map: depth map
            depth_shift: depth shift value if we are training for depth shift estimation
            focal_scale: focal scale if we are training for focal scale estimation
        Returns:
            a sample from shifted point cloud
        """

        # Intrinsic parameters. Those values are defined for all images in the dataset
        fx, fy = 5.8262448167737955e+02, 5.8269103270988637e+02
        cx, cy = 3.1304475870804731e+02, 2.3844389626620386e+02
        
        # Create a grid of pixel coordinates
        H, W = depth_map.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        if depth_shift is not None:
            # Add to depth map a random depth shift if we train for depth shift estimation
            depth_map += depth_shift
        if focal_scale is not None:
            # Multiply to the focal length by a random focal scale if we train for focal scale estimation
            fx *= focal_scale
            fy *= focal_scale
        # Convert to normalized camera coordinates
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # Back-project to 3D camera coordinates
        z_cam = depth_map
        x_cam = z_cam * x_norm
        y_cam = z_cam * y_norm
    
        pcd = np.stack((x_cam, y_cam, z_cam), axis=-1).reshape(-1, 3)
        # Take a sample for training from the point cloud 
        idx = np.random.choice(len(pcd), 4096, replace=False)
        sampled_pcd = pcd[idx]
        return pcd, sampled_pcd


class BlendedMVSDataset(Dataset):
    """Custom PyTorch Dataset for NYU Depth Dataset
    
    Creates pairs depth and RGB images for depth estimation
    or shifted point cloud and the value of the focal scale or depth shift 
    for pointNet restoration.
    """

    def __init__(self, data, train_category_id, test_category_id, train_shift=False, train_focal_length=False, train=False):
        """
        Args:
            data: A dataframe containing depth maps path, RGB images path, camera parameters paths, category id of each image
            train_category_id: a list of category id for training
            test_category_id: a list of category id for testing
            Train: to specify whether the dataset is for training or testing
            train_shift: to train PoinNet for depth shift estimation
            train_focal_length: to train PoinNet for focal scale estimation
        """

        self.root = '/kaggle/input/blendedmvs/BlendedMVS 2/'
        self.train = train
        self.train_shift = train_shift
        self.train_focal_length = train_focal_length
        
        self.data = pl.DataFrame(data) # create polars dataframe. I use polars to filter data easily. (it is faster than pandas and I like it more)
        # split data into train and test by filtering on categery id.
        self.train_data = self.data.filter(pl.col('category_id').is_in(train_category_id)).to_pandas()
        self.test_data = self.data.filter(pl.col('category_id').is_in(test_category_id)).to_pandas()
        
        
    def __len__(self): 
        if self.train:
            return len(self.train_data)
        return len(self.test_data)
        
    def __getitem__(self, idx):
        """Get a pair of depth and RGB images
        or shifted point cloud and the value of the focal scale or depth shift 
        for pointNet training.
        
        Returns:
            - image: RGB image, depth_gt: depth image
            - shifted_points: shifted point cloud, depth_shift: depth shift
            - shifted_points: shifted point cloud, focal_scale: focal scale
        """
         
        if self.train:
            # for training phase
            # read camera file and exract intrinsic and depth information. a camera file is associated to each image.
            cam_path  = self.train_data.loc[idx, 'cam_path']
            with open(cam_path, 'r') as f:
                lines = list(f.read().splitlines())
                #extrinsic_matrix = np.array([l.split() for l in lines[1:5]], dtype=float)
                intrinsic_matrix =np.array([l.split() for l in lines[7:10]], dtype=float)
                depth_info = np.array(lines[11].split(), dtype=float)
            
            # read depth map
            depth_map = cv2.imread(self.train_data.loc[idx, 'depth_map_path'], cv2.IMREAD_UNCHANGED)

            if self.train_shift:
                # If we train for depth shift estimation, 
                # Generate a random value for the shift 
                depth_shift = np.random.uniform(-0.25, 0.8)
                # Generate a shifted point cloud
                _, shifted_points = self._generate_point_cloud(depth_map, intrinsic_matrix, depth_shift=depth_shift)
                shifted_points = torch.Tensor(shifted_points)
                depth_shift = torch.Tensor([depth_shift])
                return  shifted_points, depth_shift
            if self.train_focal_length:
                # If we train for focal scale estimation, 
                # Generate a random value for the focal scale 
                focal_scale = np.random.uniform(0.8, 1.2)
                # Generate a shifted point cloud
                _, shifted_points = self._generate_point_cloud(depth_map, intrinsic_matrix, focal_scale=focal_scale)
                shifted_points = torch.Tensor(shifted_points)
                focal_scale = torch.Tensor([focal_scale])
                return  shifted_points, focal_scale
            
            # read the image
            img = cv2.imread(self.train_data.loc[idx, 'img_path'])[:, :, ::-1]
            
        else:
            # sampe operation as the if block above, but for the test set
            cam_path  = self.test_data.loc[idx, 'cam_path']
            with open(cam_path, 'r') as f:
                lines = list(f.read().splitlines())
                #extrinsic_matrix = np.array([l.split() for l in lines[1:5]], dtype=float)
                intrinsic_matrix =np.array([l.split() for l in lines[7:10]], dtype=float)
                depth_info = np.array(lines[11].split(), dtype=float)
            depth_map = cv2.imread(self.train_data.loc[idx, 'depth_map_path'], cv2.IMREAD_UNCHANGED)

            if self.train_shift:
                depth_shift = np.random.uniform(-0.25, 0.8)
                _, shifted_points = self._generate_point_cloud(depth_map, intrinsic_matrix, depth_shift=depth_shift)
                shifted_points = torch.Tensor(shifted_points)
                depth_shift = torch.Tensor([depth_shift])
                return  shifted_points, depth_shift
            if self.train_focal_length:
                focal_scale = np.random.uniform(0.8, 1.2)
                _, shifted_points = self._generate_point_cloud(depth_map, intrinsic_matrix, focal_scale=focal_scale)
                shifted_points = torch.Tensor(shifted_points)
                focal_scale = torch.Tensor([focal_scale])
                return  shifted_points, focal_scale
            
            img = cv2.imread(self.test_data.loc[idx, 'img_path'])[:, :, ::-1]
        
        # Resize to 256x256 the depth map and the image for computational reasons
        depth_map = cv2.resize(depth_map, (256, 256), interpolation = cv2.INTER_LINEAR)  # We chose interpolation = cv2.INTER_NEAREST for precise value of depth
        #depth_map = depth_map/depth_info[3]
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)
        img = img/255.
        img = torch.Tensor(img)
        depth_map = torch.Tensor(depth_map)
        return img, depth_map
    
    @staticmethod
    def _generate_point_cloud(depth_map, intrinsic_matrix, depth_shift=None, focal_scale=None):
        """Get point cloud from depth map.
        args:
            depth_map: depth map
            intrinsic_matrix: intrinsic matrix containing the focal length, center of the camera.
            depth_shift: depth shift value if we are training for depth shift estimation
            focal_scale: focal scale if we are training for focal scale estimation

        Returns:
            - image: RGB image, depth_gt: depth image
            - shifted_points: shifted point cloud, depth_shift: depth shift
            - shifted_points: shifted point cloud, focal_scale: focal scale
        """


        # Intrinsic parameters
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        # Create a grid of pixel coordinates
        H, W = depth_map.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        if depth_shift is not None:
            depth_map += depth_shift
        if focal_scale is not None:
            fx *= focal_scale
            fy *= focal_scale
        # Convert to normalized camera coordinates
        x_norm = (u - cx/3) / (fx/3)
        y_norm = (v - cy/3) / (fy/3)
        
        # Back-project to 3D camera coordinates
        z_cam = depth_map
        x_cam = z_cam * x_norm
        y_cam = z_cam * y_norm
    
        pcd = np.stack((x_cam, y_cam, z_cam), axis=-1).reshape(-1, 3)
        idx = np.random.choice(len(pcd), 4096, replace=False)
        sampled_pcd = pcd[idx]
        return pcd, sampled_pcd

####################### Model   #######################
class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out
class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels
        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1,
                               bias=True)

        self.conv_branch = nn.Sequential(nn.ReLU(inplace=True),
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True),
                                         nn.BatchNorm2d(num_features=self.mid),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True))
        self.relu = nn.ReLU(inplace=True)

        self._init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.relu(x)

        return x
    def _init_params(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, additional_channels=0, upfactor=2, planar_guidance=False):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor
        self.planar_guidance = planar_guidance

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels + additional_channels, midchannels=self.outchannels)
        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #upconv(self.outchannels, self.outchannels)
        self._init_params()

    def forward(self, low_x, high_x, depth_scaled=None):
        x = self.ftb1(low_x)
        x = x + high_x
        if depth_scaled is not None:
            x = torch.cat([x, depth_scaled], dim=1)
        x = self.ftb2(x)
        x = self.upconv(x)
        return x

    def _init_params(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class local_planar_guidance(nn.Module):
    def __init__(self, num_in_filters, num_out_filters, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)
        self.sigmoid = nn.Sigmoid()
        self.max_depth = 10.
        self.reduc = torch.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False, kernel_size=1, stride=1, padding=0))
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(
                                          nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, bias=False, kernel_size=1, stride=1, padding=0),
                                          nn.ReLU()
                                      )
                                     )

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, x):
        reduc = self.reduc(x)
        theta = self.sigmoid(reduc[:, 0, :, :]) * math.pi / 3
        phi = self.sigmoid(reduc[:, 1, :, :]) * math.pi * 2
        dist = self.sigmoid(reduc[:, 2, :, :]) * self.max_depth
        n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
        n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
        n3 = torch.cos(theta).unsqueeze(1)
        n4 = dist.unsqueeze(1)

        reduc = torch.cat([n1, n2, n3, n4], dim=1)
        plane_normal = reduc[:, :3, :, :]
        plane_normal = F.normalize(plane_normal, 2, 1)
        plane_dist = reduc[:, 3, :, :]
        plane_eq = torch.cat([plane_normal, plane_dist.unsqueeze(1)], 1)
        
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]
        
        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3))#.cuda()
        u = u / self.upratio #(u - (self.upratio - 1) * 0.5) / self.upratio
        
        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio))#.cuda()
        v = v / self.upratio #(v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)
    
class Decoder(nn.Module):
    def __init__(self, planar_guidance=True):
        super(Decoder, self).__init__()
       
        self.outchannels =  1
        self.max_depth = 10.
        self.planar_guidance = planar_guidance
        self.elu = nn.ELU
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        additional_channels = 0
        if self.planar_guidance:
            additional_channels = 1
          
        self.conv = FTB(inchannels=2048, midchannels=512)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True)
        self.upconv = upconv(256, 256) #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #
        self.ffm2 = FFM(inchannels=1024, midchannels=256, outchannels=256, upfactor=2)
        self.ffm1 = FFM(inchannels=512, midchannels=256, outchannels=256, additional_channels=additional_channels, upfactor=2)
        self.ffm0 = FFM(inchannels=256, midchannels=256, outchannels=256, additional_channels=additional_channels, upfactor=2)
        ao_additionnal_channels = 0
        if self.planar_guidance:
            ao_additionnal_channels = 3
        self.out_adapt_block = nn.Sequential(
            nn.Conv2d(in_channels=256+ao_additionnal_channels, out_channels=256 // 2, kernel_size=3, padding=1, stride=1, bias=True), 
            nn.BatchNorm2d(num_features=256 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256 // 2, out_channels=1, kernel_size=3, padding=1, stride=1, bias=True), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        if self.planar_guidance:
            self.lpg8x8     = local_planar_guidance(256, 256, 8)
            self.lpg4x4     = local_planar_guidance(256, 256, 4)
            self.lpg2x2     = local_planar_guidance(256, 256, 2)
            self.get_depth  = nn.Conv2d(4, 1, 3, 1, 1, bias=False)
        
        self._init_params()
    def _init_params(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)  
                    
    def forward(self, features):
        # features' shape: # 1/32, 1/16, 1/8, 1/4
        x_32x = self.conv(features[3])  # 1/32
        x_32 = self.conv1(x_32x)
        x_16 = self.upconv(x_32)  # 1/16
        x_8 = self.ffm2(features[2], x_16)  # 1/8
        if self.planar_guidance:
    
            depth_8x8 = self.lpg8x8(x_8)
            depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.max_depth
            depth_8x8_scaled_ds = F.interpolate(depth_8x8_scaled, scale_factor=1/8, mode='nearest')
            x_4 = self.ffm1(features[1], x_8, depth_8x8_scaled_ds) # 1/4
        else:
            x_4 = self.ffm1(features[1], x_8) # 1/4
        
        if self.planar_guidance:
            depth_4x4 = self.lpg4x4(x_4)
            depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.max_depth
            depth_4x4_scaled_ds = F.interpolate(depth_4x4_scaled, scale_factor=1/4, mode='nearest')
            x_2 = self.ffm0(features[0], x_4, depth_4x4_scaled_ds)  # 1/2
        else:
            x_2 = self.ffm0(features[0], x_4) # 1/2

        if self.planar_guidance:
            depth_2x2 = self.lpg2x2(x_2)
            depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.max_depth
            depth_8x8_scaled_ds = F.interpolate(depth_8x8_scaled, scale_factor=1/2, mode='nearest')
            depth_4x4_scaled_ds = F.interpolate(depth_4x4_scaled, scale_factor=1/2, mode='nearest')
            depth_2x2_scaled_ds = F.interpolate(depth_2x2_scaled, scale_factor=1/2, mode='nearest')
            x_2 = torch.cat([x_2, depth_2x2_scaled_ds, depth_4x4_scaled_ds, depth_8x8_scaled_ds], dim=1)
        
        x = self.out_adapt_block(x_2)
        #x = self.sigmoid(x)
          # original size
        if self.planar_guidance:
            concat1 = torch.cat([x, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
            depth = self.get_depth(concat1)
            return depth, depth_2x2_scaled, depth_2x2_scaled, depth_2x2_scaled
        
        depth = x #self.softplus(x) * self.max_depth
        return depth


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) #nn.BatchNorm2d
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) #nn.BatchNorm2d
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) #nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)  #nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        return features

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = torchvision.models.resnet50(pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


class DepthModel(nn.Module):
    def __init__(self, planar_guidance=True, pretrained=True):
        super(DepthModel, self).__init__()
        self.encoder_modules = resnet50(pretrained=pretrained)
        self.decoder_modules = Decoder(planar_guidance=planar_guidance)

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit = self.decoder_modules(lateral_out)
        return out_logit

```

```python
# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math

from collections import namedtuple
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

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
# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math

from collections import namedtuple
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

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

#### Depth shift and focal scale network

```python
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=1):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if x.shape[0] > 1:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            x = self.fc3(x)
        else: 
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)


class get_model(nn.Module):
    def __init__(self, k=1):
        super(get_model, self).__init__()
       
        channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        if x.shape[0] > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
            x = self.fc3(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.dropout(self.fc2(x)))
            x = self.fc3(x)            
        return x
    
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

### Quantitative Results

```python
train_loss_nuy_depth_adapt, test_loss_nuy_depth_adapt = get_performance(
    perf_log='/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/models_performance/perfomence_nyudepth_bts_depthmap.json',
    model='Adapt on NUY depth')
train_loss_nuy_depth_adapt_lpg, test_loss_nuy_depth_adapt_lpg = get_performance(
    perf_log='/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/models_performance/perfomence_nyudepth_adapt_lpg_depthmap_final.json',
    model='Adapt with LPG on NUY depth')

train_loss_blendedmvs_depth_adapt, test_loss_blendedmvs_depth_adapt = get_performance(
    perf_log='/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/models_performance/perfomence_blendedmvs_adapt_final.json',
    model='Adapt on Blended MVS  depth')

train_loss_blendedmvs_depth_adapt_lpg, test_loss_blendedmvs_depth_adapt_lpg = get_performance(
    perf_log='/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/models_performance/perfomence_ blendedmvs_adapt_with_lpg_final.json',
    model='Adapt with LPG on Blended MVS depth')
```

```python
visualize_loss(train_loss_nuy_depth_adapt, 
                   test_loss_nuy_depth_adapt, 
                   train_loss_nuy_depth_adapt_lpg, 
                   test_loss_nuy_depth_adapt_lpg, 
                   train_loss_blendedmvs_depth_adapt, 
                   test_loss_blendedmvs_depth_adapt, 
                   train_loss_blendedmvs_depth_adapt_lpg, 
                   test_loss_blendedmvs_depth_adapt_lpg)
```

We plot the evolution of the training and test loss for both models on both dataset. Both models converge on both dataset except for the model adapt on blended MVS were the test loss is fluctuating along a single value which does not show any decreasing trend and that could be a sign of the inability of the model on that dataset. This could be due to the fact that the model is to small to generalize on that data. 

The table below resume the evaluation metrics on both dataset and we have the adapt model with the local planar shows slight increase in both metrics accuracy under threshold and relative MAE, for both datasets, comparing to the adapt modal alone. For the accuracy under threshold the highest the better and for relative MAE the lower the better.

|         | Accuracy under threshold      | Relative MAE       |  RMSE       |
|----------------|----------------|----------------|----------------|
| NUY Depth  (Adaptive)| 0.530  | 4.633 | 1.353 |
| NUY Depth  (Adaptive with local planar)|0.013 |1.454| 17.2 |
| Blended MVS (Adaptive)| 0.812  | 5.275  |  22.457 |
| Blended MVS (Adaptive with local planar)| 0.557  | 0.503  |  0.860 |

### Qualitative Results

The pipeline for inference starts by predicting the depth map, on which we apply a sigmoid function to have positive values. The depth map prediction model outputs depth maps with negative values. we Have test using sigmoid, ReLU or Softplus activation function to output the final depth map during training but it led to bad results. Then, we estimate the focal length and the depth shift. To do so, we create a 3D point cloud from a standard focal length and camera optical center. Then we predict refine the focal length by estimating a focal scale from the constructed 3D point. We use the new focal length to construct new 3D point cloud and to refine then the depth by estimating a depth shift. Again use the refined depth to refine another time the focal length. Then we have the final depth and the focal length that we use to generate the final 3D point cloud. 

```python
def refine_focal(depth, focal, model, u0, v0):
    # reconstruct PCD from depth
    pcd, _, _ = generate_point_cloud(depth, cx=u0, cy=v0, fx=focal, fy=focal)
    outputs = model(torch.Tensor(pcd.reshape(-1, 3)).unsqueeze(0).to('cpu').permute((0, 2, 1)))
    return outputs

def refine_shift(depth_wshift, model, focal, u0, v0):
    # reconstruct PCD from depth
    pcd, _, _ = generate_point_cloud(depth_wshift, cx=u0, cy=v0, fx=focal, fy=focal)
    outputs = model(torch.Tensor(pcd.reshape(-1, 3)).unsqueeze(0).to('cpu').permute((0, 2, 1)))
    return outputs

def reconstruct3D_from_depth(pred_depth, shift_model, focal_model):
    cam_u0 = 256 // 2
    cam_v0 = 256 // 2
    proposed_scaled_focal = (256 // 2 / np.tan((60/2.0)*np.pi/180)) # tan(FOV) = A/F  where A : is the (height of the image / 2) and F is the focal length
    # recover focal
    focal_scale_1 = refine_focal(pred_depth, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()
    
    shift_1 = refine_shift(pred_depth, shift_model, predicted_focal_1, cam_u0, cam_v0)
    depth_scale_1 = pred_depth - shift_1.item()
    # recover focal
    #focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, focal_model, u0=cam_u0, v0=cam_v0)
    #predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()
    
    return depth_scale_1, predicted_focal_1

def eval(depth_model_checkpoint='/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/models_performance/nyudepth_bts_depthmap..pth.tar',
          shift_model_checkpoint='/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/models_performance/best_nuy_depth_shift.pth.tar', 
          focal_model_checkpoint='/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/models_performance/best_nuy_focal_scale.pth.tar',
          test_loader=None):
    mean = np.array([122.54569849, 104.78370761, 100.02426444])
    std = np.array([73.74125527, 75.45537148, 78.87226132])
    model = BtsModel()#(pretrained=False, planar_guidance=True)
    model.load_state_dict(torch.load(depth_model_checkpoint, map_location=torch.device('cpu'))['state_dict'])
    model.to('cpu')
    model.eval()

    shift_model = get_model()
    shift_model = torch.nn.DataParallel(shift_model, device_ids=[0])
    shift_model.load_state_dict(torch.load(shift_model_checkpoint, map_location=torch.device('cpu'))['state_dict'])
    shift_model.to('cpu')
    shift_model.eval()

    focal_model = get_model()
    focal_model = torch.nn.DataParallel(focal_model, device_ids=[0])
    focal_model.load_state_dict(torch.load(focal_model_checkpoint, map_location=torch.device('cpu'))['state_dict'])
    focal_model.to('cpu')
    focal_model.eval()

    sigmoid = nn.Sigmoid()
    for i, (images, depth) in enumerate(test_loader):
        images = images.to('cpu').permute((0, 3, 1, 2))
        with torch.no_grad():
            pred_depth = model(images)[0] #sigmoid()
        break
    
    pred_depth = pred_depth.detach().squeeze(1).cpu().numpy()
    images = images.detach().permute((0, 2, 3, 1)).cpu().numpy() 
    
    all_point = {}
    for i in range(5):
        depth_map = torch.Tensor(pred_depth[i]).numpy()
        rgb_image = (images[i]*std + mean).astype(np.uint8)
        depth_scale_1, predicted_focal_2 = reconstruct3D_from_depth(depth_map, shift_model, focal_model)
        print(predicted_focal_2)
        point_cloud_camera, colors, mask = generate_point_cloud(
            depth_map,
            rgb_image= rgb_image, #(255*rgb_image).astype(np.uint8),
            fx=(256 // 2 / np.tan((60/2.0)*np.pi/180)) ,
            fy=(256 // 2 / np.tan((60/2.0)*np.pi/180)) ,
            cx=256/2,
            cy=256/2
            )
        all_point[f'image_{i}'] = (point_cloud_camera, colors, rgb_image, depth_scale_1, mask)
    return all_point
```

**Inference to generate point cloud**

```python
f = h5py.File('nyu_depth_v2_labeled.mat','r')
depth =  f.get('depths')
images =  f.get('images')
depth_images = np.array(depth)
images = np.array(images)
test_idx = np.load('/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/splits/nyu_depth_test_idx.npy').tolist()
test_data = {'images': images[test_idx], 'depth': depth_images[test_idx]}
test_dataset = NUYDepth(test_data, train=False)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
```
```python
visualize_point(pred_point=all_point, gt_point=gt_nuy_data, eval_mode=True)
```

**figure of examples**

Qualitativly, we can see that the approach can capture the global shapes, but the point cloud are very distorded. For example if we compare the left views and top views between group truth and the prediction, we can see easily the huge difference which could be due to the depth prediction.


```python
train_category_id = np.load('/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/splits/train_category_id.npy')
test_category_id = np.load('/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/splits/test_category_id.npy')
view_mode = False
test_dataset = BlendedMVSDataset(data, train_category_id, test_category_id, train=False)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
all_point = eval(depth_model_checkpoint='/Users/ridalefdali/Desktop/Personal Project/Phd_assignement/models_performance/best_blendedmvs_adapt_with_lpg_final.pth.tar',  test_loader=test_loader)
```

```python
visualize_point(pred_point=all_point, gt_point=gt_blendedmvs_data, eval_mode=True)
```