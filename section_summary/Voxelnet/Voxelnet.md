# VoxelNet
在学习了基于点的识别之后，我们来看第二种基于体素的方法，说到这个方向，一定能想起来的就是voxelnet和second等。本来我是打算去看second的详细代码，但是我真的。。。在图像方面感觉没有那么奇怪的框架，但是到了点云方面，我发现好多好多模型都被集成到了哪些框架，反正没人关注，我就直接点名：什么openpcdet、mmdetection3d之类的。我个人认为，这种对于初学者真的不友好，你把东西都集成到了你的框架里，你可能觉着用起来舒服，但是对于我这样的初学者，本来下个代码，然后看看代码，调试一下，就好了。你倒好，不仅要配环境，半天配好了环境，看代码的时候，你还把模型的架构都写成config的结构，那我也不好对模型debug吧，我怎么具体了解模型的内在呢，而且后续我再更改模型，我对框架都不熟悉，怎么改。本来就是很简单的一件事：git代码--->学代码。结果现在整成了:git框架--->学框架--->学代码。大无语。。。
其实这都不算什么，最最让我无语的。这个框架给你多少钱呀，你连自己的仓库都不要了，《废弃》我一整个笑死。我直接放出来让你丢人
@import "1.png"
okk，吐槽完了，直接来看voxelnet吧，好在我发现了一个神器的宝藏，推荐大家优先学习下面的链接。
参考：http://t.csdn.cn/PV9Ae
## data
关于模型的学习呢，我个人感觉就是看两部分：模型的框架，数据处理。除此之外呢，就是论文中可能提到的创新点是如何通过代码实现的。我们手下来看下数据部分。
事情的发展好像不是那么简单，是这样的，这个模型中有一个文件是crop.py主要完成的任务，就是对点云数据进行裁剪。换句话说，摄像头得到的图像都是前视图的，但是点云却是360度的，那么就需要将图片覆盖不到的地方的点云裁剪掉。这一部分是很重要的，同时也是我某一项任务的前提。所以这里我准备细看下。
首先就是数据集的部分，上面文章采用的是mini kitti的数据集。给了链接，但是csdn的下载dddd。但是这能难得到我吗，这不能！
帮大家准备好了，百度网盘自取：链接: https://pan.baidu.com/s/1kv50CiiQS1ijrgpnwpgr8A 提取码: jxgb 
--来自百度网盘超级会员v5的分享（想了想，这个超级会员v5就不删除啦）
首先我们要做的就是读取点云和图片的信息。
```ruby
def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    #points = points[:, :3]  # exclude luminance
    return points
```
第二部分就是对标定参数的读取，首先我们来简单了解下kitti数据集，主要分为四个方面：激光点云、标定数据、图像数据、标定文件。
激光点云是以.bin存储的，其中，激光雷达坐标系是z为高度方向，x为汽车前进方向，y为前进左手边方向，满足右手定则。二进制存储，每个点包含四个数据xyz以及反射强度。bin文件可以使用np读取。
```ruby
np.formfile('0000.bin',dtype = np.float32).reshape(-1,4)
```
标定信息使用txt进行存储，标定信息我们能看到的一共有15列，第一列是类别信息；第二列是截断程度，这是一个0到1的值，描述的是障碍物在视野中的截断程度，0表示没有截断；第三列是遮挡程度，取值为（0，1，2，3）0代表完全可见。第四列是观测角度，也就是将物体旋转到正前方，其与车身方向的夹角；第五到八列是二维检测框的左上xy右下xy；第九到十一列对应三维物体的高度宽度长度，单位是米；第12到14列对应的是物体在相机坐标系下的中心坐标xyz；第十五列为旋转角，（-π，π），表示车体朝向，即物体前进方向与相机坐标系x轴的夹角。
图像数据简单的就不用说了，就是image
最后就是标定数据，也是用txt文件存储的，一般是七行，前面的P0到P3表示的是相机的内参，因为有四个相机，所以这里有四个。后面的R0_rect是0号相机的校准矩阵，目的是为了使4个相机成像达到共同的效果，保证4个相机光心在一个xoy面上，这个参数的使用，一般是点云经过外参变化后，需要乘这个矫正矩阵得到相机坐标系下的坐标。后面的Tr_velo_to_cam、Tr_imu_to_velo，顾名思义分别是从激光雷达到相机和imu到相机的外参矩阵。
了解到这些之后，我们直接来看代码
```ruby
def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]
    #
    P = np.array(lines[CAM]).reshape(3,4)
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
    Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3,:3] = np.array(lines[4][:9]).reshape(3,3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect
```
应该还是很清晰的，这整个过程需要使用到相机的内参，激光雷达到相机的外参，以及一个矫正矩阵（虽然我自己的项目中没有整个参数，感觉不是必要的）。
了解这些后，我们就来看下主函数,并按照步骤一步步走。
```ruby
IMG_ROOT = '/media/billy/New Volume/KITTI/testing/image_2/'
PC_ROOT = '/media/billy/New Volume/KITTI/testing/velodyne/'
CALIB_ROOT = '/media/billy/New Volume/KITTI/testing/calib/'
PC_CROP_ROOT = '/media/billy/New Volume/KITTI/testing/crop/'
for frame in range(0, 7518):
    img_dir = IMG_ROOT + '%06d.png' % frame
    pc_dir = PC_ROOT + '%06d.bin' % frame
    calib_dir = CALIB_ROOT + '%06d.txt' % frame
    points = align_img_and_pc(img_dir, pc_dir, calib_dir)
    output_name = PC_CROP_ROOT + '%06d.bin' % frame
    print('Save to %s' % output_name)
    points[:,:4].astype('float32').tofile(output_name)
```
首先是一堆目录，用来读取图片点云标定文件以及最后保存的文件夹。看到调用```align_img_and_pc```函数，之后得到points之后，就直接保存啦。所以来看那个函数
```ruby
def align_img_and_pc(img_dir, pc_dir, calib_dir):
    
    img = imread(img_dir)
    pts = load_velodyne_points( pc_dir )
    P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir)

    pts3d, indices = prepare_velo_points(pts)
    pts3d_ori = pts3d.copy()
    reflectances = pts[indices, 3]
    pts3d, pts2d_normed, idx = project_velo_points_in_img( pts3d, Tr_velo_to_cam, R_cam_to_rect, P  )
    #print reflectances.shape, idx.shape
    reflectances = reflectances[idx]
    #print reflectances.shape, pts3d.shape, pts2d_normed.shape
    assert reflectances.shape[0] == pts3d.shape[1] == pts2d_normed.shape[1]

    rows, cols = img.shape[:2]

    points = []
    for i in range(pts2d_normed.shape[1]):
        c = int(np.round(pts2d_normed[0,i]))
        r = int(np.round(pts2d_normed[1,i]))
        if c < cols and r < rows and r > 0 and c > 0:
            color = img[r, c, :]
            point = [ pts3d[0,i], pts3d[1,i], pts3d[2,i], reflectances[i], color[0], color[1], color[2], pts2d_normed[0,i], pts2d_normed[1,i]  ]
            points.append(point)

    points = np.array(points)
    return points
```
这个函数的前三行，没的说，图像点云标定文件的读取。我们看第四行的函数```prepare_velo_points```。首先输入是刚才读取到的点云的数据，我debug了一下，点云读进来是array的形式，shape是(115384,4)。然后做一个简单的滤除，这里是根据反射强度进行滤除，这是利用掩码的形式，也是我们经常使用的方法。后面呢，就是将点云的第四个维度全部置为1，也就是反射强度置为1。实际上这里的反射强度没什么用，甚至可以把那一个维度去掉。最后return将dim0和1进行交换。
```ruby
def prepare_velo_points(pts3d_raw):
    '''Replaces the reflectance value by 1, and tranposes the array, so
        points can be directly multiplied by the camera projection matrix'''
    pdb.set_trace()
    pts3d = pts3d_raw
    # Reflectance > 0
    indices = pts3d[:, 3] > 0
    pts3d = pts3d[indices ,:]
    pts3d[:,3] = 1
    return pts3d.transpose(), indices
```
然后呢，后面主函数写的很迷，就是得到pts3d之后，又做了一次？上面的函数得到了滤除之后的结果和掩码```pts3d, indices = prepare_velo_points(pts)```嗷嗷，我懂了，后面实际上是对反射强度又做了一次掩码，将反射强度大于0的点的反射强度单独提取出来。
```ruby
    reflectances = pts[indices, 3]
```
之后，就进到了最后一个函数中！这个函数的输入包括三个标定的参数和之前滤除反射强度之后的点云。
```ruby
pts3d, pts2d_normed, idx = project_velo_points_in_img( pts3d, Tr_velo_to_cam, R_cam_to_rect, P  )
def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
        numpy array. Returns the 2D projection of the points that
        are in front of the camera only an the corresponding 3D points.'''
    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2,:]>=0)
    pts2d_cam = Prect.dot(pts3d_cam[:,idx])
    return pts3d[:, idx], pts2d_cam/pts2d_cam[2,:], idx
```
卧槽，这函数这么短（内心窃喜）。第一行，就是简单的矩阵相乘，我们可以看下debug过程中，各个数组的shape。
```ruby
(Pdb) pts3d.shape
(4, 103329)
(Pdb) T_cam_velo.shape
(4, 4)
(Pdb) Rrect.shape
(4, 4)
```
这里就将点云转换到了相机坐标系。然后这个输出结果再做一次掩码，这回是取z大于0的点，这里其实基本已经可以理解了，因为我们已经知道了在相机坐标系下，z轴的方向是自车行驶的纵向方向，所以z>0也就是自车前方的所有点。
之后用在用相机的内参乘这些点，得到的就是像素坐标系的点把。
最后这里return了三个变量```return pts3d[:, idx], pts2d_cam/pts2d_cam[2,:], idx```首先第一个就是滤除之后的点在激光雷达下的坐标，第二个参数实际上应该是点云点经过转换后在像素坐标系的位置，为什么要除以```pts2d_cam[2,:]```首先在像素坐标系肯定是xy，没啥问题的。但是我们在计算矩阵计算完之后，得到的array的shape为[3,53817],也就是这里的第三个维度应该是1的，所以最终就会这样做。第三个呢就是idx也就是得到z方向大于0的掩码。
这里就将反射强度再次做了一次掩码。
```ruby
    reflectances = reflectances[idx]
```
接下来就是涉及图像的部分了，首先我们肯定要确定image的宽高
```ruby
    rows, cols = img.shape[:2]
```
后面呢，就是对初步滤除后转换到像素坐标系的点进行遍历，然后分别取出xy坐标值，这里的判断就是简单的要求这个坐标在不在图片像素大小里。接下来呢，提取了对应像素点的颜色信息？为什么要这么做呢，不知道。反正最终是将所有信息都放在了一个列表里。
```
point = [ pts3d[0,i], pts3d[1,i], pts3d[2,i], reflectances[i], color[0], color[1], color[2], pts2d_normed[0,i], pts2d_normed[1,i]  ]
```
也就是[x,y,z,反射强度,R,G,B,计算后的像素x,y]这里为什么要放计算后的像素xy呢，因为我们在计算完之后，得到的数字基本上就是小数，但是到像素坐标系下，坐标肯定只能是整数呀。最终我们得到points转成array的形式，return出来。维度是(点数，9)
最后保存，也可以看出来就是对points[:,:4]保存成了点云数据，这里就只有xyz和反射强度。
#### 本来今天说看Voxelnet的，哈哈哈哈，结果整起了坐标系转换。我裂开。明天继续。
首先我做的就是去跑一下代码，让他能够训练起来，然后我去debug，但是很奇怪，这个模型需要那么大的显存吗？我使用的是默认的设置，一些参数都没有改，但是还是会炸显存，主要是batchsize调到1也会炸。这咋整！于是我只好暂时的放弃了，去看了下dataset，其中有一部分真的又臭又长，好难看呀。
但是整体的代码流程基本上明白了，也就是这个体素是怎么划分的，如何进行表达的。每个体素的大小，是代码一开始就设定好的。于是我就找到了一个参数，一个叫做self.T的参数，这个参数是控制每个体素内的点的个数，默认设置是35，也就是如果这个体素内的点超过35，那就缩减到35个。我将这个参数调了很小 ，然后显存明显就够用了。然后就开始我愉快的debug过程。
首先我们还是看下dataset,就简单的看下getitem
```ruby
    def __getitem__(self, i):
        # import pdb;
        # pdb.set_trace()
        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        image_file = self.image_path + '/' + self.file_list[i] + '.png'
        calib = utils.load_kitti_calib(calib_file)#这里加载标定矩阵
        Tr = calib['Tr_velo2cam']
        gt_box3d = utils.load_kitti_label(label_file, Tr)
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)#得到激光雷达点云的xyz
        if self.type == 'velodyne_train':
            image = cv2.imread(image_file)
            # data augmentation
            lidar, gt_box3d = aug_data(lidar, gt_box3d)
            # pdb.set_trace()
            # specify a range
            lidar, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d)

            # voxelize
            voxel_features, voxel_coords = self.preprocess(lidar)
            # pdb.set_trace()
            # bounding-box encoding
            pos_equal_one, neg_equal_one, targets = self.cal_target(gt_box3d)

            return voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, image, calib, self.file_list[i]
```
这里的重点主要在后面， 前面的部分都是很简单的，主要就是确定一些文件的读取路径，然后提取文件中的东西，主要包括：激光雷达数据、标定数据、标签数据、图片数据。这里有一点是要说明的，就是要知道kitti数据集中的label都是在相机坐标系下标注的，因此，我们在做点云的检测时，需要将这些标签读取，然后利用标定数据进行坐标系转换。
接下来，就是数据增强。
可以简单看下数据增强的部分，按照概率分为了三个部分：部分旋转、全局旋转、缩放。
```ruby
def aug_data(lidar, gt_box3d_corner):
    np.random.seed()
    # pdb.set_trace()
    choice = np.random.randint(1, 10)
    # choice = 8
    if choice >= 7:
        for idx in range(len(gt_box3d_corner)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()
                # check collision
                tmp = box_transform(
                    gt_box3d_corner[[idx]], t_x, t_y, t_z, t_rz)
                is_collision = False
                for idy in range(idx):
                    iou = cal_iou2d(tmp[0,:4,:2],gt_box3d_corner[idy,:4,:2])
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = gt_box3d_corner[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                gt_box3d_corner[idx] = box_transform(
                    gt_box3d_corner[[idx]], t_x, t_y, t_z, t_rz)

        gt_box3d = gt_box3d_corner
    elif choice < 7 and choice >= 4:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        gt_box3d = box_transform(gt_box3d_corner, 0, 0, 0, r=angle)
    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        gt_box3d = gt_box3d_corner * factor
    return lidar, gt_box3d
```
可以看到利用随机数choice将数据增强分为成三个方式，第一个就是局部的旋转，也就是根据box3d选择出在框内的点云，然后进行旋转，当然label的框也是会旋转的。第二种是对全部的点云进行旋转。第三种是对点云进行缩放，也就是让点云的xyz扩大一定的倍数即可。这里没什么很难的地方，只不过有些繁琐，同时需要对旋转矩阵有些了解。这里后续还需要继续跟进。
下一个步骤就是对点云进行过滤，你知道的kitti数据集中的label都是在相机坐标系，且只标注了相机可见范围的目标，但是激光雷达的点云会是360度，这就意味着会有很多没用的点，所以这里需要过滤一下。作者是一开始在config中配置了，体素划分的边界，然后只需要过滤掉边界外的点云即可。
```ruby
def get_filtered_lidar(lidar, boxes3d=None):
    # pdb.set_trace()
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]

    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)#这个函数用来求两个数组的交集
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= cfg.xrange[0]) & (boxes3d[:, :, 0] < cfg.xrange[1])
        box_y = (boxes3d[:, :, 1] >= cfg.yrange[0]) & (boxes3d[:, :, 1] < cfg.yrange[1])
        box_z = (boxes3d[:, :, 2] >= cfg.zrange[0]) & (boxes3d[:, :, 2] < cfg.zrange[1])
        box_xyz = np.sum(box_x & box_y & box_z,axis=1)

        return lidar[filter_xyz], boxes3d[box_xyz>0]

    return lidar[filter_xyz]
```
接下来就是对点云的体素划分
```ruby
    def preprocess(self, lidar):
        # pdb.set_trace()
        # shuffling the points
        np.random.shuffle(lidar)
        voxel_coords = ((lidar[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) / (
                        self.vw, self.vh, self.vd)).astype(np.int32)
        # convert to  (D, H, W)
        voxel_coords = voxel_coords[:,[2,1,0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, \
                                                  return_inverse=True, return_counts=True)
        #去除重复的元素，并排序
        #return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式存储。==原始点在新列表的位置==
        #return_counts：如果为 true，返回去重数组中的元素在原数组中的出现次数。
        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = lidar[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[:self.T, :]
                voxel_counts[i] = self.T
            # augment the points
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)
        return np.array(voxel_features), voxel_coords
```
可以看出来，voxel_coords就是划分的体素。
其实后面我还看了网络模型的处理，只能说很蒙，所以这里就暂时写到这里啦。
所以这里就只能参考网上别人写的文章啦。参考可以在csdn搜一下。因为实际上voxelnet和pointpillar的模型结构包括其他地方有很多都是相同的，所以我准备这里就先做初步了解，然后pointpillar会做超级详细的讲解。