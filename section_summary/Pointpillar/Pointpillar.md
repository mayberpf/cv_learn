# pointpillar + GDB
首先我下载的代码是：https://github.com/zhulf0804/PointPillars
具体的操作步骤可以看代码中的readme，为了更快的代码实现，并且减小内存的压力，因此这里我使用的是KITTI-mini的数据集。
整体的理论我们在看完代码之后做一个总结。
在准备好数据集之后，需要先做一个预处理
```ruby
cd PointPillars/
python pre_process_kitti.py --data_root your_path_to_kitti
```
### pre_process_kitti
这一部分应该是所有利用kitti数据集进行操作都需要做的事情。
这里我们首先简单看下预处理，这里对数据进行操作，然后生成了一些新的东西，我们先来简单看下这里生成了什么，首先是一堆pkl文件，根据参考的别人的文章，这里的pkl文件主要是将imageset中定义的用于训练验证和测试的文件列表进行进一个的细化。因为在imageset中txt文件中只是存储了文件名。而pkll中利用字典的形式存放每个样本的详细信息：图片数据信息、激光雷达信息、标定校准信息、标签信息。这里先做一个简单了解，后面再调试代码的时候，会详细展开看看里面有什么。
其次，还生成了一个velodyne_reduced文件夹，这个文件夹中存放的是所有经过裁剪的点云，因为标签只有相机可见范围内，所以将相机以外的点云做了裁剪。
然后就是kitti_gt_database 文件夹，该文件夹下存储的是每个目标所占的点云，并且将点云的中心点移动到了坐标原点。存储文件的命名方法为：样本序号_类别_第i个目标。
接下来详细的看下代码。首先对数据预处理，当然是要做一件事，那就是确定好data的目录。这里在参数中需要确定。
```ruby
    parser.add_argument('--data_root', default='/home/ktd/rpf_ws/PointPillars/kitti', 
                        help='your data root for kitti')
```
其余的参数都无伤大雅，简单改改就好，接下来就到main函数。在main函数中最主要的函数就一个```create_data_info_pkl```。然后简单看下main中，主要是对训练集、验证集、测试机创建pkl文件。所以我们主要来看下那个函数，主要做了什么。
```ruby
def create_data_info_pkl(data_root, data_type, prefix, label=True, db=False):
    sep = os.path.sep
    print(f"Processing {data_type} data..")
    ids_file = os.path.join(CUR, 'dataset', 'ImageSets', f'{data_type}.txt')#这里需要读取数据的文件名
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()]
    split = 'training' if label else 'testing'
```
前面部分，说实话，只是简单的对数据地址进行一个读取和存储，首先我们要知道pkl中的东西实际上就是一个字典，也就是会存储各种东西。这样吧，我们将一些简单的零散的说明写在代码中注释出来。
```ruby
    kitti_infos_dict = {}
    if db:
        kitti_dbinfos_train = {}
        db_points_saved_path = os.path.join(data_root, f'{prefix}_gt_database')
        os.makedirs(db_points_saved_path, exist_ok=True)
```
接下来这一部分就是将文件的一些内容填到创建的字典中暂时看是包括了图片的地址、点云的地址、标定参数等
```ruby
    for id in tqdm(ids):
        cur_info_dict={}
        img_path = os.path.join(data_root, split, 'image_2', f'{id}.png')
        lidar_path = os.path.join(data_root, split, 'velodyne', f'{id}.bin')
        calib_path = os.path.join(data_root, split, 'calib', f'{id}.txt') 
        cur_info_dict['velodyne_path'] = sep.join(lidar_path.split(sep)[-3:])
```
接下来读取图片，主要是为了获取图片的大小，图片地址，图片id等
```ruby
        img = cv2.imread(img_path)
        image_shape = img.shape[:2]
        cur_info_dict['image'] = {
            'image_shape': image_shape,
            'image_path': sep.join(img_path.split(sep)[-3:]), 
            'image_idx': int(id),
        }
```
然后是读取标定数据，并存储到字典中。
```ruby
        calib_dict = read_calib(calib_path)
        cur_info_dict['calib'] = calib_dict
```
接下来读取雷达的数据，因为在预处理这一部分，主要是针对雷达点云进行操作的，所以下面将会是重点。
```ruby
        lidar_points = read_points(lidar_path)
        reduced_lidar_points = remove_outside_points(
            points=lidar_points, 
            r0_rect=calib_dict['R0_rect'], 
            tr_velo_to_cam=calib_dict['Tr_velo_to_cam'], 
            P2=calib_dict['P2'], 
            image_shape=image_shape)
        saved_reduced_path = os.path.join(data_root, split, 'velodyne_reduced')
        os.makedirs(saved_reduced_path, exist_ok=True)
        saved_reduced_points_name = os.path.join(saved_reduced_path, f'{id}.bin')
        write_points(reduced_lidar_points, saved_reduced_points_name)
```
首先激光雷达点云第一个主要的处理函数就是```remove_outside_points```顾名思义就是将相机视野之外的点云进行滤除。这里就不再详细说了，因为之前在写voxelnet时，写过这一部分。滤除之后，将点云保存到velodyne_reduced的文件夹下。
```ruby
        if label:
            label_path = os.path.join(data_root, split, 'label_2', f'{id}.txt')
            annotation_dict = read_label(label_path)
            annotation_dict['difficulty'] = judge_difficulty(annotation_dict)
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
                points=reduced_lidar_points,
                r0_rect=calib_dict['R0_rect'], 
                tr_velo_to_cam=calib_dict['Tr_velo_to_cam'],
                dimensions=annotation_dict['dimensions'],
                location=annotation_dict['location'],
                rotation_y=annotation_dict['rotation_y'],
                name=annotation_dict['name'])
            cur_info_dict['annos'] = annotation_dict
```
这里针对标签数据进行读取，这里有一个不是很重要的函数```judge_difficulty```，这个函数的主要功能就是判断标签文件中的标签识别的困难程度。然后重要的函数是```get_points_num_in_bbox```，这个函数就比较复杂啦。包括这个函数和之后的函数，最主要的就是里面的一个函数```points_in_bboxes_v2```。
```ruby
            if db:
                indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
                    points_in_bboxes_v2(
                        points=lidar_points,
                        r0_rect=calib_dict['R0_rect'].astype(np.float32), 
                        tr_velo_to_cam=calib_dict['Tr_velo_to_cam'].astype(np.float32),
                        dimensions=annotation_dict['dimensions'].astype(np.float32),
                        location=annotation_dict['location'].astype(np.float32),
                        rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                        name=annotation_dict['name']    
                    )
```
这里最主要的就是point_in_bboxes_v2这个函数。我们主要去看下那个函数干了什么。
```ruby
def points_in_bboxes_v2(points, r0_rect, tr_velo_to_cam, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4) 
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3) 
    location: shape=(n, 3) 
    rotation_y: shape=(n, ) 
    name: shape=(n, )
    return:
        indices: shape=(N, n_valid_bbox), indices[i, j] denotes whether point i is in bbox j. 
        n_total_bbox: int. 
        n_valid_bbox: int, not including 'DontCare' 
        bboxes_lidar: shape=(n_valid_bbox, 7) 
        name: shape=(n_valid_bbox, )
    '''
    pdb.set_trace()
    n_total_bbox = len(dimensions)
    n_valid_bbox = len([item for item in name if item != 'DontCare'])
    location, dimensions = location[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotation_y, name = rotation_y[:n_valid_bbox], name[:n_valid_bbox]
    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1)
    bboxes_lidar = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)#对location的xyz进行转换

    bboxes_corners = bbox3d2corners(bboxes_lidar)#z这里得到的是八个角点xyz
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    pdb.set_trace()
    indices = points_in_bboxes(points[:, :3], frustum_surfaces) # (N, n), N is points num, n is bboxes number
    return indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name
```
首先我们可以看下注释，在注释中阐述了每个输入变量和输出变量的shape，然后分析下输入变量都代表什么    points: shape=(N, 4) 点云的数组、tr_velo_to_cam: shape=(4, 4)激光雷达到相机的旋转平移矩阵、r0_rect: shape=(4, 4)相机的内参矩阵、dimensions: shape=(n, 3)标签数据的bbox的长宽高、location: shape=(n, 3)bboxes的中心的xyz 、rotation_y: shape=(n, )bboxes的旋转角度 、name: shape=(n, )bboxes的标签名。
然后根据bboxes的数量对数据进行操作。接下来呢，因为标签是在相机坐标系下的数据，所以这里需要一个步骤就是将2D--->3D，转换到激光雷达坐标系下，这里肯定是对location进行操作。
之后进行一个group的划分，其实不是很理解，所以这里做详细一点。首先第一个函数bbox3d2corners，是根据bbox的中心点和长宽高得到3D框的八个角点的xyz。因为还有一个旋转角，所以这里的代码写的比较复杂。接下来是group_rectangle_vertexs，这里建议看下角点的转换，就是每个点对应的哪个角，方便后面理解。group_rectangle_vertexs这个函数是将八个角点再次拆分成6个面，然后拼接在一起，得到输出维度就是(n,6,4,3),后面的一个函数就更看不懂了，group_plane_equation，首先求出每个面上的2两个向量，然后做叉乘，就得到了垂直于该面的向量。然后用了一个np.einsum('ijk,ijk->ij')这确实理解不了了，反正最后输出的维度肯定是(n,6),然后在将这个和法向量拼接。最终输出(n,6,4)
接下来进到points_in_bboxes中，看名字感觉这个就是找到位于bboxes内的所有点。看代码最终的输出可以知道，得到的是一个mask，也就是掩码。这里的第四个量我看不太懂，但是前三个就是每个面的法向量呗，然后就计算每个点是在这个面的里面还是外面，外面的话，就让其等于False。说白了就是一个过滤点是否在bboxes里面的掩码。
到这里，整体points_in_bboxes_v2函数就没了。继续下面的预处理。
```ruby
                pdb.set_trace()
                for j in range(n_valid_bbox):
                    db_points = lidar_points[indices[:, j]]
                    db_points[:, :3] -= bboxes_lidar[j, :3]
                    db_points_saved_name = os.path.join(db_points_saved_path, f'{int(id)}_{name[j]}_{j}.bin')
                    write_points(db_points, db_points_saved_name)
```
这里就是根据上面得到的掩码，得到bboxes包括的点云，再分别保存，到文件夹中。
```ruby
                    db_info={
                        'name': name[j],
                        'path': os.path.join(os.path.basename(db_points_saved_path), f'{int(id)}_{name[j]}_{j}.bin'),
                        'box3d_lidar': bboxes_lidar[j],
                        'difficulty': annotation_dict['difficulty'][j], 
                        'num_points_in_gt': len(db_points), 
                    }
                    if name[j] not in kitti_dbinfos_train:
                        kitti_dbinfos_train[name[j]] = [db_info]
                    else:
                        kitti_dbinfos_train[name[j]].append(db_info)
    
        kitti_infos_dict[int(id)] = cur_info_dict
    saved_path = os.path.join(data_root, f'{prefix}_infos_{data_type}.pkl')
    write_pickle(kitti_infos_dict, saved_path)
    if db:
        saved_db_path = os.path.join(data_root, f'{prefix}_dbinfos_train.pkl')
        write_pickle(kitti_dbinfos_train, saved_db_path)
    return kitti_infos_dict
```
到这里基本就结束了，最后这里无非就是存储的一些命名在pkl中，所以接下来我们看这个预处理产生的几个pkl分别都包括什么就好了！
首先看下训练的pkl，叫做kitti_infos_train.pkl，这个是按照训练文件的序号进行排列的。接下来展示的只是一个文件的内容。
```ruby
{'velodyne_path': 'training/velodyne/000000.bin', 'image': {'image_shape': (370, 1224), 'image_path': 'training/image_2/000000.png', 'image_idx': 0}, 'calib': {'P0': array(), 'P1': array(), 'P2': array(), 'P3': array(), 'R0_rect': array(), 'Tr_velo_to_cam': array(), 'Tr_imu_to_velo': array()}, 'annos': {'name': array(['Pedestrian'], dtype='<U10'), 'truncated': array([0.]), 'occluded': array([0]), 'alpha': array([-0.2]), 'bbox': array([[712.4 , 143.  , 810.73, 307.92]]), 'dimensions': array([[1.2 , 1.89, 0.48]]), 'location': array([[1.84, 1.47, 8.41]]), 'rotation_y': array([0.01]), 'difficulty': array([0]), 'num_points_in_gt': array([377])}}
```
其次是kitti_dbinfos_train.pkl，这个文件是按照标签的类别进行排列的，接下来就展示一个行人的内容，这里应该是所有数据中的行人的bbox过滤出来的
```ruby
(Pdb) data['Pedestrian']
[{'name': 'Pedestrian', 'path': 'kitti_gt_database/0_Pedestrian_0.bin', 'box3d_lidar': array([ 8.731381 , -1.8559176, -1.5996994,  0.48     ,  1.2      ,
        1.89     ,  0.01     ], dtype=float32), 'difficulty': 0, 'num_points_in_gt': 377}, {'name': 'Pedestrian', 'path': 'kitti_gt_database/5_Pedestrian_0.bin', 'box3d_lidar': array([23.311281 ,  8.52229  , -1.8116788,  0.96     ,  0.65     ,
        1.87     ,  1.59     ], dtype=float32), 'difficulty': 0, 'num_points_in_gt': 70}, {'name': 'Pedestrian', 'path': 'kitti_gt_database/10_Pedestrian_2.bin', 'box3d_lidar': array([23.799772 , -8.312204 , -1.4643983,  0.72     ,  1.09     ,
        1.96     ,  1.75     ], dtype=float32), 'difficulty': 2, 'num_points_in_gt': 23}]
```
其余产生的pkl文件都和kitti_infos_train.pkl差不多，只不过包括了训练集、验证集、测试集等。
### 到这里只是一小部分！！！
后面我们将详细阐述dataset及网络模型，这个体素是怎么做的。至于其他的代码比如像优化器，学习率等等，就不再赘述了。
### dataset
这一部分是在训练过程中的dataset
整体看下来，dataset实际上就是一个简单的读取文件，其主要内容就是数据增强的地方。
```ruby
def __getitem__(self, index):
    data_info = self.data_infos[self.sorted_ids[index]]
    image_info, calib_info, annos_info = \
        data_info['image'], data_info['calib'], data_info['annos']
    # point cloud input
    velodyne_path = data_info['velodyne_path'].replace('velodyne', self.pts_prefix)
    pts_path = os.path.join(self.data_root, velodyne_path)
    pts = read_points(pts_path)
```
到这里，都是读取文件，后面就是由于bboxes是相对相机坐标系做的，所以这里还需要利用标定的矩阵，将坐标系转换到激光雷达坐标系。所以下面需要读取标定的矩阵。
```ruby
    tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
    r0_rect = calib_info['R0_rect'].astype(np.float32)
    # pdb.set_trace()
    # annotations input
```
这里有一个self.remove_dont_care函数，看名字，就能理解，因为在label中，会有一个类别名叫做don't care，所以这里将这些类别删除。
```ruby
    annos_info = self.remove_dont_care(annos_info)#在数据中可能存在dont care，所以这里将其remove掉
```
后面这里就是对数据进行读取，主要包括对标签数据，包括类别名字、中心点、Bboxes长宽高、旋转角，然后将这些拼接为一个数组，这样得到的数据类型就是(bboxes个数，7)。
```ruby
    annos_name = annos_info['name']
    annos_location = annos_info['location']
    annos_dimension = annos_info['dimensions']
    rotation_y = annos_info['rotation_y']
    gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)#相机坐标系下的bboxes
    gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
```
后面做了一件事，因为在做目标检测的时候，我们预测的类别只能数字，所以这里需要将类别对应数字，这是在dataset类的一开始就定义好的。但是并不是所有的类别都有index，所以不包括的就用-1代替了。这部分操作，就类似目标检测的onehot。
```ruby
    gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]#这里的self.CLASSES不一定包含所有的类别，不包含的话就用-1
```
到这里基本dataset就结束了，也就是整体的数据读取都搞定了，将这里数据都存放在一个字典里面。最后就是数据增强的地方啦！这很复杂，这里先简单的说下数据增强包括：1、将之前提取的bboxes内的点云，和数据点云进行合并2、物体噪声3、随机翻转4、全局旋转缩放平移5、点的随机滤除6、物体的随机滤除7、点的打乱。详细的说明看看今天晚上有没有时间把。
```ruby
    data_dict = {
        'pts': pts,
        'gt_bboxes_3d': gt_bboxes_3d,
        'gt_labels': np.array(gt_labels), 
        'gt_names': annos_name,
        'difficulty': annos_info['difficulty'],
        'image_info': image_info,
        'calib_info': calib_info
    }
    if self.split in ['train', 'trainval']:
        data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)
    else:
        data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])
    return data_dict
```
### net
在看了上面两部分，都是数据的处理，发现一个问题，就是代码中对体素的处理，并不是在数据处理上进行的，包括Voxelnet，也是这样，体素的划分，都是在网络中进行的。所以这部分当然需要认真看下。
首先我们看训练的代码，了解到模型的输入，batched_pts、batched_gt_bboxes、batched_labels。但是这里看了下数据类型，是的，这是个列表，之后的所有处理都在这里模型中！于是接下来我们就简单看下。
```ruby
bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
    pointpillars(batched_pts=batched_pts, 
        mode='train',
        batched_gt_bboxes=batched_gt_bboxes, 
        batched_gt_labels=batched_labels)
```
接下来进到模型的forward中，首先就是第一个函数，self.pillar_layer，这一层就是模型对点进行处理，最终得到体素的转换。但是当我看到这一部分的时候，我发现这些函数在套娃，一个套一个，最终进到一个函数中！最终进到这里！
```ruby
voxel_num = hard_voxelize(points, voxels, coors,
    num_points_per_voxel, voxel_size,
    coors_range, max_points, max_voxels, 3,
    deterministic)
```
你以为只要进入这个函数就可以啦，但是我告诉你，这个函数进不去！但是可以运行，我也不太清楚为什么，根据我的初步研究，实际上这个函数是写在了cpp中，因为我们在拿到代码仓库的时候，做了一个setup.py的安装，同时我也找到了那个cpp文件，然后我就看到了一个我比较熟悉的文件，一个.cu文件，之前在学cuda的时候，知道这是一个gpu运行的文件，点进去之后，很长很多，对我这样一个很少用cpp的人，基本就是天书，因此我准备看看cpu下运行的cpp文件是怎么写的。（真的只是简单的看看）因为关于如何调试cpp文件，我学了。这里就简单介绍一下！关于GDB的内容在下面。
cpp代码我不是很懂，但是看了下CPU运行的voxelization的文件，我大体应该知道这是干什么的啦，是这样的，前面我们有点的一个数组，又新建了体素的数组，这个体素的数组主要有两个：一个是用来存放点的数据的数组，另一个是存放的每个体素的坐标。而这个cpp文件是干的这件事：将点云切分一个又一个体素，并且将点信息放进体素中。最终返回的一共存放了多少个体素。
```ruby
# select the valid voxels
voxels_out = voxels[:voxel_num]
coors_out = coors[:voxel_num].flip(-1) # (z, y, x) -> (x, y, z)
num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
return voxels_out, coors_out, num_points_per_voxel_out
```
之后就是将空的体素去掉。返回的就是包含点信息的体素数组、体素坐标数组、每个体素包含点个数的数组。
接下来就是这个代码中一个管用的操作，我们知道在图像的处理中，将多张图片存放在一个数组中，第一个维度就是batch_size，但是这里代码中将一个batch的多帧点云先存放在数组中，然后利用torch.cat的操作将其拼接。
```ruby
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
###===这里的self.voxel_layer就是那个划分pillar的函数，后面是套娃，就不粘上来啦==###
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        pdb.set_trace()
###===存放在列表中，后续cat操作===###
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)
        return pillars, coors_batch, npoints_per_pillar
```
最终得到的还是上面说的那三个，只不过这里就是将列表化成数组，同时coors_batch需要注意下，不是简单的cat操作，而是增加了一个维度，为了说明是哪一帧的点云。
到这里整个self.pillar_layer就结束了，整体看下来感觉写的不多，但是我想说这块真的不好懂。我感觉这种数据操作包括后面对bboxes的解码都是很难的很麻烦的。
接下来我们就进入到self.pillar_encoder部分。上面的部分说过了，只是体素的划分，这里也是一个简单的数据处理，首先我们要清楚输入，对输入的解释，下在下面代码绿色的注释中了。接下来做了一些事情，一点点看，首先就是计算了每个体素的中心点坐标，然后将计算体素内每个点到这个中心点的距离。
```ruby
    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4(体素个数，32，xyz+反射强度)，32是每个体素的点数，不够就补零
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        (体素个数，1+3)，3指的是体素所在的坐标，1指的是体素在哪帧点云
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        (体素个数，)每个体素包含的点数
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)#=====这样里面就不再是xyz而是距离中心点的距离
```
后面第2步是这样的，因为coors_batch中是每个体素的坐标也就是index，这里index乘体素的大小，在加上一个self.x_offset，这样计算出的就是pillar中每个点距离体素中心点的xyz距离。是不是感觉这个和上面的没啥区别，那你就错了，这两个地方的中心点可不一样，第一个是体素内所有点的中心点，而第二个是体素的中心点。
```ruby
        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)
```
后面第三步就是将特征进行拼接。可以看到这里cat了pillar的xyz坐标+反射强度、所有点到中心点的xyz距离、所有点到体素中心点的xy距离。
```ruby
        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
```
第四部分一个非常漂亮的实现！妈耶~我是看了很久，想了很久才知道这部分是做了什么！！首先我们要知道每一个体素内的的点一定是大于0，小于32的（最大值代码中可以设定更改），况且，点云检测和图像检测最大的不同就是，点云是稀疏的。所以很多体素中会有xyz均等于0的点，也就是空点。但是上面我们在做的是一个中心点距离计算，那就意味着这些空点，计算之后，这个距离就不再是0了，但是实际上这些距离并不是初始的特征，所以需要一个mask掩码，将那些空点的feature全部等于0。第四部分整体就是做的这件事。当你了解了实现的内容是什么，再看代码，真的，漂亮的实现！美如画的代码！
```ruby
        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]
```
第五第六部分在说之前，首先清楚，pointpillar的优势是什么，伪图片！处理速度优。所以这里的embedding就是将输入特征转变为类似图片的特征，然后就可以直接扔给backbone、neck、head等。详细看下，输入特征在这里只是经过一个conv（1dconv+bn+relu），这里的conv主要是改变通道数(9--->64)，然后求一个最大值，这里不用说了，就是pointnet的一个原理。
第六部分一开始又新建了一个列表，我动动脚指头都知道，最后它又要cat。然后呢，这里将之前每个体素的对应的index拿出来，后面对点云进行划分网格，self.x_l和self.y_l就是长宽的网格数。一开始创建的canvas就是类似图片的形式啦，可以看出来(长，宽，通道数)，然后根据之前拿到的体素的index也就是xy坐标，把feature放进去。然后append到列表，最后cat。就得到我们在图像中非常熟悉的(bs,channel,w,h)！！！
到这里网络模型基本完成一般了！！！
```ruby
        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)
        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.y_l, self.x_l)
        pdb.set_trace()
        return batched_canvas
```
接下来的三个部分不就不详细说明了，无非就是backbone、neck、head。我的建议是这部分不如直接看论文的图。更方便了解，因为我简单看了下网络结构，和论文里话的基本没差。
@import "pointpillar1.png"
经过检测头输出了三个部分：cls_pred、bbox_pred、dir_cls_pred。这三个分别表示的是类别预测、bbox的位置预测、朝向预测。简单说下这里的三个输出的shape分别是(6,18,248,216)、(6,42,248,216)、(6,12,6248,216)、(6,12,248,216)。
接下来就进入到后面的解码过程，说实话，解码的过程中，有些地方我也是很迷糊。
```ruby
device = bbox_cls_pred.device
feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
anchors = self.anchors_generator.get_multi_anchors(feature_map_size)#anchors(248,216,3,2,7)
batched_anchors = [anchors for _ in range(batch_size)]
```
后面解码的部分，首先第一步就是生成anchor，在图像的检测中，anchor-base的，像yolo，模型最终生成的其实是偏移量，所以也会根据提前设好的anchor大小，在每个网格上生成anchor，然后将偏移量和anchor相加，最后得到的才是bboxes。那么我感觉在这里也是同样的道理！所以会先生成anchor。
忘记说了，上面的输出dim==1那个维度，分别是18、42、12。其中分别是6 * 3 锚框个数 * 类别数、6锚框个数 *7表示xyzwhl旋转角、6锚框个数 * 2个朝向。这里为什么是6个锚框，我也不太清楚，因为我只找到了3个锚框。。。
anchor的生成过程，这里就不细说了，我个人感觉不好理解。最终anchor的维度是(248,216,3,2,7)，我个人感觉，实际上代码只给了3个锚框的大小，但是因为有两个朝向，所以预测的锚框有6个。
再往后就会根据mode的不同，进行操作，分了三类，实际上训练是一类，验证和测试是一样的。这里就只看下训练啦。训练部分主要是一个函数anchor_target。这里很迷，涉及到一些正负样本。
我们分开看吧，首先输入是一个又一个的列表。
```ruby
#输入
batched_anchors: [(y_l, x_l, 3, 2, 7), (y_l, x_l, 3, 2, 7), ... ]
batched_gt_bboxes: [(n1, 7), (n2, 7), ...]
batched_gt_labels: [(n1, ), (n2, ), ...]
#输出
rt_dict = dict(
    batched_labels=torch.stack(batched_labels, 0), # (bs, y_l * x_l * 3 * 2)
    batched_label_weights=torch.stack(batched_label_weights, 0), # (bs, y_l * x_l * 3 * 2)
    batched_bbox_reg=torch.stack(batched_bbox_reg, 0), # (bs, y_l * x_l * 3 * 2, 7)
    batched_bbox_reg_weights=torch.stack(batched_bbox_reg_weights, 0), # (bs, y_l * x_l * 3 * 2)
    batched_dir_labels=torch.stack(batched_dir_labels, 0), # (bs, y_l * x_l * 3 * 2)
    batched_dir_labels_weights=torch.stack(batched_dir_labels_weights, 0) # (bs, y_l * x_l * 3 * 2)
)
这部分就不详细说了，建议去看代码。很多地方理解的不是很好，所以就不多多说啦。到这里整个网络模型基本就结束啦。后面的mode为val或者test时，要做的就是bboxes的解码，主要包括的（根据代码注释的顺序）0预处理，主要包括维度变换sigmoid等、1获取bboxes的分数，这个分数将会用来后续的nms、2解码预测的偏移量生成bboxes、3nms操作、4如果 bbox 编号高于 self.max_num，则过滤一些 bbox。
```
##### GDB
一开始是想对cpp文件进行debug的，但是后面因为一些原因就没做，好难呀。所以只是简单介绍一下GDB。首先就是安装
```ruby
pip install gdb
conda install gdb
```
所有人都推荐去官网学习，系统还详细。所以我也推荐一下。现在学了gdb之后，关于代码调试可能就没什么难的啦，gdb和pdb调试一切！然后还学了一招，就是当你不太了解某个命令的时候，可以使用下面的命令，相信对你会有所帮助的。
```ruby
man gdb
```
记下来就是一个简单的示例帮助了解gdb，因为需要调试代码，所以肯定就需要一个代码，这里简单的写了一个c的代码，毕竟c和c++没差。代码就是简单的定义了一个数组，然后打印。
```ruby
#include<stdio.h>
int main()
{
        int arr[4] = {1,2,3,4};
        for (int i=0;i<4;i++){
                printf("%d\n",arr[i]);
        }
        return 0;
}
```
接下来就是整个调试的过程，首先我们要对文件进行编译，这个没的说，但是我们需要告诉电脑，这个需要调试。加一个-g
```ruby
gcc -g test.c
```
编译完成之后，就会出现一个a.out的可执行文件。可以直接运行这个可执行文件。
```ruby
./a.out
#输出
(base) ktd@ktd-Alienware:~/rpf_ws/gdb_c$ ./a.out 
1
2
3
4
```
如果是要调试的话，在可执行文件前面加一个gdb即可。
```ruby
gdb ./a.out
#输出
(base) ktd@ktd-Alienware:~/rpf_ws/gdb_c$ gdb ./a.out 
GNU gdb (Ubuntu 8.1.1-0ubuntu1) 8.1.1
######这里省略很多########
Reading symbols from ./a.out...done.
(gdb) 
```
然后就可以进行调试啦，接下来简单说下命令，其实和pdb没有什么区别，基本相同，n--->next运行下一行、p--->print打印、s--->step进入函数、b--->break打断点、r--->run运行代码，这里也就是可以在运行的时候打断点，后面可以跟代码的行数和函数名字。list可以查看原代码，我们就能知道代码的行数。
除此之外，可以输入shell+终端指令，这样就可以运行shell的命令啦，还有非常棒的一件事就是gdb有log日志记录功能，但是pdb没有。
```ruby
set logging on
```
其他没有提到的就不说了，毕竟这不是重点。

### loss
okk，到这里，已经接近尾声啦，因为loss函数，其实没啥说的，我们刚才已经说过了，就是在检测头的输出包括三个：类别预测、bboxes的位置预测、朝向预测。所以在loss中也是分成了三个部分。
```ruby
    def forward(self,
                bbox_cls_pred,
                bbox_pred,
                bbox_dir_cls_pred,
                batched_labels, 
                num_cls_pos, 
                batched_bbox_reg, 
                batched_dir_labels):
        '''
        bbox_cls_pred: (n, 3)
        bbox_pred: (n, 7)
        bbox_dir_cls_pred: (n, 2)
        batched_labels: (n, )
        num_cls_pos: int
        batched_bbox_reg: (n, 7)
        batched_dir_labels: (n, )
        return: loss, float.
        '''
        # 1. bbox cls loss
        # focal loss: FL = - \alpha_t (1 - p_t)^\gamma * log(p_t)
        #             y == 1 -> p_t = p
        #             y == 0 -> p_t = 1 - p
        pdb.set_trace()
        nclasses = bbox_cls_pred.size(1)
        batched_labels = F.one_hot(batched_labels, nclasses + 1)[:, :nclasses].float() # (n, 3)
        bbox_cls_pred_sigmoid = torch.sigmoid(bbox_cls_pred)
        weights = self.alpha * (1 - bbox_cls_pred_sigmoid).pow(self.gamma) * batched_labels + \
             (1 - self.alpha) * bbox_cls_pred_sigmoid.pow(self.gamma) * (1 - batched_labels) # (n, 3)
        cls_loss = F.binary_cross_entropy(bbox_cls_pred_sigmoid, batched_labels, reduction='none')
        cls_loss = cls_loss * weights
        cls_loss = cls_loss.sum() / num_cls_pos
        # 2. regression loss
        reg_loss = self.smooth_l1_loss(bbox_pred, batched_bbox_reg)
        reg_loss = reg_loss.sum() / reg_loss.size(0)
        # 3. direction cls loss
        dir_cls_loss = self.dir_cls(bbox_dir_cls_pred, batched_dir_labels)
        # 4. total loss
        total_loss = self.cls_w * cls_loss + self.reg_w * reg_loss + self.dir_w * dir_cls_loss
        
        loss_dict={'cls_loss': cls_loss, 
                   'reg_loss': reg_loss,
                   'dir_cls_loss': dir_cls_loss,
                   'total_loss': total_loss}
        return loss_dict
```
这里面唯一不容易理解的weight那个参数是怎么计算的。哈哈哈，这里我也不太清楚，但是感觉无伤大雅。整体流程就是类别预测，首先生成one-hot独热码，然后对预测值做sigmoid，然后交叉熵损失函数计算类别损失、其余位置信息和朝向信息，直接调用损失函数即可。easy~easy~
