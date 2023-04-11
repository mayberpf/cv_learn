# 欢迎来到点云的世界
## pointnet
最近全面开始学习点云的目标检测，之前也接触过点云方面的目标检测，像pointnet之类的也学习过，还有pointpillar等都做过复现，但是一直不深入，更别说如何去改进创新。同时由于研究院的工作需要，因此来学一下点云的目标检测。
## 初始
因为基本上就没有接触过点云，所以很多东西都不知道。但是反观图像的话，图像的有基本图像处理的一些库：像cv2、pil等。点云同样也是的，所以我在学习点云目标检测算法之前简单的了解了一下open3d。之所以是open3d是因为，据说open3d的python接口要比pcl的好的多。除此之外，就没有什么点云方面的基础了。
## POINTNET
我们按照顺序来，先是pointnet，然后是pointnet++。
先声明一下，代码链接在这里：https://github.com/yanx27/Pointnet_Pointnet2_pytorch
在代码中的参数设定中，可以设置不同的网络模型，包括pointnet做分类分割pointnet2做分类分割等。
```ruby
    parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
```
我简单的看了代码的全部部分，实际上因为是上游任务，就是点云分类，所以代码比较简单，就是基本上除了网络模型，其他代码都是一些容易理解的代码。所以本篇文章主要讲述的是pointnet和pointnet2的网络结构。
首先是pointnet。
我先说下我的个人观点，第一次看点云的网络结构，尤其是这个pointnet，给我一种很不能理解的感觉。尤其是T-net那一部分。有关pointnet的论文，我是很早之前看过的，这个模型或者说论文的亮点就是先利用升维操作点云数据，在利用最大值对点云进行分类。反而是前面说对点云进行旋转平移变换，我个人感觉没有必要。但是模型中确实做了旋转平移变换，这就导致我一开始很难看懂。
@import "pointnet.png"
```ruby
class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        pdb.set_trace()
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat
if __name__ =='__main__':
    point_inputs = torch.rand(32,3,2500)
    # print(point_inputs.shape)
    model = get_model()
    out = model(point_inputs)
    print(out[0].shape)
```
这是模型的主干，以及我写的主函数进行测试和debug。这里最主要的可以看出来就是PointNetEncoder，后面进行的一系列线性层激活函数到softmax都是为了将维度降下来，降维等于类别数。你可能比较好奇这里的输出为什么是两个，我可以先告诉你x是类别的输出也就是一个bn*类别数的tensor，trans_feat表示的是旋转平移矩阵。接下来我们就看下PointNetEncoder这里有什么。
```ruby
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
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
        if self.feature_transform:
            self.fstn = STNkd(k=64)
    def forward(self, x):
        import pdb;pdb.set_trace()
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
            return torch.cat([x, pointfeat], 1), trans, trans_feat
```
简单来说，PointNetEncoder里面主要做了三件事：1、生成了一个3*3的矩阵，对点云进行旋转。2、生成一个64 *64的矩阵对点云特征进行旋转。3、升维求最值。在上面代码中有```self.stn = STN3d(channel)```和```self.fstn = STNkd(k=64)```分别代表的就是第一第二件事。详细的代码可以去链接下载一下看，我个人是感觉有些无厘头，不好解释。
除此之外，我对modelnet40这个数据集以及代码中的dataset进行了解。数据集中每个点云是xyzrgb的txt格式。然后在dataset中直接读取就好了，在数据处理方面，初始化将点云类别和点云地址放在一起，getitem中在np.loadtxt读取点云的txt文件。唯一值得注意的就是有一步对点云进行了采样，比如原始点云10000个点，但是加载进来1024个点，这些点可以通过随机采样进来，也可以通过最远点采样进来。
差点忘了，loss！！！pointnet的loss分为了两部分，这也是我不能理解的，pointnet2的loss就只有一部分。
```ruby
class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
```
这里我们已经知道了模型的输出是两部分，预测和旋转矩阵。那么loss函数毋庸置疑肯定有一部分是用来计算预测类别和真实类别的偏差loss，第二部分呢就是计算64*64旋转矩阵的loss，这也就是我不能理解的地方，因为根本就没有真实值呀。然后你继续看，代码中设定```mat_diff_loss_scale=0.001```旋转矩阵的loss占总共loss的0.001，我。。。你直接等于0不就好了。但是本着学习的态度，我还是去看了下，具体这个loss是怎么算的。
```ruby
def feature_transform_reguliarzer(trans):
    pdb.set_trace()
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]#生成对角线为1，其余元素全是0的数组
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
```
好消息是我知道他是怎么算的啦，坏消息是他这么算有什么依据吗，这是我又一个不理解的地方。它这么做的最后算出来loss，如果假设loss==0，那么trans实际上应该是一个64 *64的单位矩阵。意思？这个旋转矩阵最好的结果就是不旋转？如果真的要解释，我是这么想的：一开始因为我们对点云做了3 *3 的旋转，后来将点云升维，对点云特征做64 *64的旋转。所以最好的结果就是一开始的3 * 3旋转做到了极致，此时后面64 * 64 已经不需要旋转了，所以它应该是一个单位矩阵。
## POINTNET2
这个模型就会感觉比pointnet的可解释性强的多，毕竟那个旋转矩阵真的不是很能理解。首先还是要仔细看论文，作者提出了两个方式对点云进行划分。这里就只说其中一个代码，具体还要看论文了解。这个会比pointnet要舒服的多，因为上面刚说了loss，这里就先从loss说起。
```ruby
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss
```
四个字：简！洁！明！了！
然后我们来看整个网络模型。
@import "pointnet2.png"
```ruby
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
    def forward(self, xyz):
        pdb.set_trace()
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        pdb.set_trace()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x,l3_points
```
看到这个输出，你有没有“卧槽”，哈哈哈哈，又是两个输出，天都塌了！你做个分类任务，输出一个array不就好了，整两个，多难懂呀！其实还好，首先我们知道pointnet2的优化是什么。那就是他对点云做了一个基于中心的划分，然后用这个中心点来代替这这一区域的特征，是不是有点图像卷积的意思啦。所以这里的两个输出，x表示的就是分类的数组，l3_points表示的是最后的点云特征。所以说这里即使不return l3_points也是可以滴。
接下来我们简单说说，pointnet2的网络主要在```PointNetSetAbstractionMsg```和```PointNetSetAbstraction```这两部分。
```ruby
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)#16,512,16
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat
```
整体上，了解一下，这部分主要做的就是中心点的提取，以及中心点特征的提取，包括后面又进行了一次，实际上是在做完一次的基础上对中心点及其特征再进行提取。里面的几个主要的函数```farthest_point_sample#最远点采样```这个是获取中心点函数。```query_ball_point```这是根据中心点以及self.radius_lists确定不同半径的特征，然后卷积提取最后cat在一起。
上面的函数进行两次，第一次的中心点个数为512，第二次128。然后就进入到模型的另一个重要模块```PointNetSetAbstraction```
```ruby
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        import pdb;pdb.set_trace()
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
```
说实话，这一部分和之前的那个部分有异曲同工之妙，只不过前面是对中心点提取获取一个数量，但是这里实际上就是将前面128中心点提取成一个点，这一操作是在```sample_and_group_all```函数中进行的，当然作者也将中心点坐标cat到了特征中，具体可以看那个函数是怎么做的。最后进行升维和max操作。
最后利用线性层及softmax输出就是结果。