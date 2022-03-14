# EfficientGCN-v2

#### 介绍
在EfficientGCNv1的基础上引入了时间语义信息和节点语义信息的升级版，我们叫做EfficientGCN-v2。

#### 软件架构
软件架构说明


#### 训练命令

1.  生成最后结果文件。 `./submission.sh`
2.  训练模型。`./train.sh`
3.  stack boosting10折模型的结果,训练弱分类器。 `./stack-boosting.sh`

#### 单个模型框架
![模型框架](figs/frame_works.png)
#### 框架内各个子模块
![各个子模块](figs/module.png)

#### 最终融合模型介绍
| 第K折 |  模型文件   | 创新点  | 准确率 |
| ----  |  ----  | ----  |   ----   |
| 第0折 | net/4BranchNetEff  | 各个branch网络修改为SGN |     -    |
| 第1折 | net/4BranchNet_with_tem2  | 在efficent_gcn基础上加入时间信息和节点信息 |     -     |
| 第2折 | net/efficent_gcn_with_joint_v3  | 在efficent_gcn基础上加入节点信息  |     -     |
| 第3折 | net/4BranchNet_with_tem2  | 在efficent_gcn基础上加入时间信息和节点信息 |     -     |
| 第4折 | net/4BranchNet_with_tem2  | 在efficent_gcn基础上加入时间信息和节点信息 |     -     |
| 第5折 | net/efficent_gcn_with_joint_v2 |在efficent_gcn基础上加入节点信息  |     -     |
| 第6折 | net/4BranchNetEff  | 各个branch网络修改为SGN |     -     |
| 第7折 | net/efficent_gcn_with_joint_v2_with_tem2  | 在efficent_gcn基础上加入时间信息和节点信息 |    -      |
| 第8折 | net/efficent_gcn_with_joint_v2_with_tem2  | 在efficent_gcn基础上加入时间信息和节点信息 |     -     |
| 第9折 | net/efficent_gcn_with_joint_v2_with_tem2  | 在efficent_gcn基础上加入时间信息和节点信息 |    -      |
| - | - | - |    B榜准确率65.45741325     |
#### 项目文件介绍
- net 存放Model 文件
- efficentgcn 存放子模块文件
- configs 存放配置文件
- utils 存放模型保存读取记录文件
- final_test_B 存放B榜最终模型与结果文件
- data1 存放离线处理后的.npy数据文件
- feeder 存放DataLoder数据读取和预处理文件
- checkpoint 存放训练模型和日志文件
- figs 存放模型示意图
#### 使用的baseline模型的github地址
`https://github.com/yfsong0709/EfficientGCNv1`
`https://github.com/microsoft/SGN`
