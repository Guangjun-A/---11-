# from paddle import nn
# import efficentgcn.utils as U
from efficentgcn.model_joint_tem.attentions import Attention_Layer
from efficentgcn.model_joint_tem.layers import Spatial_Tem_Graph_Layer, \
    Temporal_Tem_Basic_Layer,SpatialGraphConvJoint,\
    Spatial_Graph_Joint_Tem_Layer, Temporal_Basic_Layer, Spatial_Graph_Layer, \
    Spatial_Graph_Joint_Layer
import paddle
# from .. import utils as U
# from .attentions import Attention_Layer
# from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer
import paddle.nn.initializer as init
from efficentgcn.model_joint_tem.activations import *
import math
import yaml
from efficentgcn.dataset.graphs import Graph

"""
第一个分支（joint分支）加入了IDX

"""


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class EfficientGCN(nn.Layer):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(EfficientGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape
        self.joint_input_branches = nn.LayerList([EfficientGCN_Blocks_Joint(
            init_channel=stem_channel,
            block_args=block_args[:fusion_stage],
            input_channel=num_channel,
            **kwargs
        )])

        # input branches
        self.vec_input_branches = nn.LayerList([EfficientGCN_Blocks_Joint(
            init_channel=stem_channel,
            block_args=block_args[:fusion_stage],
            input_channel=num_channel,
            **kwargs)])
        self.bone_input_branches = nn.LayerList([EfficientGCN_Blocks_Joint(
            init_channel=stem_channel,
            block_args=block_args[:fusion_stage],
            input_channel=num_channel,
            **kwargs)])
        self.input_branches = self.joint_input_branches.\
            extend(self.vec_input_branches).\
            extend(self.bone_input_branches)

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage - 1][0]
        # self.add_sublayer('init_bn', nn.BatchNorm2D(input_channel))
        self.data_bn = nn.BatchNorm2D(num_input * last_channel)
        self.main_stream = EfficientGCN_Main_Blocks(
            init_channel=num_input * last_channel,
            block_args=block_args[fusion_stage:],
            **kwargs
        )

        # output
        last_channel = num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]
        self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)
        # init parameters
        self.init_param()

    def forward(self, x):

        N, I, C, T, V, M = x.shape  # I为三种输入
        x = x.transpose((1, 0, 5, 2, 3, 4)).reshape((I, N * M, C, T, V))
        # self.tem = self.one_hot(N, self.seg, 25)  # (N, V, T, T)
        # self.tem = self.tem.transpose((0, 3, 1, 2))  # .to(device)  # (N, T, V, T)
        #
        # input branches
        x = paddle.concat([branch(x[i]) for i, branch in enumerate(self.input_branches)], axis=1)

        # main stream
        x = self.main_stream(x)

        # output
        _, C, T, V = x.shape
        feature = paddle.reshape(x, [N, M, C, T, V])
        feature = paddle.transpose(feature, [0, 2, 3, 4, 1])
        # feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        # self.dropout(feature)
        out = self.classifier(feature)
        out = paddle.reshape(out, [N, -1])

        return out  # , feature

    def init_param(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                weight_init_(layer, 'Normal', mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2D):
                weight_init_(layer, 'Normal', mean=1.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm1D):
                weight_init_(layer, 'Normal', mean=1.0, std=0.02)


class BN_Layer(nn.Layer):
    def __init__(self, input_channel):
        super(BN_Layer, self).__init__()
        self.data_bn = nn.BatchNorm2D(input_channel)

    def forward(self, x):
        if self.data_bn:
            x.stop_gradient = False
        x = self.data_bn(x)
        return x


class EfficientGCN_Blocks(nn.Sequential):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_sublayer('init_bn', BN_Layer(input_channel))
            self.add_sublayer('stem_scn',
                              Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_sublayer('stem_tcn', Temporal_Tem_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        temporal_layer = import_class(f'efficentgcn.model_joint_tem.layers.Temporal_Tem_Sep_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_sublayer(f'block-{i}_scn',
                              Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_sublayer(f'block-{i}_tcn-{j}',
                                  temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_sublayer(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel


class EfficientGCN_Main_Blocks(nn.Sequential):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Main_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_sublayer('init_bn', BN_Layer(input_channel))
            self.add_sublayer('stem_scn',
                              Spatial_Graph_Joint_Tem_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_sublayer('stem_tcn', Temporal_Tem_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        temporal_layer = import_class(f'efficentgcn.model_joint_tem.layers.Temporal_Tem_Sep_Layer')
        seg = 500
        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_sublayer(f'block-{i}_scn',
                              Spatial_Graph_Joint_Tem_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                seg = seg // s
                self.add_sublayer(f'block-{i}_tcn-{j}',
                                  temporal_layer(channel, temporal_window_size, stride=s, seg=seg, **kwargs))
            self.add_sublayer(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel

#
class EfficientGCN_Blocks_Joint(nn.Sequential):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks_Joint, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_sublayer('init_bn', BN_Layer(input_channel))
            self.add_sublayer('stem_scn',
                              Spatial_Graph_Joint_Tem_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_sublayer('stem_tcn', Temporal_Tem_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        temporal_layer = import_class(f'efficentgcn.model_joint_tem.layers.Temporal_Tem_Sep_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_sublayer(f'block-{i}_scn',
                              Spatial_Graph_Joint_Tem_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_sublayer(f'block-{i}_tcn-{j}',
                                  temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_sublayer(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel

class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(EfficientGCN_Classifier, self).__init__()

        self.add_sublayer('gap', nn.AdaptiveAvgPool3D(1))
        self.add_sublayer('dropout', nn.Dropout(drop_prob))
        self.add_sublayer('fc', nn.Conv3D(curr_channel, num_class, kernel_size=1))


def weight_init_(layer,
                 func,
                 weight_name=None,
                 bias_name=None,
                 bias_value=0.0,
                 **kwargs):
    """
    In-place params init function.
    Usage:
    .. code-block:: python

        import paddle
        import numpy as np

        data = np.ones([3, 4], dtype='float32')
        linear = paddle.nn.Linear(4, 4)
        input = paddle.to_tensor(data)
        print(linear.weight)
        linear(input)

        weight_init_(linear, 'Normal', 'fc_w0', 'fc_b0', std=0.01, mean=0.1)
        print(linear.weight)
    """

    if hasattr(layer, 'weight') and layer.weight is not None:
        getattr(init, func)(**kwargs)(layer.weight)
        if weight_name is not None:
            # override weight name
            layer.weight.name = weight_name

    if hasattr(layer, 'bias') and layer.bias is not None:
        init.Constant(bias_value)(layer.bias)
        if bias_name is not None:
            # override bias name
            layer.bias.name = bias_name


class Model(nn.Layer):
    def __init__(self, config, input_mode):
        super(Model, self).__init__()
        A = Graph('ntu').A
        parts = Graph('ntu').parts
        self.__activations = {
            'relu': nn.ReLU(),
            'relu6': nn.ReLU6(),
            'hswish': HardSwish(),
            'swish': Swish(),
        }

        with open(config, 'r', encoding="utf-8") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        kwargs = {
            'data_shape': (len(input_mode), 4, None, None, None),  # N, I=3, C=4, T, V, M
            'num_class': 30,
            'A': paddle.to_tensor(A),
            'parts': parts,
        }

        args['model_args'].update({
            'act': self.__activations[args['model_args']['act_type']],
            'block_args': self.rescale_block(args['model_args']['block_args'],
                                             args['model_args']['scale_args'],
                                             int(args['model_type'][-1])),
        })
        args.update(**kwargs)

        self.model = self.create(args['model_type'], **args['model_args'], **kwargs)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def rescale_block(block_args, scale_args, scale_factor):
        channel_scaler = math.pow(scale_args[0], scale_factor)
        depth_scaler = math.pow(scale_args[1], scale_factor)
        new_block_args = []
        for [channel, stride, depth] in block_args:
            channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
            depth = int(round(depth * depth_scaler))
            new_block_args.append([channel, stride, depth])
        return new_block_args

    def create(self, model_type, act_type, block_args, scale_args, **kwargs):
        kwargs.update({
            'act': self.__activations[act_type],
            'block_args': self.rescale_block(block_args, scale_args, int(model_type[-1])),
        })
        return EfficientGCN(**kwargs)


if __name__ == "__main__":
    pass
    # from .activations import *
    # import math
    # import yaml
    # from efficentgcn.dataset.graphs import Graph
    # A =Graph('ntu').A
    # parts = Graph('ntu').parts
    # with open("2001.yaml", 'r') as f:
    #     args = yaml.load(f, Loader=yaml.FullLoader)
    # print(args)
    N, I, C, T, V, M = 5, 3, 4, 500, 25, 1
    data = paddle.randn((N, I, C, T, V, M), dtype=paddle.float32)
    # __activations = {
    #     'relu': nn.ReLU(),
    #     'relu6': nn.ReLU6(),
    #     'hswish': HardSwish(),
    #     'swish': Swish(),
    # }
    # kwargs = {
    #     'data_shape': (3, 2, None, None, None),
    #     'num_class': 30,
    #     'A': paddle.to_tensor(A),
    #     'parts': parts,
    # }
    #
    model = Model('../configs/efficentgcn.yaml', 'JVB')
    y = model(data)
    print(sum(param.numel() for param in model.model.joint_input_branches.parameters()))
    # print(len(y))
    # print(y.shape)
    # model = EfficientGCN((I, C, None, None, None), **args['model_args'])
