import sys
sys.path.append('../mcunet')

import random, numpy as np
from torch import nn

from mcunet.tinynas.elastic_nn.modules import DynamicMBConvLayer, DynamicConvLayer
from mcunet.tinynas.nn.modules import ConvLayer, IdentityLayer, MBInvertedConvLayer
from mcunet.tinynas.nn.networks import MobileInvertedResidualBlock
from mcunet.utils import make_divisible, val2list, MyNetwork

from models.yolo import Detect

class OFAProxylessNASDets(MyNetwork):

    default_anchors = [
        (10, 13), (16, 30), (33, 23),
        (30, 61), (62, 45), (59, 119),
        (116, 90), (156, 198), (373, 326)
    ]

    def __init__(self, n_classes=20, bn_param=(0.1, 1e-3),
                 width_mult_list=1.0, ks_list=3, expand_ratio_list=6, depth_list=4):

        self.width_mult_list = val2list(width_mult_list, 1)
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        base_stage_width = [
            32, 32,             # NOTE: first_conv, first_block
            32, 32, 64, 96,     # NOTE: backbone
            64, 32,             # NOTE: fpn
            64, 96,             # NOTE: pan
            32, 64, 96          # NOTE: head
        ]

        input_channel = [make_divisible(base_stage_width[0] * width_mult, 8) for width_mult in self.width_mult_list]
        first_block_width = [make_divisible(base_stage_width[1] * width_mult, 8) for width_mult in self.width_mult_list]

        # first conv layer
        if len(input_channel) == 1:
            first_conv = ConvLayer(
                3, max(input_channel), kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
            )
        else:
            first_conv = DynamicConvLayer(
                in_channel_list=val2list(3, len(input_channel)), out_channel_list=input_channel, kernel_size=3,
                stride=2, act_func='relu6'
            )
        # first block
        if len(first_block_width) == 1:
            first_block_conv = MBInvertedConvLayer(
                in_channels=max(input_channel), out_channels=max(first_block_width), kernel_size=3, stride=1,
                expand_ratio=1, act_func='relu6',
            )
        else:
            first_block_conv = DynamicMBConvLayer(
                in_channel_list=input_channel, out_channel_list=first_block_width, kernel_size_list=3,
                expand_ratio_list=1, stride=1, act_func='relu6',
            )
        first_block = MobileInvertedResidualBlock(first_block_conv, None)

        input_channel = first_block_width

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1

        stride_stages = [2, 2, 2, 2] + [1] * 7 # NOTE: fix stride for neck/head
        if depth_list is None:
            n_block_list = [2, 3, 4, 3] + [1] * 7
            self.depth_list = [4, 4]
            print('Use MobileNetV2 Depth Setting')
        else:
            n_block_list = [max(self.depth_list)] * 4 + [1] * 7 # NOTE: fix depth for neck/head # 2

        width_list = []
        for base_width in base_stage_width[2:]:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        for j, (width, n_block, s) in enumerate(zip(width_list, n_block_list, stride_stages)):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1

                if j > 3: input_channel = output_channel # NOTE: neck/head channel correction
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(input_channel, 1), out_channel_list=val2list(output_channel, 1),
                    kernel_size_list=ks_list, expand_ratio_list=expand_ratio_list, stride=stride, act_func='relu6',
                )

                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

                blocks.append(mb_inverted_block)
                input_channel = output_channel

        super(OFAProxylessNASDets, self).__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [
            len(block_idx) for block_idx in self.block_group_info
        ]

        # NOTE: static layers
        p3_channels, p4_channels, p5_channels = [width[0] for width in width_list[-3:]]

        anchors = [item for sublist in OFAProxylessNASDets.default_anchors for item in sublist]
        anchors = [anchors[i:i+6] for i in range(0, len(anchors), 6)]
        self.stride = np.array([8, 16, 32])
        anchors = [item / stride for item, stride in zip(anchors, self.stride)]

        # NOTE: upsamples
        self.fpn_p5_to_p4 = nn.Sequential(
            ConvLayer(p5_channels, p4_channels, kernel_size=1, stride=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act'),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.fpn_p4_to_p3 = nn.Sequential(
            ConvLayer(p4_channels, p3_channels, kernel_size=1, stride=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act'),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # NOTE: downsamples
        self.pan_p3_to_p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvLayer(p3_channels, p4_channels, kernel_size=1, stride=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act'),
        )
        self.pan_p4_to_p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvLayer(p4_channels, p5_channels, kernel_size=1, stride=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act'),
        )

        # NOTE: head
        self.detect = Detect(nc=n_classes, anchors=anchors, ch=(p3_channels, p4_channels, p5_channels), inplace=True)
        self.model = [self.detect]
        self.detect.stride = self.stride

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAProxylessNASDets'

    def forward(self, x):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        # blocks
        features = []
        for stage_id, block_idx in enumerate(self.block_group_info[:4]):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

            # NOTE: extract features
            if stage_id in {1, 2, 3}:
                features.append(x)

        # NOTE: compute helper function
        def compute(stage_id, block_idx, x):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)
            return x

        # NOTE: fpn
        p5 = features[2]
        p4 = compute(4, self.block_group_info[4], self.fpn_p5_to_p4(p5) + features[1])
        p3 = compute(5, self.block_group_info[5], self.fpn_p4_to_p3(p4) + features[0])

        # NOTE: pan
        p4 = compute(6, self.block_group_info[6], self.pan_p3_to_p4(p3) + p4)
        p5 = compute(7, self.block_group_info[7], self.pan_p4_to_p5(p4) + p5)

        return self.detect([p3, p4, p5])

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
            if stage_id in {3, 5, 7}: _str += '\n'
        return _str

    @property
    def config(self):
        return {
            'name': OFAProxylessNASDets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'upsamples': [
                block[0].config for block in [self.fpn_p5_to_p4, self.fpn_p4_to_p3]
            ],
            'downsamples': [
                block[1].config for block in [self.pan_p3_to_p4, self.pan_p4_to_p5]
            ],
            'final_conv': [
                block.config for block in [self.head_p3, self.head_p4, self.head_p5]
            ]
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_weights_from_net(self, proxyless_model_dict):
        model_dict = self.state_dict()
        for key in proxyless_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = proxyless_model_dict[key]
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, wid=None, ks=None, e=None, d=None):
        # add information to add a width multiplier
        # width_mult_id = val2list(wid, 3 + len(self.block_group_info))
        # print(' * Using a wid of ', wid)
        for m in self.modules():
            if hasattr(m, 'out_channel_list'):
                if wid is not None:
                    m.active_out_channel = m.out_channel_list[wid]
                else:
                    m.active_out_channel = max(m.out_channel_list)

        # n_channel_choices = [len(m.out_channel_list) for m in self.modules() if hasattr(m, 'out_channel_list')]
        # print(n_channel_choices)
        # exit()
        # def set_output_channel(m):
        #     if hasattr(m, 'active_out_channel') and hasattr(m, 'out_channel_list'):
        #         m.active_out_channel = make_divisible(max(m.out_channel_list) * wid, 8)
        # set_output_channel(self.first_conv)
        # set_output_channel(self.feature_mix_layer)
        # for b in self.blocks:
        #     set_output_channel(b.mobile_inverted_conv)

        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.mobile_inverted_conv.active_kernel_size = k
            if e is not None:
                block.mobile_inverted_conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type='depth'):
        # only used for progressive shrinking
        if constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        elif constraint_type == 'width_mult':
            self.__dict__['_widthMult_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None
        self.__dict__['_widthMult_include_list'] = None

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        # sample width_mult, move to last to keep the same randomness
        width_mult_setting = random.randint(0, len(self.width_mult_list) - 1)

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        raise NotImplementedError()

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        if len(self.width_mult_list) > 1:
            print(' * WARNING: sorting is not implemented right for multiple width-mult')

        for block in self.blocks[1:]:
            block.mobile_inverted_conv.re_organize_middle_weights(expand_ratio_stage)
