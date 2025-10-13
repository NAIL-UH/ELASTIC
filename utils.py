import sys
sys.path.append('../mcunet')

import random, numpy as np, math
from torch import nn

from mcunet.tinynas.elastic_nn.modules import DynamicMBConvLayer, DynamicConvLayer
from mcunet.tinynas.nn.modules import ConvLayer, IdentityLayer, MBInvertedConvLayer
from mcunet.tinynas.nn.networks import MobileInvertedResidualBlock
from mcunet.utils import make_divisible, val2list, MyNetwork

from models.yolo import Detect
from ofamodel import OFAProxylessNASDets

# TODO: replace with better size predictor
class ProxylessNASDets(MyNetwork):

    default_anchors = [ # 416x416 resolution
        (10, 13), (16, 30), (33, 23),
        (30, 61), (62, 45), (59, 119),
        (116, 90), (156, 198), (373, 326)
    ]

    def __init__(self, config, n_classes=20, bn_param=(0.1, 1e-3),
                 width_mult_list=1.0, ks_list=3, expand_ratio_list=6, depth_list=4):

        self.width_mult_list = val2list(width_mult_list, 1)
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        # self.base_stage_width = base_stage_width

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        ks = config['ks'] # 4 blocks * 3 depth + 7 static
        e = config['e']
        self.d = config['d']
        # self.wid = config['wid']

        d = max(self.depth_list)
        self.ks = [ks[i:i+d] for i in range(0, len(ks)-7, d)] + [[k] for k in ks[-7:]]
        self.e = [e[i:i+d] for i in range(0, len(e)-7, d)] + [[r] for r in e[-7:]]
        self.d[-7:] = [1]*7 # NOTE: fix depth for neck/head

        # self.ks = [ks[i:i+d] for i in range(0, len(ks), d)]
        # self.e = [e[i:i+d] for i in range(0, len(e), d)]

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
            n_block_list = [max(self.depth_list)] * 4 + [1] * 7 # NOTE: fix depth for neck/head

        width_list = []
        for base_width in base_stage_width[2:]:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        for j, (width, n_block, s) in enumerate(zip(width_list, n_block_list, stride_stages)): # 4 + 7
            n_block = self.d[j]
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
                    kernel_size_list=val2list(self.ks[j][i], 1), expand_ratio_list=val2list(self.e[j][i], 1), stride=stride, act_func='relu6',
                )

                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

                blocks.append(mb_inverted_block)
                input_channel = output_channel

        super(ProxylessNASDets, self).__init__()
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

class MobileDetArchEncoder:
    def __init__(
        self,
        image_size_list=None,
        ks_list=None,
        expand_list=None,
        depth_list=None,
        n_stage=None,
    ):
        self.image_size_list = [224] if image_size_list is None else image_size_list
        self.ks_list = [3, 5, 7] if ks_list is None else ks_list
        self.expand_list = (
            [3, 4, 6]
            if expand_list is None
            else [int(expand) for expand in expand_list]
        )
        self.depth_list = [2, 3, 4] if depth_list is None else depth_list
        if n_stage is not None:
            self.n_stage = n_stage
        else:
            raise NotImplementedError

        # build info dict
        self.n_dim = 0
        self.r_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="r")

        self.k_info = dict(id2val=[], val2id=[], L=[], R=[])
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="k")
        self._build_info_dict(target="e")

    @property
    def max_n_blocks(self):
        return self.n_stage * max(self.depth_list) + 7

    def _build_info_dict(self, target):
        if target == "r":
            target_dict = self.r_info
            target_dict["L"].append(self.n_dim)
            for img_size in self.image_size_list:
                target_dict["val2id"][img_size] = self.n_dim
                target_dict["id2val"][self.n_dim] = img_size
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        else:
            if target == "k":
                target_dict = self.k_info
                choices = self.ks_list
            elif target == "e":
                target_dict = self.e_info
                choices = self.expand_list
            else:
                raise NotImplementedError
            for i in range(self.max_n_blocks):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for k in choices:
                    target_dict["val2id"][i][k] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = k
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)

    def arch2feature(self, arch_dict):
        ks, e, d, r = (
            arch_dict["ks"],
            arch_dict["e"],
            arch_dict["d"],
            arch_dict["image_size"],
        )

        feature = np.zeros(self.n_dim)
        for i in range(self.max_n_blocks):
            nowd = i % max(self.depth_list)
            stg = i // max(self.depth_list)
            if i >= self.max_n_blocks - 7 or nowd < d[stg]:
                feature[self.k_info["val2id"][i][ks[i]]] = 1
                feature[self.e_info["val2id"][i][e[i]]] = 1
        feature[self.r_info["val2id"][r]] = 1
        return feature

    def feature2arch(self, feature):
        img_sz = self.r_info["id2val"][
            int(np.argmax(feature[self.r_info["L"][0] : self.r_info["R"][0]]))
            + self.r_info["L"][0]
        ]
        assert img_sz in self.image_size_list
        arch_dict = {"ks": [], "e": [], "d": [], "image_size": img_sz}

        d = 0
        for i in range(self.max_n_blocks):
            skip = True
            for j in range(self.k_info["L"][i], self.k_info["R"][i]):
                if feature[j] == 1:
                    arch_dict["ks"].append(self.k_info["id2val"][i][j])
                    skip = False
                    break

            for j in range(self.e_info["L"][i], self.e_info["R"][i]):
                if feature[j] == 1:
                    arch_dict["e"].append(self.e_info["id2val"][i][j])
                    assert not skip
                    skip = False
                    break

            is_tail = i >= self.max_n_blocks - 7
            if skip and not is_tail:
                arch_dict["e"].append(0)
                arch_dict["ks"].append(0)
            elif skip and is_tail:
                raise ValueError(f"Missing kernel or expansion for fixed block {i}")
            else:
                d += 1

            if i < self.max_n_blocks - 7:
                if (i + 1) % max(self.depth_list) == 0:
                    arch_dict["d"].append(d)
                    d = 0
            elif i == self.max_n_blocks - 7:
                if d > 0 and (len(arch_dict["d"]) < self.n_stage):
                    arch_dict["d"].append(d)
                arch_dict["d"].extend([1] * 7)

        return arch_dict

    def random_sample_arch(self):
        return {
            "ks": random.choices(self.ks_list, k=self.max_n_blocks),
            "e": random.choices(self.expand_list, k=self.max_n_blocks),
            "d": random.choices(self.depth_list, k=self.n_stage) + [1]*7,
            "image_size": random.choice(self.image_size_list),
        }

    def mutate_resolution(self, arch_dict, mutate_prob):
        if random.random() < mutate_prob:
            arch_dict["image_size"] = random.choice(self.image_size_list)
        return arch_dict

    def mutate_arch(self, arch_dict, mutate_prob):
        for i in range(self.max_n_blocks):
            if random.random() < mutate_prob:
                arch_dict["ks"][i] = random.choice(self.ks_list)
                arch_dict["e"][i] = random.choice(self.expand_list)

        for i in range(self.n_stage):
            if random.random() < mutate_prob:
                arch_dict["d"][i] = random.choice(self.depth_list)
        return arch_dict

class SizePredictor:
    def __init__(
        self,
        n_classes=20,
        bn_param=(0.1, 1e-3),
        width_mult_list=1.5,
        ks_list=[1, 3, 5],
        expand_ratio_list=[1, 2, 3, 4, 5, 6],
        depth_list=[1, 2, 3, 4, 5],
    ):
        self.n_classes = n_classes
        self.bn_param = bn_param
        self.width_mult_list = width_mult_list
        self.ks_list = ks_list
        self.expand_ratio_list = expand_ratio_list
        self.depth_list = depth_list

    def get_efficiency(self, config):
        model = ProxylessNASDets(
            config=config,
            n_classes=self.n_classes,
            bn_param=self.bn_param,
            width_mult_list=self.width_mult_list,
            ks_list=self.ks_list,
            expand_ratio_list=self.expand_ratio_list,
            depth_list=self.depth_list
        )
        return sum(p.numel() for p in model.parameters())

class MemoryPredictor:
    def __init__(self, base_stage_width=None, width_mult=1, max_memory=0x14000, verbose=False):
        if base_stage_width is None:
            base_stage_width = [
                32, 32,             # NOTE: first_conv, first_block
                32, 32, 64, 96,     # NOTE: backbone
                64, 32,             # NOTE: fpn
                64, 96,             # NOTE: pan
                32, 64, 96          # NOTE: head
            ]
        self.base_stage_width = [x*width_mult for x in base_stage_width]
        self.max_memory = max_memory
        self.verbose = verbose

        if verbose:
            print(f'max_memory: {max_memory}')

    def __call__(self, config):
        return self.get_efficiency(config)

    def get_efficiency(self, config, res=(224, 224)):
        expand = config['e']
        depth = config['d']
        w, h = res
        stage_width = self.base_stage_width

        # factoring constants
        factor2 = 2 / 8 # multiply by 2 for ping pong, divide by 8 for byte
        factor4 = 4 / 8 # multiply by 4 for gapped output + ping pong
        factor6 = 6 / 8 # multiply by 5 for ungapped output, gapped output, ping pong

        not_exceeded = True
        memory_list = []
        reserved = 0
        def f(w, h, stage_width, factor, e=1, reserved=0):
            return math.ceil(w * h * stage_width * factor * e + reserved)

        # first conv/block
        memory_list.append(f(w, h, stage_width[0], factor2))
        w, h = w // 2, h // 2
        memory_list.append(f(w, h, stage_width[1], factor2))

        # blocks
        for i in range(4):
            width = stage_width[i + 2]
            for d in range(depth[i]):
                e = expand[5*i + d]

                # downscaled
                if not d:
                    w, h = w // 2, h // 2

                # factor selection
                factor = factor6 if width > 64 else factor4
                memory_list.append(f(w, h, width, factor, e, reserved))

            # memory reservation
            if i in {1, 2, 3}:
                if width > 64: reserved += f(w, h, width, factor4)
                else: reserved += f(w, h, width, factor2)

            print(reserved)

        w5, w4, w3 = [w*2**i for i in range(3)]
        h5, h4, h3 = [h*2**i for i in range(3)]

        # fpn
        memory_list.append(f(w5, h5, stage_width[5], factor2, expand[20], reserved))
        memory_list.append(f(w4, h4, stage_width[4], factor2, expand[21], reserved))

        # pan
        memory_list.append(f(w4, w4, stage_width[4], factor2, expand[22], reserved))
        memory_list.append(f(w3, h3, stage_width[3], factor2, expand[23], reserved))

        # head
        memory_list.append(f(w3, w3, stage_width[3], factor2, expand[24], reserved))
        memory_list.append(f(w4, h4, stage_width[4], factor2, expand[25], reserved))
        memory_list.append(f(w5, h5, stage_width[5], factor2, expand[26], reserved))

        # streaming mode
        for i in range(2):
            if memory_list[i] > self.max_memory:
                memory_list[i] = self.max_memory

        maximum = max(memory_list)
        if maximum > self.max_memory:
            not_exceeded = False

        if self.verbose and not not_exceeded:
            idx = [i for i, memory in enumerate(memory_list) if memory > self.max_memory]
            print(f'Memory Exceeded: {maximum} > {self.max_memory} at layers {idx}')
            print(f'Expand Index: {[i - 2 for i in idx]}')

        return memory_list, maximum, not_exceeded