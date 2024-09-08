from functools import reduce
from operator import add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class Backbone(nn.Module):

    def __init__(self, typestr):
        super(Backbone, self).__init__()

        self.backbone = typestr

        # feature extractor initialization
        if typestr == 'resnet50':
            self.feature_extractor = resnet.resnet50(weights=resnet.ResNet50_Weights.DEFAULT)
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
            # define model
            self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
            self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        elif typestr =='dinov2_vitb14':
            
            self.feature_extractor = torch.hub.load('', 'dinov2_vitb14', source='local').cuda()
            self.upsample = nn.Upsample(size=None, scale_factor=2, mode='nearest')
            self.C2F1_0 = C2f(768,256)
            self.C2F1_1 = C2f(256,256)
            self.C2F1_2 = C2f(256,256)
            
            self.C2F2_0 = C2f(768, 512)
            self.C2F2_1 = C2f(512, 512)
            self.C2F2_2 = C2f(512, 512)
            
            self.C2F3_0 = C2f(768, 1024)
            self.C2F3_1 = C2f(1024, 1024)
            self.C2F3_2 = C2f(1024, 1024)
        else:
            raise Exception('Unavailable backbone: %s' % typestr)
        self.feature_extractor.eval()

        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def extract_feats(self, img):
        feats = []
        c = 768
        h = 28
        w = 28
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        with torch.no_grad():
            features_dict = self.feature_extractor.forward_features(img)
            feat = features_dict['x_norm_patchtokens']
        feat = feat.permute(0,2,1).contiguous()
        feat = feat.view(-1,c,h,w)
        feats.append(feat)
        # block_1
        # feat_1 = self.upsample(feat)
        # feat_1 = self.upsample(feat_1)
        # feat_1 = self.C2F1_0(feat_1)
        # feat_1 = self.C2F1_1(feat_1)
        # feat_1 = self.C2F1_2(feat_1)
        # feats.append(feat_1)
        # # block_2
        # feat_2 = self.upsample(feat)
        # feat_2 = self.C2F2_0(feat_2)
        # feat_2 = self.C2F2_1(feat_2)
        # feat_2 = self.C2F2_2(feat_2)
        # feats.append(feat_2)
        # # block_3
        feat_3 = self.C2F3_0(feat)
        # feat_3 = self.C2F3_1(feat_3)
        # feat_3 = self.C2F3_2(feat_3)
        feats.append(feat_3)
        
        return feats

    # def extract_feats(self, img):
    #     r""" Extract input image features """
    #     feats = []
    #     bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
    #     # Layer 0
    #     feat = self.feature_extractor.conv1.forward(img)
    #     feat = self.feature_extractor.bn1.forward(feat)
    #     feat = self.feature_extractor.relu.forward(feat)
    #     feat = self.feature_extractor.maxpool.forward(feat)
    #
    #     # Layer 1-4
    #     for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
    #         res = feat
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)
    #
    #         if bid == 0:
    #             res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)
    #
    #         feat += res
    #
    #         if hid + 1 in self.feat_ids:
    #             feats.append(feat.clone())
    #
    #         feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
    #
    #     return feats
    
if __name__ =="__main__":
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T
    feat_extr_method = Backbone("dinov2_vitb14").to('cuda:0').extract_feats
    # query_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    query_img = Image.open("1.jpg").convert('RGB')
    query_img = Image.fromarray(np.uint8(query_img))
    transform = T.Compose([
        T.Resize((28*14,28*14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    query_img = transform(query_img)
    query_img = query_img.unsqueeze(0).cuda()
    res = feat_extr_method(query_img)
    print(len(res))
    for i in res:
        print(i.shape)
