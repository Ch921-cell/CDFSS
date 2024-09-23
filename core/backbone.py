from functools import reduce
from operator import add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet



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
        elif typestr =='dinov2_vitl14':
            
            self.feature_extractor = torch.hub.load('', 'dinov2_vitl14', source='local').cuda()
            self.upsample = nn.Upsample(size=None, scale_factor=2, mode='nearest')
            self.downsample = nn.MaxPool2d(kernel_size=2)
        else:
            raise Exception('Unavailable backbone: %s' % typestr)
        self.feature_extractor.eval()

        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def extract_feats(self, img):
        feats = []
        c = 1024
        h = 28
        w = 28
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # with torch.no_grad():
        #     features_dict = self.feature_extractor.forward_features(img)
        #     feat = features_dict['x_norm_patchtokens']
        # feat = feat.permute(0,2,1).contiguous()
        # feat = feat.view(-1,c,h,w)
        # feats.append(feat)
        # return feats
        # with torch.no_grad():
        #     features_dict = self.feature_extractor.get_intermediate_layers(img, n=[20, 21, 22, 23])
        # for i in features_dict:
        #     i = i.permute(0, 2, 1).contiguous()
        #     feats.append(i.view(-1, c, h, w))
        with torch.no_grad():
            temp = self.feature_extractor.forward_features(img)
            feat = temp['x_norm_patchtokens']
        feat = feat.permute(0, 2, 1).contiguous()
        feats.append(feat.view(-1, c, h, w))
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
    feat_extr_method = Backbone("dinov2_vitl14").to('cuda:0').extract_feats
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
    # print(res1.shape)
    # print(res_ori.shape)
    # print(res1.equal(res_ori))
    print(len(res))
    for i in res:
        print(i.shape)
