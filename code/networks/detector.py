import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

class FeatureExtraction(nn.Module):
    def __init__(self, feature_extraction_cnn='mobilenet_v2'):
        super(FeatureExtraction, self).__init__()

        if feature_extraction_cnn == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)

            mobilenetv2_layers = ['conv1', 'bottleneck1_1', 'bottleneck2_1', 'bottleneck2_2',
                                   'bottleneck3_1', 'bottleneck3_2', 'bottleneck3_3', 'bottleneck4_1',
                                   'bottleneck4_2', 'bottleneck4_3', 'bottleneck4_4', 'bottleneck5_1',
                                   'bottleneck5_2', 'bottleneck5_3', 'bottleneck6_1', 'bottleneck6_2',
                                   'bottleneck6_3', 'bottleneck7_1', 'conv2']

            last_layer = 'bottleneck3_3'
            last_layer_idx = mobilenetv2_layers.index(last_layer)

            self.model1 = nn.Sequential(*list(model.features.children())[:last_layer_idx+1])
            self.model2 = nn.Sequential(*list(model.features.children())[last_layer_idx+1:-1])

    def forward(self, img):
        feat1 = self.model1(img)
        feat2 = self.model2(feat1)

        return feat1, feat2


class Line_Pooling_Layer(nn.Module):
    def __init__(self, size, step=49):
        super(Line_Pooling_Layer, self).__init__()

        self.step = step
        self.f_size = int(np.sqrt(self.step))
        self.size = size


    def forward(self, feat_map, line_pts, ratio):

        b = line_pts.shape[1]
        line_pts = line_pts[:, :, :4] / (self.size - 1) * (self.size / ratio - 1)
        line_pts = (line_pts / (self.size / ratio - 1) - 0.5) * 2  # [-1, 1]    

        idxX = torch.tensor([0, 2])
        idxY = torch.tensor([1, 3])

        grid_X = line_pts[:, :, idxX]  # Width
        grid_Y = line_pts[:, :, idxY]  # Height

        line_X = F.interpolate(grid_X, self.step, mode='linear', align_corners=True)[0]
        line_Y = F.interpolate(grid_Y, self.step, mode='linear', align_corners=True)[0]

        line_X = line_X.view(line_X.size(0), self.f_size, self.f_size, 1)
        line_Y = line_Y.view(line_Y.size(0), self.f_size, self.f_size, 1)
        line_grid = torch.cat((line_X, line_Y), dim=3)

        _, c, h, w = feat_map.shape
        feat = feat_map.expand(b, c, h, w)

        f_lp = F.grid_sample(feat, line_grid)

        return f_lp

class Fully_connected_layer(nn.Module):
    def __init__(self):
        super(Fully_connected_layer, self).__init__()

        self.linear_1 = nn.Linear(7 * 7 * 352, 128) # 1024->32
        self.linear_2 = nn.Linear(128, 1024) # 1024->32


    def forward(self, x):
        x = x.view(x.size(0), -1)

        fc1 = self.linear_1(x)
        fc1 = F.relu(fc1)
        fc2 = self.linear_2(fc1)
        fc2 = F.relu(fc2)

        return fc1, fc2

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()

        self.linear = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.linear(x)
        #to be changed!
        x = F.log_softmax(x, dim=1)

        return x

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.linear = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.linear(x)

        return x

class SLNet(nn.Module):
    def __init__(self, cfg):
        super(SLNet, self).__init__()

        self.feature_extraction = FeatureExtraction()

        self.fully_connected = Fully_connected_layer()
        self.regression = Regression()
        self.classification = Classification()

        size = torch.FloatTensor(cfg.size).cuda()
        self.line_pooling = Line_Pooling_Layer(size=size)


    def forward(self, img, line_pts, feat1=None, feat2=None):
        
        if feat1 is None:
            feat1, feat2 = self.feature_extraction(img)

        # Line pooling
        lp1 = self.line_pooling(feat1, line_pts, torch.tensor(8))
        lp2 = self.line_pooling(feat2, line_pts, torch.tensor(16))

        lp_concat = torch.cat((lp1, lp2), dim=1)
        fc_out1, fc_out2 = self.fully_connected(lp_concat)  

        # Classification & Regression
        reg_out = self.regression(fc_out2)
        cls_out = self.classification(fc_out2)

        return {'reg': reg_out, 'cls': cls_out}


