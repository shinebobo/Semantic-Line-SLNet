import cv2

import matplotlib.pyplot as plt

from libs.utils import *
from libs.modules import *

class Visualize_plt(object):

    def __init__(self, cfg=None):

        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.size = to_tensor(np.float32(cfg.size))

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255
        self.test_size=cfg.test_size

        self.param = {'linewidth': [0, 1, 2, 3, 4], 'color': ['yellow', 'red', 'lime', 'blue']}

        self.show = {}


    def load_image(self, dir_name, file_name, name):
        # org image
        img = cv2.imread(dir_name + file_name)
        img = cv2.resize(img, (self.width, self.height)) 
        self.show[name] = img
        self.img = img

    def show_image(self):
        plt.figure()
        plt.imshow(self.img[:, :, [2, 1, 0]])

    def save_fig(self, path, name):
        mkdir(path)
        # plt.axis('off')
        plt.savefig(path + name, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_img(self, dir_name, file_name, list):
        disp = self.line
        for i in range(len(list)):
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp)


    def draw_lines_plt(self, pts, idx1, idx2, linestyle='-', zorder=1, idx=0):
        endpts = find_endpoints(pts, [self.width-1, self.height-1])
        pt_1 = (endpts[0], endpts[1])
        pt_2 = (endpts[2], endpts[3])
        plt.plot([pt_1[0], pt_2[0]], [pt_1[1], pt_2[1]],
                linestyle=linestyle,
                linewidth=self.param['linewidth'][idx1],
                color=self.param['color'][idx2],
                zorder=zorder)

    def draw_lines_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=1):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(data.shape[0]):
            pts = data[i].astype(np.int64)
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            img = cv2.line(img, pt_1, pt_2, color, s)

        self.show[name] = img

    def display_for_train_detector(self, batch, out, idx):
        
        img_name = batch['img_name'][0]
        candidates = batch['train_data'][0].cuda()
        self.show['img_name'] = img_name

        img = to_np(batch['img_rgb'][0].permute(1, 2, 0))
        img = np.uint8(img[:, :, [2, 1, 0]] * 255)
        self.show['img'] = img

        left_line = candidates[candidates[:, 5] == 1][:, :4]
        mid_line = candidates[candidates[:, 6] == 1][:, :4]
        right_line = candidates[candidates[:, 7] == 1][:, :4]
        neg_line = candidates[candidates[:, 4] == 1][:, :4]

        cls_out = torch.argmax(out['cls'], dim=1)
        out_left_line = candidates[cls_out == 1][:, :4] + \
                out['reg'][cls_out == 1] * self.size
        out_mid_line = candidates[cls_out == 2][:, :4] + \
                out['reg'][cls_out == 2] * self.size
        out_right_line = candidates[cls_out == 3][:, :4] + \
                out['reg'][cls_out == 3] * self.size
        out_neg_line = candidates[cls_out == 0][:, :4] + \
                out['reg'][cls_out == 0] * self.size


        left_line = to_np(left_line)
        mid_line = to_np(mid_line)
        right_line = to_np(right_line)
        neg_line = to_np(neg_line)
        out_left_line = to_np2(out_left_line)
        out_mid_line = to_np2(out_mid_line)
        out_right_line = to_np2(out_right_line)
        out_neg_line = to_np2(out_neg_line)

        self.draw_lines_cv(data=left_line, name='left')
        self.draw_lines_cv(data=mid_line, name='mid')
        self.draw_lines_cv(data=right_line, name='right')
        self.draw_lines_cv(data=neg_line, name='neg')
        self.draw_lines_cv(data=out_left_line, name='out_left')
        self.draw_lines_cv(data=out_mid_line, name='out_mid')
        self.draw_lines_cv(data=out_right_line, name='out_right')
        self.draw_lines_cv(data=out_neg_line, name='out_neg')

        self.save_img(dir_name=self.cfg.output_dir + 'train/display/',
                    file_name=str(idx)+'.jpg',
                    list=['left', 'mid', 'right', 'neg', 'out_left', 'out_mid', 'out_right', 'out_neg'])

    def display_for_test(self, batch, out):
        img_name = batch['img_name'][0]
        self.load_image(dir_name=self.cfg.img_dir,
                        file_name=img_name,
                        name='img')
        self.show_image()
        if 'left' in out.keys():
            for i in range(self.test_size):
                pts_pri = to_np(out['left'][0])
                self.draw_lines_plt(pts_pri[i], 4, 1, '-', zorder=2)
        if 'middle' in out.keys(): 
            for i in range(self.test_size):
                pts_pri = to_np(out['middle'][0])
                self.draw_lines_plt(pts_pri[i], 4, 2, '-', zorder=2)
        if 'right' in out.keys(): 
            for i in range(self.test_size):
                pts_pri = to_np(out['right'][0])
                self.draw_lines_plt(pts_pri[i], 4, 3, '-', zorder=2)
        if batch['left_gt'][0].shape[0] == 1:
            self.draw_lines_plt(batch['left_gt'][0][0], 4, 0, '--', zorder=3)
        if batch['right_gt'][0].shape[0] == 1:
            self.draw_lines_plt(batch['right_gt'][0][0], 4, 0, '--', zorder=3)
        if batch['middle_gt'][0].shape[0] == 1:
            self.draw_lines_plt(batch['middle_gt'][0][0], 4, 0, '--', zorder=3)

        self.save_fig(path=self.cfg.output_dir + 'test/out_single/',
                      name=img_name[:-4] + '.png')
    
    def draw_all_lines(self, batch, out, candidates):
        img_name = batch['img_name'][0]
        self.load_image(dir_name=self.cfg.img_dir,
                        file_name=img_name,
                        name='img')
        self.show_image()
        
        for i in range(200, 210):
             self.draw_lines_plt(candidates[0][i],4,0,'-',zorder=1, idx=i) 
        
        for i in range(200, 210):
            self.draw_lines_plt(out[i],4,1,'--', zorder=2, idx=i)
        
        self.save_fig(path=self.cfg.output_dir + 'test/out_single/',
                      name=img_name[:-4] + '.png')


