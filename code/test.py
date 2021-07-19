import numpy as np

from libs.utils import *

class Test_Process_NMS(object):
    def __init__(self, cfg, dict_DB):

        self.cfg = cfg
        self.dataloader = dict_DB['testloader']

        self.SLNet = dict_DB['SLNet']
        self.forward_model = dict_DB['forward_model']
        self.batch_size = self.cfg.batch_size['test_line']
        self.size = to_tensor(np.float32(cfg.size))
        self.candidates = load_pickle(self.cfg.pickle_dir + 'detector_test_candidates')
        self.candidates = to_tensor(self.candidates).unsqueeze(0)
        self.candidates = self.candidates.float()
        self.cand_num = self.candidates.shape[1]
        self.step = create_forward_step(self.candidates.shape[1],
                                        cfg.batch_test_size['test_line'])

        self.visualize = dict_DB['visualize']
        self.test_size=cfg.test_size

        #test
        self.test_log=cfg.output_dir + 'test/log/logfile.txt'

    def run(self, SLNet, mode='test'):
        result = {'out': {'left': [], 'middle': [], 'right': []},
                  'gt': {'left': [], 'middle': [], 'right': []}}

        with torch.no_grad():
            self.SLNet.eval()

            for i, self.batch in enumerate(self.dataloader):  # load batch data

                self.img_name = self.batch['img_name'][0]
                self.img = self.batch['img'].cuda()
                left_gt = self.batch['left_gt'][0][:, :4]
                middle_gt = self.batch['middle_gt'][0][:, :4]
                right_gt = self.batch['right_gt'][0][:, :4]
                mul_gt = self.batch['mul_gt'][0][:, :4]

                # semantic line detection
                out = self.forward_model.run_detector(img=self.img,
                                                      line_pts=self.candidates,
                                                      step=self.step,
                                                      model=self.SLNet)
                # reg result          
                out['pts'] = self.candidates[0] + out['reg'] * self.size

                print("ses", len(self.candidates[0]))
                # cls result

                # primary 3 lines(left, middle, right)
                lf_sorted = torch.argsort(out['cls'][:, 1], descending=True)
                mid_sorted = torch.argsort(out['cls'][:, 2], descending=True)
                rg_sorted = torch.argsort(out['cls'][:, 3], descending=True)
                out['left'] = out['pts'][lf_sorted[0:self.test_size], :].unsqueeze(0)
                out['middle'] = out['pts'][mid_sorted[0:self.test_size], :].unsqueeze(0)
                out['right'] = out['pts'][rg_sorted[0:self.test_size], :].unsqueeze(0)                

                # visualize
                self.visualize.display_for_test(batch=self.batch, out=out)
                # self.visualize.draw_all_lines(batch=self.batch, out=out['pts'], candidates=self.candidates)

                # record output data
                result['out']['left'].append(out['left'])
                result['out']['middle'].append(out['middle'])
                result['out']['right'].append(out['right'])
                result['gt']['left'].append(left_gt)
                result['gt']['middle'].append(middle_gt)
                result['gt']['right'].append(right_gt)
    

                print('image %d ---> %s done!' % (i, self.img_name))


        # save pickle
        save_pickle(dir_name=self.cfg.output_dir + 'test/pickle/',
                    file_name='result',
                    data=result)
    
    
                


