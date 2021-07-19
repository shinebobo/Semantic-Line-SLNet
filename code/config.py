import os
import wandb

class Config(object):
    def __init__(self):

        # proj & output dir
        self.proj_dir = os.path.dirname(os.getcwd()) + '/'
        self.output_dir = self.proj_dir + 'train_with_val2/'
        # dataset dir
        self.dataset = 'SEL'  # ['SEL', 'SEL_Hard']
        if self.dataset == 'SEL':
            self.dataset_dir = '/home/shinebobo/Semantic-Line-SLNet/code/newpickles/'
            self.img_dir = self.dataset_dir + 'images/'
        elif self.dataset == 'SEL_Hard':
            self.dataset_dir = '/media/dkjin/3535ee90-f88e-4f09-b6aa-cfbf15169dde' \
                               '/Github/Semantic-Line-DRM/Semantic_line_detection/preprocessed/output/SEL_Hard/'
            self.img_dir = self.dataset_dir + 'images/'

        # other dir
        self.pickle_dir = self.dataset_dir
        self.weight_dir = self.output_dir + 'train/weight/'
        self.paper_weight_dir = '/home/shinebobo/Semantic-Line-SLNet/paper_weight/'

        # setting for train & test
        self.run_mode = 'train'  # ['train', 'test', 'test_paper']
        self.resume = True

        self.gpu_id = "0"
        self.seed = 123
        self.num_workers = 4
        self.epochs = 40
        self.ratio_left = 0.4
        self.ratio_mid = 0.2
        self.ratio_right = 0.4
        self.ratio_neg = 0
        self.batch_size = {'img': 1,
                           'train_line': 50,
                           'test_line': 10}
        self.batch_test_size = {'img': 1,
                           'train_line': 10,
                           'test_line': 200}
        self.test_size=1
        self.ratio_val=0.2

        # optimizer
        self.lr = 1e-5
        self.milestones = [40, 80, 120, 160, 200, 240, 280]
        self.weight_decay = 5e-4
        self.gamma = 0.5

        #wandb
        wandb.config.epochs=self.epochs
        wandb.config.lr=self.lr
        wandb.config.weight_decay=self.weight_decay
        wandb.config.gamma=self.gamma
        wandb.config.batch_size=self.batch_size


        # other setting
        self.height = 80
        self.width = 60
        # changed before 400 400
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


        # option for visualization
        self.draw_auc_graph = True
