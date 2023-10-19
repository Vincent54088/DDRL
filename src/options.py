import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument("--data_path", default="/root/FJY_project/dataset/data", help="data folder path")
    self.parser.add_argument("--test_path", default="/root/FJY_project/dataset", help="data folder path")
    self.parser.add_argument('--patch_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    self.parser.add_argument('--workers', type=int, default=2, help='# of threads for data loader')
    self.parser.add_argument('--nz', type=int, default=24, help='the dimension of style latent vector')
    self.parser.add_argument('--tch', type=int, default=24, help='the dimension of content channel')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='./log', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='./PTH', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=100, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=1000, help='freq (iter) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=2, help='freq (epoch) of saving models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    self.parser.add_argument("-lr", "--lr", default=0.0002, type=float, metavar="LR", help="initial learning rate (default le-5)")
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--concat', type=bool, default=True, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--disc_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--disc_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--disc_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay [lambda,step]')
    self.parser.add_argument('--n_ep', type=int, default=200, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=30, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default='', help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=5, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--mi_iter', type=int, default=4, help='# of iterations for updating MI estimintor')
    self.parser.add_argument('--gpu', type=int, default=1, help='gpu')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument("--test-foggyImage-folder", default="/root/FJY_project/dataset/multiScences", help="data folder path")
        self.parser.add_argument("--test-FreefoggyImage-folder", default="/root/FJY_project/dataset/gtScences", help="data folder path")

        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size for testing')
        self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
        self.parser.add_argument('--tch', type=int, default=24, help='cropped image size for training')
        self.parser.add_argument('--workers', type=int, default=2, help='for data loader')
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')
        self.parser.add_argument('--nz', type=int, default=24, help='the dimension of style latent vector')

        # ouptput related
        self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
        self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str, default='./result', help='path for saving result images and models')

        # model related
        self.parser.add_argument('--concat', type=bool, default=True, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
        self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
        self.parser.add_argument('--resume', type=str, default='./PTH/Mix.pth', help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=1, help='gpu')
        self.parser.add_argument('--disc_scale', type=int, default=3, help='scale of discriminator')
        self.parser.add_argument('--disc_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--disc_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
        

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        # set irrelevant options
        self.opt.dis_scale = 3
        self.opt.dis_norm = 'None'
        self.opt.dis_spectral_norm = False
        return self.opt
