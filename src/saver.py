import os
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# tensor to PIL Image
def tensor2img(img):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
  # img = (np.transpose(img, (1, 2, 0))) * 255.0
  return img.astype(np.uint8)

# save a set of images
def save_imgs(imgs, names, path):
  if not os.path.exists(path):
    os.mkdir(path)
  for img, _ in zip(imgs, names):
    # print(names)
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(os.path.join(path, names + '.png'))

class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.display_imgs_dir = os.path.join(opts.display_dir, 'images')
    self.model_dir = os.path.join(opts.result_dir, 'Mix1')
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.display_freq = opts.display_freq
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq

    # make directory
    if not os.path.exists(self.display_dir):
      os.makedirs(self.display_dir)
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
      os.makedirs(self.image_dir)
    if not os.path.exists(self.display_imgs_dir):
      os.makedirs(self.display_imgs_dir)

    # create tensorboard writer
    self.writer = SummaryWriter(log_dir=self.display_dir)
    self.writer2 = SummaryWriter(log_dir=self.display_imgs_dir)


  # write losses and images to tensorboard
  def write_display(self, total_it, model):
    
    if (total_it + 1) % self.display_freq == 0:
      # write loss
      members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
      for m in members:
        self.writer.add_scalar(m, getattr(model, m), total_it)
    if (total_it + 1) % self.img_save_freq == 0:
      # write img
      image_dis = torchvision.utils.make_grid(model.image_display, nrow=model.image_display.size(0)//2)/2 + 0.5
      self.writer2.add_image('Image', image_dis, total_it)

  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % 10 == 0:
      assembled_images = model.assemble_outputs()
      img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
      torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
    elif ep == -1:
      assembled_images = model.assemble_outputs()
      img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
      torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

  # save model
  def write_model(self, ep, total_it, model,type = 0):
    if type == -1:
      model.save_model('%s/best_Mix.pth' % self.model_dir, ep, total_it)
    elif ep % self.model_save_freq == 0:
      print('--- save the model @ ep %d ---' % (ep))
      model.save_model('%s.pth' % self.model_dir, ep, total_it)
    
