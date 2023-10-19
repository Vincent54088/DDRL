import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch
from torch.utils.data.dataloader import DataLoader
# import numpy as np
# import cv2
# import torch.nn.functional as F
import random
from options import TrainOptions
from Data_Process.dataset import DataLoaderTrain
from models.Disentangle_Model import DisentangleNet
from saver import Saver
from utils.metric import batch_PSNR

def test(opts,dataloader,model):
    PSNR = 0.0
    model = model.eval()
    cnt = 0
    
    for idx,(input_a, input_b,_) in enumerate(dataloader):
        cnt += 1
        input_a = input_a.cuda(opts.gpu)
        input_b = input_b.cuda(opts.gpu)
        with torch.no_grad():
            fog_content_z = model.enc_c.forward_a(input_a)          
            style_mu,_ = model.enc_s.forward_a(input_b)            
            output = model.gen.forward_b(fog_content_z,style_mu)         
            psnr = batch_PSNR(output*0.5+0.5, input_b*0.5+0.5, 1.)        
            PSNR = PSNR + psnr

    return PSNR/cnt
        


def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()
    
    # daita loader
    print('\n--- load dataset ---')
    train_dataset = DataLoaderTrain(opts.data_path,patch_size=opts.patch_size,mode='train')
    test_dataset = DataLoaderTrain(opts.test_path,patch_size=opts.patch_size,mode='test')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opts.workers)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers)

    # model
    print('\n--- load model ---')
    model = DisentangleNet(opts)
    model.set_gpu(opts.gpu)
    if opts.resume == '':
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opts)
    # train
    print('\n--- train ---')
    best_psnr = 0.0
    X_A = None
    X_B = None
    
    # seed = 1024
    # random.seed(seed)
    # torch.manual_seed(seed)
    for ep in range(ep0, opts.n_ep):
        model = model.train()
        for it, (images_a, images_b,name) in enumerate(train_loader):
            if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
                continue

            # input data
            images_a = images_a.cuda(opts.gpu).detach()
            images_b = images_b.cuda(opts.gpu).detach()

            # update model
            if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                model.update_D_content(images_a,images_b,name)

                # X_A = torch.cat([X_A,images_a],dim = 0) if X_A != None else images_a
                # X_B = torch.cat([X_B,images_b],dim = 0) if X_B != None else images_b
                # if X_A.size(0) == 8:
                #     model.update_MI_estimator(X_A,X_B)
                #     X_A = None
                #     X_B = None
                continue
            else:
                # print('update EG')
                model.update_D(images_a, images_b)
                model.update_EG()
                

            # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            print('total_it: %d (ep %d, it %d), lr %08f best_psnr %04f' % (total_it, ep, it, model.gen_optimizer.param_groups[0]['lr'],best_psnr))
            total_it += 1

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        psnr = test(opts,test_loader,model)
        print('avg psnr:{}'.format(psnr))
        if psnr > best_psnr:
            saver.write_model(ep, total_it,model,type=-1)
            best_psnr = psnr

        # Save network weights
        saver.write_model(ep, total_it, model)
    return

if __name__ == '__main__':
    main()
