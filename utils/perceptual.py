import torch
import torch.nn as nn
from torchvision.models import vgg19
# 计算特征提取模块的感知损失
def vgg16_loss(feature_module,loss_func,y,y_):
    out=feature_module(y)
    out_=feature_module(y_)
    loss=loss_func(out,out_)
    return loss

class PerceptualLoss(nn.Module):
    def __init__(self,pth_path,loss_func,layer_indexs=None,device=None):
        super(PerceptualLoss, self).__init__()
        self.creation=loss_func
        self.layer_indexs=layer_indexs
        self.device=device

        self.vgg = vgg19(pretrained=False)
        checkpoint = torch.load(pth_path)
        self.vgg.load_state_dict(checkpoint)
        self.vgg = self.vgg.features.to(device)
        self.vgg.eval()
        # 冻结参数
        for parm in self.vgg.parameters():
            parm.requires_grad = False
        # self.vgg = self.vgg[0:self.layer_indexs + 1].to(device)

    def forward(self,y,y_):
        loss=0
        for index in self.layer_indexs:
            feature_module=self.vgg[0:index + 1]
            loss+=vgg16_loss(feature_module,self.creation,y,y_)
        return loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.ones((1, 3, 256, 256))
    y = torch.zeros((1, 3, 256, 256))
    x,y=x.to(device),y.to(device)

    layer_indexs = [3, 8, 15, 22]
    # 基础损失函数：确定使用那种方式构成感知损失，比如MSE、MAE
    loss_func = nn.MSELoss().to(device)
    # 感知损失
    per = PerceptualLoss('./PTH/vgg/vgg19.pth',loss_func, layer_indexs, device)
    perceptual_loss=per(x,y)
    print(perceptual_loss)
