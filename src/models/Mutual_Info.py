import torch
import torch.nn as nn 
import torch.nn.functional as F


#Minimize mutual loss of style and content
class CLUB(nn.Module):  #CLUB: Mutual Information Contrastive learning Upper Bound
    '''
    This class provide the CLUB estimation to I(X,Y)
    @param X_dim : the channels of Content  feature Map. Size(b,256,64,64)
    @param Y_dim : the dim of latent style feature (8) 
    @param hidden_size : the dimension of the hidden layer of the approximation network q(Y|X)
    '''
    def __init__(self,X_dim,Y_dim,hidden_size = 1024,map_size = 64):
        super().__init__()
        self.net_x = nn.Sequential(
            nn.Conv2d(X_dim,X_dim,kernel_size=3,stride=2,padding=1),#32
            nn.BatchNorm2d(X_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(X_dim,X_dim,kernel_size=3,stride=2,padding=1),#16
            nn.BatchNorm2d(X_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(X_dim,X_dim,kernel_size=3,stride=2,padding=1),#8
            nn.BatchNorm2d(X_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(X_dim,X_dim,kernel_size=3,stride=2,padding=1),#4
            nn.Flatten()#Size(b,96*4*4)
        )
        map_size = map_size // 16 #16
        #p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(
            nn.Linear(X_dim*map_size*map_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,Y_dim),
        )
        #p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(
            nn.Linear(X_dim * map_size*map_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,Y_dim),
            nn.Tanh()
        )
        

    def get_mu_logvar(self,X_samples):
        z_x = self.net_x(X_samples)
        mu = self.p_mu(z_x)
        logvar = self.p_logvar(z_x)
        return mu,logvar

    def forward(self,X,Y):
        '''
        @param X : latent content feature map. 
        @param Y : latent style feature map. 
        '''
        mu,logvar = self.get_mu_logvar(X)

        batch_size = X.size(0)
        random_index = torch.randperm(batch_size).long()

        #log of conditional probability of positive sample pairs
        positive = -(mu - Y)**2 /2./logvar.exp()
        # positive = -(mu - Y)**2 /2.
        
        # mu = mu.unsqueeze(1)
        # Y = Y.unsqueeze(0)
        #log of conditional probability of negative sample pairs
        #(mu - neg_sample) ：calculate the probability of q(Y_j|X_i) Size([b,b,Y_dim])  
        # negative = -((mu - Y)**2).mean(dim=1)/2./logvar.exp() #Size([b,Y_dim])
        negative = -((mu - Y[random_index])**2)/2./logvar.exp()
        # negative = -((mu - Y)**2).mean(dim=1)/2.
        
        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean() #upper bound
    
    def loglikeli(self,X,Y):
        mu,logvar = self.get_mu_logvar(X)
        return (-(mu - Y)**2 /logvar.exp() - logvar).sum(dim = 1).mean(dim = 0)
        # return (-(mu - Y)**2).sum(dim = 1).mean(dim = 0)
        
    
    def learning_loss(self,X,Y):
        return -self.loglikeli(X,Y)

