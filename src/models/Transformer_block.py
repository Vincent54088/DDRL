import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
import math
# from models.network import gaussian_weights_init

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    # m.weight.data.normal_(0.0, 0.02)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class RLN(nn.Module):
	r"""Revised LayerNorm"""
	def __init__(self, dim, eps=1e-5, detach_grad=False):
		super(RLN, self).__init__()
		self.eps = eps
		self.detach_grad = detach_grad

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

		self.meta1 = nn.Conv2d(1, dim, 1)
		self.meta2 = nn.Conv2d(1, dim, 1)

		trunc_normal_(self.meta1.weight, std=0.02)
		nn.init.constant_(self.meta1.bias, 1)

		trunc_normal_(self.meta2.weight, std=0.02)
		nn.init.constant_(self.meta2.bias, 0)

	def forward(self, input):
		mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
		std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

		normalized_input = (input - mean) / std

		if self.detach_grad:
			rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
		else:
			rescale, rebias = self.meta1(std), self.meta2(mean)

		out = normalized_input * self.weight + self.bias
		return out, rescale, rebias


def window_partition(x, window_size):
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)#Size(b,33,8,33,8,C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)#(b,33,33,8,8,C)->(b*33*33,64,C)
	return windows


def window_reverse(windows, window_size, H, W):
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x


def get_relative_positions(window_size):
	coords_h = torch.arange(window_size)
	coords_w = torch.arange(window_size)

	coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='ij'))  # 2, Wh, Ww
	coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
	relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

	relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
	relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

	return relative_positions_log

class Mlp(nn.Module):
	def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.network_depth = network_depth

		self.mlp = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1),
			nn.ReLU(True),
			nn.Conv2d(hidden_features, out_features, 1)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.network_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.mlp(x)

class WindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		relative_positions = get_relative_positions(self.window_size)
		self.register_buffer("relative_positions", relative_positions)
		self.meta = nn.Sequential(
			nn.Linear(2, 256, bias=True),
			nn.ReLU(True),
			nn.Linear(256, num_heads, bias=True)
		)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, qkv):# nW*B, window_size**2, C
		B_, N, _ = qkv.shape

		qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
		#(3,B_,2,N,12)
		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))#(B_,2,N,N)

		relative_position_bias = self.meta(self.relative_positions)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nHead, Wh*Ww, Wh*Ww
		attn = attn + relative_position_bias.unsqueeze(0)#(B_,2,64,64)

		attn = self.softmax(attn)#(B_,2,64,64)

		x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)#(B_,2,64,12)->(B_, 64, self.dim)
		return x

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

class Attention(nn.Module):
	def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
		super().__init__()
		self.dim = dim
		self.head_dim = int(dim // num_heads)#24/2
		self.num_heads = num_heads

		self.window_size = window_size
		self.shift_size = shift_size

		self.network_depth = network_depth
		self.use_attn = use_attn
		self.conv_type = conv_type

		if self.conv_type == 'Conv':
			self.conv = nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
				nn.ReLU(True),
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
			)

		if self.conv_type == 'DWConv':
			self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

		if self.conv_type == 'DWConv' or self.use_attn:
			self.V = nn.Conv2d(dim, dim, 1)
			self.proj = nn.Conv2d(dim, dim, 1)

		if self.use_attn:
			self.QK = nn.Conv2d(dim, dim * 2, 1)
			self.attn = WindowAttention(dim, window_size, num_heads)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			w_shape = m.weight.shape
			
			if w_shape[0] == self.dim * 2:	# QK
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)		
			else:
				gain = (8 * self.network_depth) ** (-1/4)
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def check_size(self, x, shift=False):
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

		if shift:
			x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
						  self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
		else:
			x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward(self, X):
		B, C, H, W = X.shape 

		if self.conv_type == 'DWConv' or self.use_attn:
			V = self.V(X) 

		if self.use_attn:
			QK = self.QK(X)
			QKV = torch.cat([QK, V], dim=1)

			# shift
			shifted_QKV = self.check_size(QKV, self.shift_size > 0)
			Ht, Wt = shifted_QKV.shape[2:]

			# partition windows
			shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
			qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

			attn_windows = self.attn(qkv) # nW*B, window_size**2, dim

			# merge windows
			shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

			# reverse cyclic shift
			out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
			attn_out = out.permute(0, 3, 1, 2)

			if self.conv_type in ['Conv', 'DWConv']:
				conv_out = self.conv(V)
				out = self.proj(conv_out + attn_out)
			else:
				out = self.proj(attn_out)

		else:
			if self.conv_type == 'Conv':
				out = self.conv(X)				# no attention and use conv, no projection
			elif self.conv_type == 'DWConv':
				out = self.proj(self.conv(V))

		return out


class TransformerBlock(nn.Module):
	def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
				 norm_layer=RLN, mlp_norm=False,
				 window_size=8, shift_size=0, use_attn=True, conv_type=None):
		super().__init__()
		self.use_attn = use_attn
		self.mlp_norm = mlp_norm

		self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
		self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
							  shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

		self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
		self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

	def forward(self, x):
		identity = x
		if self.use_attn: x, rescale, rebias = self.norm1(x)
		x = self.attn(x)
		if self.use_attn: x = x * rescale + rebias
		x = identity + x

		identity = x
		if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
		x = self.mlp(x)
		if self.use_attn and self.mlp_norm: x = x * rescale + rebias
		x = identity + x
		return x


class ResidualLayer(nn.Module):
	def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
				 norm_layer=RLN, window_size=8,
				 attn_ratio=0., attn_loc='last', conv_type='DWConv'):

		super().__init__()
		self.dim = dim#24
		self.depth = depth#8

		attn_depth = attn_ratio * depth#2

		if attn_loc == 'last':
			use_attns = [i >= depth-attn_depth for i in range(depth)]
		elif attn_loc == 'first':
			use_attns = [i < attn_depth for i in range(depth)]
		elif attn_loc == 'middle':
			use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

		# build blocks
		self.blocks = nn.ModuleList([
			TransformerBlock(network_depth=network_depth,
							 dim=dim, 
							 num_heads=num_heads,
							 mlp_ratio=mlp_ratio,
							 norm_layer=norm_layer,
							 window_size=window_size,
							 shift_size=0 if (i % 2 == 0) else window_size // 2,
							 use_attn=use_attns[i], conv_type=conv_type)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class PatchEmbed(nn.Module):
	def __init__(self, patch_size=2, in_chans=3, embed_dim=64, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')
		
	def forward(self, x):
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=2, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
					  padding=kernel_size//2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)
		

	def forward(self, x):
		x = self.proj(x)
		return x

class SKFusion(nn.Module):
	def __init__(self,dim,reduction = 8):
		super(SKFusion,self).__init__()
		self.h = 2
		d = max(int(dim/reduction),4)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(dim//self.h,d,1,bias=False),
			nn.ReLU(True),
			nn.Conv2d(d,dim,1,bias=False)
		)
		self.softmax = nn.Softmax(dim=1)
		self.f1 = nn.Conv2d(dim//2,dim//2,kernel_size=3,stride=1,padding=1)
		self.f2 = nn.Conv2d(dim//2,dim//2,kernel_size=5,stride=1,padding=2)
		
	
	def forward(self,featA):
		'''
		featA:Size(B,C,H,W)
		'''
		B,C,H,W = featA.shape
		F1,F2 = torch.split(featA,C//2,dim=1)
		F1 = self.f1(F1) # (B,C//2,H,W)
		F2 = self.f2(F2) # (B,C//2,H,W)
		
		feats = torch.cat([F1,F2],dim=1)
		feats = feats.view(B,self.h,C//self.h,H,W)
		
		feats_sum = torch.sum(feats,dim=1) # B,C//2,H,W
		attn = self.mlp(self.avg_pool(feats_sum)) # B,C,1,1
		attn = self.softmax(attn.view(B,self.h,C//self.h,1,1))
		out = torch.sum(feats*attn,dim = 1) # B,C//2,H,W

		out = out.view(B,C*2,H//2,W//2)
		return out

class SKFusion2(nn.Module):
	def __init__(self,dim,reduction = 8):
		super(SKFusion2,self).__init__()
		self.h = 2
		d = max(int(dim/reduction),4)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(dim//self.h,d,1,bias=False),
			nn.ReLU(True),
			nn.Conv2d(d,dim,1,bias=False)
		)
		self.softmax = nn.Softmax(dim=1)
		self.f1 = nn.Conv2d(dim,dim//2,kernel_size=3,stride=1,padding=1)
		self.f2 = nn.Conv2d(dim,dim//2,kernel_size=5,stride=1,padding=2)

		# self.apply(gaussian_weights_init)
	def forward(self,featA):
		'''
		featA:Size(B,C,H,W)
		'''
		B,C,H,W = featA.shape
		
		F1 = self.f1(featA) # (B,C//2,H,W)
		F2 = self.f2(featA) # (B,C//2,H,W)
		
		feats = torch.cat([F1,F2],dim=1)
		feats = feats.view(B,self.h,C//self.h,H,W)
		
		feats_sum = torch.sum(feats,dim=1) # B,C//2,H,W
		attn = self.mlp(self.avg_pool(feats_sum)) # B,C,1,1
		attn = self.softmax(attn.view(B,self.h,C//self.h,1,1))
		out = torch.sum(feats*attn,dim = 1) # B,C//2,H,W

		out = out.view(B,C*2,H//2,W//2)
		return out

if __name__ == '__main__':
	embed_dims=[24, 48, 96, 48, 24]
	mlp_ratios=[2., 4., 4., 2., 2.]
	depths=[8, 8, 8, 4, 4]
	num_heads=[2, 4, 6, 1, 1]
	attn_ratio=[1/4, 1/2, 3/4, 0, 0]
	i=0
	conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv']
	net = ResidualLayer(network_depth=sum(depths), dim=embed_dims[i], depth=depths[i],
					   			 num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
					   			 attn_ratio=attn_ratio[i], attn_loc='last', conv_type=conv_type[i])
	x = torch.randn(2,64,256,256)
	out = net(x)
	print('out',out.size())
	from torchsummary import summary
	summary(net.cuda(),input_size=(64,256,256))
	
