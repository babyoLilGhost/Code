from sys import platform
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from time import time
from einops import rearrange
import numbers
from torch.nn import init
from argparse import ArgumentParser
import platform
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils import evaluate, transform
parser = ArgumentParser(description='DPSU-Net')

parser.add_argument('--epoch_num', type=int, default=200, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0,1', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--data_path', type=str, default='T1', help='Path to the dataset')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='BrainImages_test', help='name of test set')
parser.add_argument('--lambda_index', type=int, default=5, help='from {0, 1, 2, 3, 4, 5}')
args = parser.parse_args()


batch_size = 1
epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
lambda_index = args.lambda_index

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
desired_sparsity = cs_ratio / 100.
sparse_ratio_ = 1.0 - cs_ratio / 100.

w1 = torch.tensor([2.0])
w2 = torch.tensor([3.0])
w3 = torch.tensor([2.0])
w4 = torch.tensor([3.0])
lambda_list = {0:0.002, 1:0.001, 2:0.0005, 3:0.0001, 4:0.00005, 5:0.00001}
Testing_data_Name = 'test.mat'
Testing_labels = sio.loadmat('/code/%s' % (Testing_data_Name))
nrtrain = Testing_labels.shape[0] 


class Lambda_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(Lambda_Conv, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.fc2 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)

    def forward(self, x, lam):
        lam = lam.unsqueeze(-1).unsqueeze(-1)
        s = self.fc1(lam)
        s = F.softplus(s)
        b = self.fc2(lam)
        x = s*self.conv(x)+b
        return x

class Attention_SEblock(nn.Module):
    def __init__(self, channels, reduction):
        super(Attention_SEblock, self).__init__()
        self.conv = Lambda_Conv(6, channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2)
        self.fc3 = nn.Linear(channels // reduction, 2)
        self.fc2.bias.data[0] = 0.1 
        self.fc2.bias.data[1] = 2
        self.fc3.bias.data[0] = 0.1
        self.fc3.bias.data[1] = 2
        self.channels = channels
    def forward(self, x, lam):
        x = self.conv(x, lam)
        x = self.avg_pool(x).view(-1, self.channels)
        x = self.fc1(x)
        x = self.relu(x)
        x_2 = self.fc2(x)        
        x_2 = F.gumbel_softmax(x_2, tau=1, hard=True)
        x_3 = self.fc3(x)        
        x_3 = F.gumbel_softmax(x_3, tau=1, hard=True)
        return x_2, x_3

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def zero_filled(x, mask, mod=False, norm=False):
    x_dim_0 = x.shape[0]
    x_dim_1 = x.shape[1]
    x_dim_2 = x.shape[2]
    x_dim_3 = x.shape[3]
    x = x.view(-1, x_dim_2, x_dim_3, 1)

    x_real = x
    x_imag = torch.zeros_like(x_real)
    x_complex = torch.cat([x_real, x_imag], 3)

    x_kspace = torch.fft.fft2(x_complex)
    y_kspace = x_kspace * mask
    xu = torch.fft.ifft2(y_kspace)

    if not mod:
        xu_ret = xu[:, :, :, 0:1]
    else:
        xu_ret = torch.sqrt(xu[..., 0:1] ** 2 + xu[..., 1:2] ** 2)

    xu_ret = xu_ret.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
    xu_ret = xu_ret.float()

    return xu_ret

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sigmoid(input * t)  
        out = (out >= 0.5).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t * 2), 2)) * grad_output 
        return grad_input, None, None, None

class blockNL(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)   
        self.norm_x = LayerNorm(1, 'WithBias') 
        self.norm_z = LayerNorm(31, 'WithBias') 

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels+1, out_channels=self.channels+1, kernel_size=1, stride=1, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
        )
        self.w3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w4 = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x, z, w3, w4):
        b, c, h, w = x.shape
        x0 = self.norm_x(x)  
        z0 = self.norm_z(z) 
        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1) 
        x1 = self.p(x0)  
        x1 = x1.view(b, c, -1) 
        x2 = self.g1(x0)
        x_v = x2.view(b, c, -1) 
        z2 = self.g2(z0) 
        z_v = z2.view(b, c, -1) 
        num_heads = 4  
        x1_heads = x1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        z1_heads = z1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        z_v_heads = z_v.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        x_v_heads = x_v.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        x1_heads = torch.nn.functional.normalize(x1_heads, dim=-1)
        z1_heads = torch.nn.functional.normalize(z1_heads, dim=-1)
        x_t_heads = x1_heads.permute(0, 1, 3, 2)  
        att_heads = torch.matmul(z1_heads, x_t_heads)  
        att_heads = self.softmax(att_heads)  
        v_heads = self.w3*z_v_heads+self.w4*x_v_heads
        out_x_heads = torch.matmul(att_heads, v_heads)  
        out_x_heads = out_x_heads.view(b, c, h, w)  
        out_x_heads = self.w(out_x_heads) + self.pos_emb(z2) + z 
        y = self.v(torch.cat([x, out_x_heads], 1)) 
        return y

class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()
               
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_qv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.w1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w2 = nn.Parameter(torch.randn(1, requires_grad=True))
    
    def forward(self, pre, cur, w1, w2):
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q,v1 = self.conv_qv1(cur_ln).chunk(2, dim=1)
        q = q.view(b, c, -1)  
        v1 = v1.view(b, c, -1)
        k, v2 = self.conv_kv(pre_ln).chunk(2, dim=1)  
        k = k.view(b, c, -1)
        v2 = v2.view(b, c, -1)
        num_heads = 4  
        q = q.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        k = k.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        v1 = v1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        v2 = v2.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 1, 3, 2))  
        att = self.softmax(att) 
        v = self.w1*v1+self.w2*v2      
        out = torch.matmul(att, v)  
        out = out.permute(0, 2, 1, 3).contiguous().view(b, c, h, w)  
        out = self.conv_out(out) + cur

        return out

class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31) 
        self.nonlo = blockNL(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
         
    def forward(self, x, z_pre, z_cur, mask, PhiTb, gate1, gate2):
        x_gate1 = gate1[:,1].view(-1, 1, 1, 1)
        x_gate2 = gate2[:,1].view(-1, 1, 1, 1)
        z = self.atten(z_pre, z_cur, w1, w2)
        x = x - self.lambda_step * x_gate1 * zero_filled(x, mask) 
        x_input = x + self.lambda_step * x_gate1 * PhiTb
        x_input = self.nonlo(x_input, z, w3, w4) 
        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_backward = F.conv2d(x_backward, self.conv_G, padding=1)
        
        x_pred = x_input + x_backward * x_gate2

        return x_pred

class RandomSmooth(nn.Module):
    def __init__(self, input_shape):
        super(RandomSmooth, self).__init__()
        self.noise_generator = nn.Sequential(
            nn.Linear(input_shape, 64), 
            nn.ReLU(),
            nn.Linear(64, input_shape)
        )
    def forward(self, x):
        noise = self.noise_generator(x) 
        return torch.clamp(x + noise, min=-1, max=1)  

class OCTUF(torch.nn.Module):
    def __init__(self, LayerNo, desired_sparsity):
        super(OCTUF, self).__init__()
        onelayer = []
        gates = []
        self.LayerNo = LayerNo
        self.patch_size = 32
        self.desired_sparsity = desired_sparsity
        self.pmask_slope = 5
        self.MyBinarize = BinaryQuantize.apply
        self.threshold = nn.Parameter(torch.tensor([0.5]))
        self.k = torch.tensor([10]).float().to(device)
        self.t = torch.tensor([0.1]).float().to(device)
        self.mask_shape = (256, 256)
        self.Phi = nn.Parameter(self.initialize_p())
        for i in range(LayerNo):
            onelayer.append(BasicBlock())        
        for i in range(LayerNo):
            gates.append(Attention_SEblock(32, 4))
        self.gates = nn.ModuleList(gates)
        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(1, 31, 3, padding=1, bias=True)
        self.fe2 = nn.Conv2d(1, 31, 3, padding=1, bias=True)
        self.random_smooth = RandomSmooth(256)

    def forward(self, x, lamda):      
        maskp0 = torch.sigmoid(self.pmask_slope * self.Phi) 
        maskpbar = torch.mean(maskp0) 
        r = self.desired_sparsity / maskpbar 
        beta = (1 - self.desired_sparsity) / (1 - maskpbar)
        le = torch.le(r, 1).float()
        maskp = le * maskp0 * r + (1 - le) * (1 - (1 - maskp0) * beta)
        u = torch.from_numpy(np.random.uniform(low=0.0, high=1.0, size=maskp0.size())).type(dtype)
        mask_matrix = self.MyBinarize(maskp - u, self.k, self.t) 
        mask = mask_matrix.unsqueeze(0).unsqueeze(-1) 
        xu_real = zero_filled(x, mask)
        x = xu_real
        x = self.random_smooth(x)      
        z_pre = self.fe(x)
        z_cur = self.fe2(x)
        lamda = lamda.repeat(x.shape[0], 1).type(torch.FloatTensor).to(device)
        gate1_s = []
        gate2_s = []
        for i in range(self.LayerNo):
            if i==0:
                gate1, gate2 = self.gates[i](torch.cat([x, z_cur], 1), lamda)
            else:
                gate1, gate2 = self.gates[i](x_dual, lamda)
            x_dual = self.fcs[i](x, z_pre, z_cur, mask, xu_real, gate1, gate2)
            x = x_dual[:, :1, :, :]
            z_pre = z_cur
            z_cur = x_dual[:, 1:, :, :]
            gate1_s.append(gate1[:,1])
            gate2_s.append(gate2[:,1])

        x_final = x

        return x_final, gate1_s, gate2_s
    
    def initialize_p(self, eps=0.01):
        x = torch.from_numpy(np.random.uniform(low=eps, high=1-eps, size=self.mask_shape)).type(dtype)
        return - torch.log(1. / x - 1.) / self.pmask_slope

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Testing_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Testing_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

model = OCTUF(layer_num, desired_sparsity)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model_dir = "/%s/MRI_DPSU_Net_layer_%d_group_%d_ratio_%d" % (args.model_dir , layer_num, group_num, cs_ratio)
model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch_num)))

ImgNum = nrtrain
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
gate1_ALL = np.zeros([1, ImgNum], dtype=np.float32)
gate2_ALL = np.zeros([1, ImgNum], dtype=np.float32)

print('\n')
print("MRI CS Reconstruction Start")

with torch.no_grad():
    for img_no, data in enumerate(rand_loader):
        batch_x = data
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
        Iorg = batch_x
        Iorg = Iorg.cpu().data.numpy().reshape(256, 256)
        start = time()
        batch_x, mean, std = transform.normalize_instance(batch_x, eps=1e-11)
        batch_x = batch_x.clamp(-6, 6)
        lamblist_encoding = torch.nn.functional.one_hot(torch.tensor([0,1,2,3,4,5]))
        lambda_encoding = lamblist_encoding[lambda_index]
        lambda_value = lambda_list[lambda_index]
        x_output, gate1_s, gate2_s = model(batch_x, lambda_encoding)
        end = time()
        mean = mean.cpu().data.numpy()
        std = std.cpu().data.numpy()
        
        gate1_s = torch.tensor(gate1_s)
        gate2_s = torch.tensor(gate2_s)
        gates1 = gate1_s.cpu().data.numpy().squeeze()
        gates2 = gate2_s.cpu().data.numpy().squeeze()
      
      
        Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)

        X_rec = Prediction_value.astype(np.float64)
        rec_PSNR = evaluate.psnr(X_rec * std + mean, Iorg.astype(np.float64))
        rec_SSIM = evaluate.ssim(X_rec * std + mean, Iorg.astype(np.float64))
        
        
        print("[%02d/%02d] Run time is %.4f, sum_gate1 is %d, sum_gate2 is %d, Proposed PSNR is %.2f, Proposed SSIM is %.4f" % (
                img_no, ImgNum, (end - start), np.sum(gates1), np.sum(gates2),  rec_PSNR, rec_SSIM))       

        im_rec_rgb = (X_rec * std + mean).astype(np.float64)

        scipy.io.savemat("/code/%s/Vision_Net_Rec/[%02d] DPSU_Rec_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.mat"
                         % (args.result_dir,img_no,cs_ratio, epoch_num, rec_PSNR, rec_SSIM),{"data":im_rec_rgb}) 
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

print("std_psnr %.2f" % np.std(PSNR_All))
print("std_ssim %.4f" % np.std(SSIM_All))

print('\n')
output_data = "CS ratio is %d, lambda is %.5f, Avg Proposed PSNR/SSIM for %s is %.2f/%.4f,  Avg gate1 is %d, sum_gate1 is %d, Avg gate2 is %d, sum_gate2 is %d, Epoch number of model is %d \n" % (
    cs_ratio, lambda_list[lambda_index], args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(gate1_ALL), np.sum(gates1), np.mean(gate2_ALL), np.sum(gates2), epoch_num)
print(output_data)

output_file_name = "/%s/PSNR_SSIM_Results_MRI_DPSU_layer_%d_group_%d_ratio_%d.txt" % (
    args.log_dir, layer_num, group_num, cs_ratio)
output_file = open(output_file_name, 'a')
output_file.write(output_data)
output_file.close()

print("MRI CS Reconstruction End")