import time
import torch
from torch import nn, Tensor
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from scaled.dataset.sfc_dataset import SFCDiffusionDatasetv1
import matplotlib.pyplot as plt
from scaled.model.unets.unet_1ds import UNet1DsModel
from matplotlib import cm
import warnings
import numpy as np
import os
import os.path as osp
from torch.distributions import Independent, Normal
from collections import OrderedDict
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

class WrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
            return self.model(x, t,return_dict=False)[0]

def validation(model,dataset,path):
    wrapped_vf = WrappedModel(model)
    #step size for ode solver
    data_0,data_1 = dataset[0]
    data_0 = data_0.unsqueeze(0).to(device)
    data_1 = data_1.unsqueeze(0).to(device)
    step_size = 0.05
    T = torch.linspace(0,1,10)
    T = T.to(device=device)
    solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
    sol = solver.sample(time_grid=T, x_init=data_0, method='midpoint', step_size=step_size, return_intermediates=True)  # sample from the model
    sol = sol.cpu().numpy()
    T = T.cpu()
    fig, axs = plt.subplots(2,10,figsize=(20,20))
    data_1 = data_1[0][0].cpu().numpy()
    print(data_1.shape)
    data_1 = data_1.reshape((257,80))
    for i in range(10):
        result = sol[i][0][0]
        result = result.reshape((257,80))
        diff = np.abs(data_1-result)
        a = axs[0,i].imshow(result,vmax=0.5,vmin=-0.5)
        b = axs[1,i].imshow(diff,vmax=0.1,vmin=-0.1)
        axs[0,i].set_aspect('equal')
        axs[0,i].axis('off')
        axs[0,i].set_title('t= %.2f' % (T[i]))
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size, ), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)
    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45
    return data.float()

# training arguments
lr = 0.00001
batch_size = 12
print_every = 1000
hidden_dim = 512
epoch = 10000

# velocity field model init

model  = UNet1DsModel(
        in_channels=2,
        out_channels=2,
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
        block_out_channels=(128, 256, 384, 512),
        add_attention=False,
        ).to(device)

model.load_state_dict(torch.load('exp_output/scaled_flowmatching_sfc/model_9000.pth',map_location='cpu'))
path = AffineProbPath(scheduler=CondOTScheduler())
optim = torch.optim.Adam(model.parameters(), lr=lr)

train_dataset = SFCDiffusionDatasetv1(
        data_dir="data/SFC/SFC_data_csv",
        data_list=[i for i in range(5, 3500)])
val_dataset  = SFCDiffusionDatasetv1(
    data_dir="data/SFC/SFC_data_csv",
    data_list=[i for i in range(3500,3990)])


train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )


# resume_from_checkpoint = 'latest'
# # Potentially load in the weights and states from a previous save
# if resume_from_checkpoint:
#     if resume_from_checkpoint != "latest":
#         resume_dir = resume_from_checkpoint
#     else:
#         resume_dir = 'exp_output/scaled_flowmatching_sfc'
#     # Get the most recent checkpoint
#     dirs_ = os.listdir(resume_dir)
#     dirs = [d for d in dirs_ if d.startswith("denoising_unet")]
#     dirs = sorted(dirs, key=lambda x: int(x.split("-")[1][:-4]))
#     path_ = dirs[-1]
#     weight = torch.load(os.path.join(resume_dir, path_),map_location='cpu')
#     model.load_state_dict(weight, strict=False)
#     print(f"Resuming from checkpoint {path_}")
#     global_step = int(path_.split("-")[1][:-4])

# train
i = 9000
start_time = time.time()
for k in range(epoch):
    for data_0,data_1 in train_dataloader:
        optim.zero_grad()
        # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
        data_0 = data_0.to(device) # sample data
        data_1 = data_1.to(device)
        t = torch.rand(data_0.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=data_0, x_1=data_1)
        loss = torch.pow(model(path_sample.x_t,path_sample.t,return_dict=False)[0] - path_sample.dx_t, 2).mean()
        loss.backward()
        optim.step()
        i+=1
        if (i+1) % print_every == 0:
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.5f} ' 
                .format(i+1, elapsed*1000/print_every, loss.item()))
            save_path = f'exp_output/scaled_flowmatching_sfc/samples/{i+1}.png'
            torch.save(model.state_dict(), f"exp_output/scaled_flowmatching_sfc/model_{i+1}.pth")
            validation(model,val_dataset,save_path)
            start_time = time.time()

# T = torch.tensor([1., 0.])
# T = T.to(device=device)
# grid_size = 200
# x_1 = torch.meshgrid(torch.linspace(-5, 5, grid_size), torch.linspace(-5, 5, grid_size))
# x_1 = torch.stack([x_1[0].flatten(), x_1[1].flatten()], dim=1).to(device)
# source distribution is an isotropic gaussian
# gaussian_log_density = Independent(Normal(torch.zeros(2, device=device), torch.ones(2, device=device)), 1).log_prob
# num_acc = 10
# log_p_acc = 0
# for i in range(num_acc):
#     _, log_p = solver.compute_likelihood(x_1=x_1, method='midpoint', step_size=step_size, exact_divergence=False, log_p0=gaussian_log_density)
#     log_p_acc += log_p
# log_p_acc /= num_acc
# _, exact_log_p = solver.compute_likelihood(x_1=x_1, method='midpoint', step_size=step_size, exact_divergence=True, log_p0=gaussian_log_density)
# likelihood = torch.exp(log_p_acc).cpu().reshape(grid_size, grid_size).detach().numpy()
# exact_likelihood = torch.exp(exact_log_p).cpu().reshape(grid_size, grid_size).detach().numpy()
# fig, axs = plt.subplots(1, 2,figsize=(10,10))
# cmin = 0.0
# cmax = 1/32 # 1/32 is the gt likelihood value
# norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
# axs[0].imshow(likelihood, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis', norm=norm)
# axs[0].set_title('Model Likelihood, Hutchinson Estimator, #acc=%d' % num_acc)
# axs[1].imshow(exact_likelihood, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis', norm=norm)
# axs[1].set_title('Exact Model Likelihood')
# fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axs, orientation='horizontal', label='density')
# plt.savefig('result.png')