import time
import torch
from torch import nn, Tensor
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from scaled.flow_matching.ode_solver import ConditionODESolver
from flow_matching.utils import ModelWrapper
from scaled.dataset.sfc_dataset import SFCDiffusionDatasetv1
import matplotlib.pyplot as plt
from scaled.model.unets.unet_1ds import UNet1DsModel
import warnings
import glob
import numpy as np
import os
import os.path as osp
import torch
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

max_checkpoints = 5
checkpoint_dir = "exp_output/scaled_flowmatching_sfc_gen"

def manage_checkpoints():
    """
    维护checkpoint目录,仅保留最新的 max_checkpoints 个权重文件。
    """
    weight_files = sorted(
        glob.glob(osp.join(checkpoint_dir, "model-*.pth")),
        key=lambda x: int(x.split("-")[-1][:-4])
    )
    while len(weight_files) > max_checkpoints:
        file_to_remove = weight_files.pop(0)
        os.remove(file_to_remove)
        print(f"Removed old checkpoint: {file_to_remove}")

class WrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
            return self.model(x, t,return_dict=False)[0]

def validation(model,dataset,path):
    wrapped_vf = WrappedModel(model)
    #step size for ode solver
    data_0,data_1 = dataset[0]
    data_0 = data_0.unsqueeze(0).to(device)
    data_1 = data_1.unsqueeze(0).to(device)
    noise = torch.rand_like(data_1)
    step_size = 0.05
    T = torch.linspace(0,1,10)
    T = T.to(device=device)
    solver = ConditionODESolver(velocity_model=wrapped_vf)
    sol = solver.sample(time_grid=T, condition_latent=data_0,x_init=noise, method='midpoint', step_size=step_size, return_intermediates=True)
    sol = sol.cpu().numpy()
    T = T.cpu()

    fig, axs = plt.subplots(3,10,figsize=(20,20))
    data_1 = data_1[0][0].cpu().numpy()
    print(data_1.shape)
    data_1 = data_1.reshape((257,80))
    for i in range(10):
        result = sol[i][0][0]
        result = result.reshape((257,80))
        diff = np.abs(data_1-result)
        a = axs[0,i].imshow(data_1,vmax=0.5,vmin=-0.5)
        b = axs[1,i].imshow(result,vmax=0.5,vmin=-0.5)
        c = axs[2,i].imshow(diff,vmax=0.1,vmin=0)
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
batch_size = 24
print_every = 100
hidden_dim = 512
epoch = 10000

# velocity field model init

model  = UNet1DsModel(
        in_channels=4,
        out_channels=2,
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
        block_out_channels=(128, 256, 384, 512),
        add_attention=False,
        ).to(device)
model.load_state_dict(torch.load('exp_output/scaled_flowmatching_sfc_gen/model_1500.pth',map_location='cpu'))

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
i = 1500
start_time = time.time()
for k in range(epoch):
    for data_0,data_1 in train_dataloader:
        optim.zero_grad()
        # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
        data_0 = data_0.to(device) # sample data
        data_1 = data_1.to(device)
        noise = torch.rand_like(data_1)
        t = torch.rand(data_0.shape[0]).to(device)
        input_data = torch.cat([data_0,noise],dim=1)
        path_sample = path.sample(t=t, x_0=noise, x_1=data_1)
        input_data = torch.cat([path_sample.x_t,noise],dim=1)
        loss = torch.pow(model(input_data,path_sample.t,return_dict=False)[0] - path_sample.dx_t, 2).mean()
        loss.backward()
        optim.step()
        i+=1
        elapsed = time.time() - start_time
        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.5f} ' 
            .format(i+1, elapsed*1000/print_every, loss.item()))
        start_time = time.time()
        if (i+1) % print_every == 0:
            save_path = f'exp_output/scaled_flowmatching_sfc_gen/samples/{i+1}.png'
            torch.save(model.state_dict(), f"exp_output/scaled_flowmatching_sfc_gen/model-{i+1}.pth")
            validation(model,val_dataset,save_path)
            manage_checkpoints()