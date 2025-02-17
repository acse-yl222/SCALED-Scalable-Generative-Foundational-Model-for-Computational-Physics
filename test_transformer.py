from scaled.model.transformers.dit_transformers import DiTTransformer1DModel
import torch
model = DiTTransformer1DModel(in_channels=4,out_channels=2,sample_size=20560,patch_size=16).to('cuda')

input = torch.zeros(1,4,20560).to('cuda')

out = model(input,torch.tensor([1]).to('cuda'),torch.tensor([1]).to('cuda')).sample

print(out.shape)