from scaled.model.unets.unet_1ds import UNet1DsModel
import torch

model = UNet1DsModel(
        in_channels=2,
        out_channels=2,
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
        block_out_channels=(64, 128, 192, 256),
        add_attention=False,
        )

input = torch.zeros(1,2,20560)

output = model(input).sample

print(output.shape)