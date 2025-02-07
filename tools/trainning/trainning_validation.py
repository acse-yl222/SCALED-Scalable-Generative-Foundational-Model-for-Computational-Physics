import random

def log_validation(
        denoising_unet,
        scheduler,
        accelerator,
        generator=None,
        valid_dataset=None
):
    logger.info("Running validation... ")
    if generator is None:
        generator = torch.manual_seed(42)
    dataset_len = len(valid_dataset)
    sample_idx = [random.randint(0, dataset_len) for _ in range(1)]

    pipe = SCALEDUrbanFlowPipeline(
        denoising_unet,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)
    results = {}
    ori_data, gt_result = valid_dataset[sample_idx[0]] # condition （9x32x32x32） gt_result (9x32x32x32)
    previous_value = ori_data[:3].unsqueeze(0).to('cuda')
    control_value = ori_data[3:].unsqueeze(0).to('cuda')
    next_value = gt_result[:3].unsqueeze(0).to('cuda')
    background_value = control_value.clone().bool()
    back_data = next_value.clone()
    back_data[:, :, 1:-1] = 1
    back_data[:, 0:1][background_value] = 0
    back_data[:, 1:2][background_value] = 0
    back_data[:, 2:3][background_value] = 0
    pre_particle = pipe(
        previous_value,
        back_data,
        num_inference_steps=25,
        guidance_scale=0,
        depth=64,
        height=128,
        width=128,
        generator=generator,
        return_dict=False,
    )
    results['WithoutBackground'] = {
        "prediction_flow": pre_particle.detach().cpu().numpy()[0],
        "gt_flow": next_value.detach().cpu().numpy()[0],
        "original_flow": previous_value.detach().cpu().numpy()[0]
    }
    del pipe
    return results




def visualize_with_diff(data_pre, data_gt, data_ori, filename):
    # Create a figure with a larger size and higher resolution
    fig = plt.figure(figsize=(24, 24), dpi=300)  # Adjust figsize to accommodate diff images
    gs = gridspec.GridSpec(4, 4, width_ratios=[1, 1, 1, 0.1])  # Add extra row for diff and column for colorbars
    
    for i in range(3):
        ax1 = plt.subplot(gs[i, 0])
        im1 = ax1.imshow(data_pre[i, 4], vmin=-1, vmax=1)
        ax1.set_title('Prediction')
        ax1.axis('off')
        
        ax2 = plt.subplot(gs[i, 1])
        im2 = ax2.imshow(data_gt[i, 4], vmin=-1, vmax=1)
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        ax3 = plt.subplot(gs[i, 2])
        im3 = ax3.imshow(data_ori[i, 4], vmin=-1, vmax=1)
        ax3.set_title('Original Data')
        ax3.axis('off')
        
        # Add colorbars in the 4th column
        cbar_ax1 = plt.subplot(gs[i, 3])
        fig.colorbar(im1, cax=cbar_ax1)
        
        cbar_ax2 = plt.subplot(gs[i, 3])
        fig.colorbar(im2, cax=cbar_ax2)
        
        cbar_ax3 = plt.subplot(gs[i, 3])
        fig.colorbar(im3, cax=cbar_ax3)
    
    # Now plot the differences in the 4th row
    diff_pre_gt = np.abs(data_pre - data_gt)  # Difference between Prediction and Ground Truth
    diff_pre_ori = np.abs(data_pre - data_ori)  # Difference between Prediction and Original
    diff_gt_ori = np.abs(data_gt - data_ori)  # Difference between Ground Truth and Original
    
    ax4 = plt.subplot(gs[3, 0])
    im4 = ax4.imshow(diff_pre_gt[0, 4], vmin=0, vmax=0.25)
    ax4.set_title('Diff Pre-GT')
    ax4.axis('off')
    
    ax5 = plt.subplot(gs[3, 1])
    im5 = ax5.imshow(diff_pre_ori[0, 4], vmin=0, vmax=0.25)
    ax5.set_title('Diff Pre-Ori')
    ax5.axis('off')
    
    ax6 = plt.subplot(gs[3, 2])
    im6 = ax6.imshow(diff_gt_ori[0, 4], vmin=0, vmax=0.25)
    ax6.set_title('Diff GT-Ori')
    ax6.axis('off')
    
    # Add colorbars for the differences
    cbar_ax4 = plt.subplot(gs[3, 3])
    fig.colorbar(im4, cax=cbar_ax4)
    
    cbar_ax5 = plt.subplot(gs[3, 3])
    fig.colorbar(im5, cax=cbar_ax5)
    
    cbar_ax6 = plt.subplot(gs[3, 3])
    fig.colorbar(im6, cax=cbar_ax6)
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    # Save the figure with higher resolution
    plt.savefig(filename, dpi=300)
    plt.close(fig)
