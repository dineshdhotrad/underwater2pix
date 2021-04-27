import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
from metricss import rmetrics,nmetrics
import pandas as pd

# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Plot losses
def plot_loss(d_losses, g_losses, num_epochs, save=False, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result(input, target, gen_image, epoch, idx, training=True, save=False, save_dir='results/', show=False, fig_size=(7, 7)):
    if not training:
        fig_size = (input.size(2) * 3 / 100, input.size(3)/100)

    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    imgs = [input, gen_image, target]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)


    if training:
        title = 'Epoch {0}'.format(epoch + 1)
        fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        save_dir = save_dir + 'Result_epoch_{:d}'.format(epoch+1) + "/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if training:
            save_fn = save_dir + f"{idx}" + '.png'
        else:
            save_fn = save_dir + 'Test_result_{:d}'.format(epoch+1) + f"{idx}" + '.png'
            fig.subplots_adjust(bottom=0)
            fig.subplots_adjust(top=1)
            fig.subplots_adjust(right=1)
            fig.subplots_adjust(left=0)
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

# Testing Results
def test(test_data_loader, G, device, epoch, save_dir, csv_dir):
    G.eval()

    ssim_list = []
    mse_list = []
    psnr_list = []
    uiqm_list = []
    uciqe_list = []
    # with torch.cuda.amp.autocast():
    with torch.no_grad():
        for idx,(x,y) in enumerate(test_data_loader):
            x_,y_ = x.to(device),y.to(device)
            y_fake = G(x_)
            plot_test_result(x_.cpu().data, y_.cpu().data, y_fake.cpu().data, epoch, idx, save=True, save_dir=save_dir)
            

            x_ = (((x_[0] - x_[0].min()) * 255) / (x_[0].max() - x_[0].min())).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)[...,::-1]
            y_ = (((y_[0] - y_[0].min()) * 255) / (y_[0].max() - y_[0].min())).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)[...,::-1]
            y_fake = (((y_fake[0] - y_fake[0].min()) * 255) / (y_fake[0].max() - y_fake[0].min())).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)[...,::-1]

            
            mse, psnr,ssim = rmetrics(y_fake,y_)
            uiqm,uciqe = nmetrics(y_fake)
            mse_list.append(mse)
            ssim_list.append(ssim)
            psnr_list.append(psnr)
            uiqm_list.append(uiqm)
            uciqe_list.append(uciqe)
    df = pd.DataFrame(data={"save_dir":save_dir, "ssim":ssim_list, "mse":mse_list, "psnr":psnr_list, "uiqm":uiqm_list, "uciqe":uciqe_list})
    df.to_csv(csv_dir+f"/Epoch_{str(epoch)}.csv")


    G.train()

    print("-------Done testing-------\n-------Saved Images, plots and Excel Sheet-------")

# Save Checkpoints
def save_checkpoint(epoch, model, optimizer, filename):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, filename)