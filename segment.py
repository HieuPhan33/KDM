# -*- coding: utf-8 -*-
"""
Created on 27/08/2020 11:14 am

@author: Soan Duong, UOW
"""
# Standard library imports
import glob
import numpy as np
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
import timeit

# Third party imports
import torch

# Local application imports
from models.hsi_net import HSINet
from utils.utils import hsi_read_data, hsi_img2rgb, convert_prob2seg


def segment_hsi(model, dataset_dir, save_seg=True, suffix='seg'):
    """
    Segment all the hsi images in a given data folder
    :param model: trained HSINet mode
    :param dataset_dir: directory of the testing dataset folder
    :param save_seg: boolean value to save the result plot
    :param suffix: suffix of the saved file
    :return: plots and saved images
    """
    # Get all .raw file in the dataset_dir
    raw_files = [f for f in glob.glob(dataset_dir + '*.raw')]
    elapsed_times = []
    for file_name in raw_files:
        print('Segmenting file: ' + file_name)
        # Read the input image
        img, _ = hsi_read_data(file_name)  # of size (H, W, n_bands)
        x = np.moveaxis(img, [0, 1, 2], [1, 2, 0])  # of size (n_bands, H, W)

        # Convert the image to torch tensor
        x = torch.Tensor(np.float32(x)[np.newaxis, ...])

        # Set the timer
        start = timeit.timeit()

        # Evaluate the input image by the trained mode
        y = model(x)  # of size (1, n_classes, H, W)

        # Convert y to ndarray
        y = y.detach().cpu().numpy()[0, ...]  # of size (n_classes, H, W)

        # Get the segmentation of x
        y_seg = convert_prob2seg(y, m_params['classes'])

        # Get the elapsed time
        end = timeit.timeit()
        elapsed_times.append(end - start)

        # Plot the images
        show_imgs(hsi_img2rgb(img), y_seg)
        if save_seg:
            seg_file = file_name[:-4] + '_' + suffix + '.png'
            plt.savefig(seg_file, bbox_inches='tight')
        plt.show()

    print('Elapsed time: %.6f +- %.6f (s)' % (np.mean(elapsed_times), np.std(elapsed_times)))


def show_imgs(rgb, seg):
    """
    Show the pseudo-color and segmented images of the hsi image
    :param rgb: pseudo-color image of size (H, W, 3)
    :param seg: segmented image of size (H, W)
    :return:
    """
    # Set colormap
    colors = ['black', 'green', 'blue', 'red', 'yellow']
    cmap = mpl.colors.ListedColormap(colors[np.min(seg):np.max(seg) + 1])

    # Plot the pseudo-color image
    fig = plt.figure(figsize=(6, 2))
    fig.subplots_adjust(bottom=0.5)
    plt.subplot(121)
    plt.imshow(rgb)
    plt.title('Pseudo-color image')

    # Plot the ground-truth image
    ax = plt.subplot(122)
    im = ax.imshow(seg, cmap=cmap)
    plt.title('Ground-truth image')
    fig.colorbar(im, ax=ax)

    # plt.savefig('test.png', bbox_inches='tight')
    # plt.show()


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # dataset_dir = 'U:/02-Data/UOW-HSI/'
    # hsi_img = '000001.raw'
    # Set the input value
    # Elapsed time: 0.001181 +- 0.001308 (s)
    dataset_dir = 'U:/02-Data-Preparation/hsi_data/for_testing/'
    nn_config_file = 'configs/hsi_5000_fold1.yml'
    model_file = 'experiments/hsi-seg-5000epochs-fold1-20200827-002545/best_model.pth'

    hsi_img = '200716_msp_4.raw'
    file_name = dataset_dir + hsi_img

    # Load the trained model
    with open(nn_config_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    m_params = cfg['model_params']
    model = HSINet(n_bands=m_params['n_bands'],
                   classes=m_params['classes'],
                   nf_enc=m_params['nf_enc'],
                   nf_dec=m_params['nf_dec'],
                   do_batchnorm=m_params['do_batchnorm'],
                   max_norm_val=None)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    # Segment hsi images in the dataset_dir
    # segment_hsi(model, dataset_dir)
    # print('Done')

    # Read the input image
    img, _ = hsi_read_data(file_name)  # of size (H, W, n_bands)
    x = np.moveaxis(img, [0, 1, 2], [1, 2, 0])  # of size (n_bands, H, W)

    # Set the timer
    start = timeit.timeit()

    # Convert the image to torch tensor
    x = torch.Tensor(np.float32(x)[np.newaxis, ...])

    # Evaluate the input image by the trained mode
    y = model(x)  # of size (1, n_classes, H, W)

    # Convert y to ndarray
    y = y.detach().cpu().numpy()[0, ...]  # of size (n_classes, H, W)

    # Get the segmentation of x
    y_seg = convert_prob2seg(y, m_params['classes'])

    # Get the elapsed time
    end = timeit.timeit()
    elapsed_time = end - start
    print('Elapsed time: %.6f' % elapsed_time)

    # Get pseudo-color image
    rgb = hsi_img2rgb(img)

    # Set colormap
    colors = ['black', 'green', 'blue', 'red', 'yellow']
    cmap = mpl.colors.ListedColormap(colors[np.min(y_seg):np.max(y_seg)+1])

    # Plot the pseudo-color image
    fig = plt.figure(figsize=(6, 2))
    fig.subplots_adjust(bottom=0.5)
    plt.subplot(121)
    plt.imshow(rgb)
    plt.title('Pseudo-color image')

    # Plot the ground-truth image
    ax = plt.subplot(122)
    im = ax.imshow(y_seg, cmap=cmap)
    plt.title('Ground-truth image')
    fig.colorbar(im, ax=ax)
    plt.show()
