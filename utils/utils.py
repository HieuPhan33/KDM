# -*- coding: utf-8 -*-
"""
Created on 18/08/2020 9:32 am

@author: Soan Duong, UOW
"""
# Standard library imports
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

# Third party imports
import torch
from PIL import Image
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# Local application imports
from utils.datasets import HSIDataset
from tqdm import tqdm


def hsi_read_header(file_name, verbose=False):
    """
    Load the information of a hyperspectral image from a header file (.hdr)
    :param file_name: file path of the header hsi file
    :param verbose: bool value to display the result (defaul: False)
    :return: 5 params
        - n_samples: number of n_samples in the image (width)
        - lines: number of lines in the image (height)
        - bands: number of bands (wave lengths)
        - data_type: data type stored in the data file
        - wave_lengths: list of wave lengths used to acquired the data
    """
    # Open a file for reading
    f = open(file_name, 'r+')

    # Read all the lines in the header file
    text = f.readlines()

    # Close the file
    f.close()

    n = 0
    while n < len(text):
        text_line = text[n].replace('\n', '')
        # Get number of samples (width)
        if 'samples' in text_line:
            n_samples = int(text_line.split(' ')[-1])

        # Get number of lines (height)
        if 'lines' in text_line:
            n_lines = int(text_line.split(' ')[-1])

        # Get number of bands/wave lengths
        if 'bands' in text_line and not(' bands' in text_line):
            n_bands = int(text_line.split(' ')[-1])

        # Get the data type
        if 'data type' in text_line:
            data_type = int(text_line.split(' ')[-1])

        # Get the wave length values
        if 'Wavelength' in text_line:
            wave_lengths = np.zeros(n_bands)
            for k in range(n_bands):
                n = n + 1
                text_line = text[n].replace(',\n', '').replace(' ', '')
                wave_lengths[k] = float(text_line)
            break
        n = n + 1

    # Convert the data_type into the string format
    if data_type == 1:
        data_type = 'int8'
    elif data_type == 2:
        data_type = 'int16'
    elif data_type == 3:
        data_type = 'int32'
    elif data_type == 4:
        data_type = 'float32'
    elif data_type == 5:
        data_type = 'double'
    elif data_type == 12:
        data_type = 'uint16'
    elif data_type == 13:
        data_type = 'uint32'
    elif data_type == 14:
        data_type = 'int64'
    elif data_type == 15:
        data_type = 'uint64'

    if verbose:     # display the outputs if it is necessary
        print('Image width = %d' % n_samples)
        print('Image height = %d' % n_lines)
        print('Bands = %d' % n_bands)
        print('Data type = %s' % data_type)

    return n_samples, n_lines, n_bands, data_type, wave_lengths


def hsi_read_data(file_name, sorted=True):
    """
    Read the image cube from the raw hyperspectral image file (.raw)
    :param file_name: file path of the raw hsi file
    :param sorted: bool value to sort the image cube in the ascending of wave_lengths
    :return: 2 params
        - img: image cube in the shape of [n_lines, n_samples, n_bands]
        - wave_lengths: list of wave lengths used to acquired the data
    """
    # Get the information from the header file
    hdr_file = file_name[:-4] + '.hdr'
    n_samples, n_lines, n_bands, data_type, wave_lengths = hsi_read_header(hdr_file)

    # Open the raw file
    f = open(file_name, 'rb')
    # Read the data in the raw file
    data = np.frombuffer(f.read(), dtype=data_type)

    # Close the file
    f.close()

    # Reshape the data into the 3D formar of lines x bands x samples]
    img = data.reshape([n_lines, n_bands, n_samples])

    # Permute the image into the correct format lines x samples x bands
    img = np.moveaxis(img, [0, 1, 2], [0, 2, 1])

    if sorted:
        # Get the sorted indices of wave_lengths in the ascending order
        indx = np.argsort(wave_lengths)

        # Get the sorted wave_lengths
        wave_lengths = wave_lengths[indx]

        # Sort the image cube in the ascending order of wave_lengths
        img = img[:, :, indx]

    return img, wave_lengths


def norm_inten(I, max_val=255):
    """
    Normalize intensities of I to the range of [0, max_val]
    :param I: ndarray
    :param max_val: maximum value of the normalized range, default = 255
    :return: normalized ndarray
    """
    I = I - np.min(I)
    I = (max_val/np.max(I)) * I

    return I


def hsi_img2rgb(img, wave_lengths=None):
    """
    Convert raw hsi image cube into a pseudo-color image
    :param img: 3D array of size H x W x bands
    :param wave_lengths: 1D array of wavelength bands
    :return: array of size H x W x 3, a pseudo-color image
    """
    # Get the indices of ascending-sorted wavelengths
    if wave_lengths is None:
        indx = list(range(img.shape[-1]))
    else:
        indx = np.argsort(wave_lengths)

    # Get the pseudo-red channel (slice of the longest wavelength)
    ind = indx[-1]
    r = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Get the pseudo-green channel (slice of the median wavelength)
    ind = indx[len(indx)//2]
    g = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Get the pseudo-blue channel (slice of the shortest wavelength)
    ind = indx[0]
    b = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Concatenate the channels into a color image
    rgb_img = np.concatenate([r, g, b], axis=-1)

    return rgb_img.astype(np.uint8)

def visualize_att(dataloader, model, num_results = 3, sv_dir = '',
                                   labels={}):
    for id,label in labels.items():
        new_dir = os.path.join(sv_dir,'class_{}'.format(label))
        os.makedirs(new_dir,exist_ok=True)
    model.eval()
    max_count = 200
    with torch.no_grad():
        for index, batch in enumerate(tqdm(dataloader)):
            # 1. Get a minibatch data for training
            image, label, name = batch['input'], batch['ground_truth_seg'], batch['name']
            image = image.to('cuda')
            #image, label, _, name = batch
            name = name[0]
            h,w = image.size(2), image.size(3)
            atts, preds = model(image)
            #atts, preds = preds[:-num_results], preds[-num_results:]
            #print(len(a))
            #print(name)
            for clf_id in range(num_results-1):
                att = atts[clf_id]
                #print(torch.sum(att))
                att = F.upsample(att,size=(h, w)).cpu().data.numpy()
                for id, label in labels.items():
                    new_dir = os.path.join(sv_dir, 'class_{}'.format(label))
                    att_ = att[:,id]
                    sv_path = os.path.join(new_dir, 'att_heatmap_{}_clf{}.jpg'.format(name,clf_id))
                    heatmap = cv2.normalize(att_[0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
                    if name == '000001' and label == 'creature':
                        print(clf_id)
                        print(sv_path)
                        #print(heatmap)
                        print(np.mean(heatmap))
                    cv2.imwrite(sv_path, heatmap)
                    del att_
            if index > max_count:
                return

def show_visual_results(x, y_gt, y_pr, classes=[0, 1, 2, 3, 4],
                        show_visual=0, comet=None, fig_name=""):
    """
    Show the pseudo-color, ground-truth, and output images
    :param x: array of size (batchsize, n_bands, H, W)
    :param y_gt: array of size (batchsize, 1, H, W)
    :param y_pr: array of size (batchsize, n_classes, H, W)
    :param classes: list of class labels
    :param show_visual: boolean to display the figure or not
    :param comet: comet logger object
    :param fig_name: string as the figure name for the comet logger
    :return:
    """
    # Select the first image in the batch to display
    y_gt = y_gt[0, ...]                      # of size (H, W)
    y_pr = y_pr[0, ...]                         # of size (n_classes, H, W)
    x = x[0, ...]                               # of size (n_bands, H, W)
    x = np.moveaxis(x, [0, 1, 2], [2, 0, 1])    # of size (H, W, n_bands)

    # Convert the probability into the segmentation result image
    y_pr = convert_prob2seg(y_pr, classes)

    # Set figure to display
    h = 2
    if plt.fignum_exists(h):  # close if the figure existed
        plt.close()
    fig = plt.figure(figsize=(9, 2))
    fig.subplots_adjust(bottom=0.5)

    # Set colormap
    colors = ['black', 'green', 'blue', 'red', 'yellow']
    cmap_ygt = mpl.colors.ListedColormap(colors[np.int(np.min(y_gt)):np.int(np.max(y_gt)) + 1])
    cmap_ypr = mpl.colors.ListedColormap(colors[np.int(np.min(y_pr)):np.int(np.max(y_pr)) + 1])
    # print('Input classes = ', np.unique(y_gt))
    # print('Output classes = ', np.unique(y_pr))

    # Plot the pseudo-color image
    plt.subplot(131)
    plt.imshow(hsi_img2rgb(x))
    plt.title('Pseudo-color image')

    # Plot the ground-truth image
    ax = plt.subplot(132)
    im = ax.imshow(y_gt, cmap=cmap_ygt)
    plt.title('Ground-truth image')
    fig.colorbar(im, ax=ax)

    # Plot the predicted segmentation image
    ax = plt.subplot(133)
    im = ax.imshow(y_pr, cmap=cmap_ypr)
    plt.title('Predicted segmentation image')
    fig.colorbar(im, ax=ax)
    plt.savefig(fig_name)

    # if show_visual:
    #     plt.show()

    if comet is not None:
        comet.log_figure(figure_name=fig_name, figure=fig)


def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.mkdir(path)
        if visual_folder is True:
            os.mkdir(path + '/visual')  # for visual results
    else:
        print("DIR already existed.")
    print('Experiment dir : {}'.format(path))


def prepare_device(n_gpu_use=1):
    """
    Setup GPU device if it is available, move the model into the configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There\'s no GPU available on this machine,"
            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available "
            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_experiment_dataloaders(cfg):
    """
    Get and return the train, validation, and test dataloaders for the experiment.
    :param classes: list of classes (denoted by a number started from 0) of the segmentation problem,
                    e.g. [0, 1, 2, 3, 4, 5]
    :param cfg: dict that contains the required settings for the dataloaders
                (dataset_dir and train_txtfiles)
    :return: train, validation, and test dataloaders
    """
    # Create an instance of the HSIDataset
    hsi_train_dataset = HSIDataset(cfg['dataset_dir'], cfg['train_txtfiles'],
                                   cfg['classes'], cfg['n_cutoff_imgs'])

    # Set params for pliting the dataset in to train and val subset with a ratio 0.9 : 0.1
    train_rate = 0.9
    num_train = len(hsi_train_dataset)
    indx = list(range(num_train))
    split_point = int(np.floor(train_rate * num_train))

    # Get the train dataloader
    train_dataloader = torch.utils.data.DataLoader(hsi_train_dataset,
                                                   batch_size=cfg['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indx[:split_point]),
                                                   pin_memory=True, num_workers=cfg['num_workers'])

    # Get the train dataloader
    val_dataloader = torch.utils.data.DataLoader(hsi_train_dataset,
                                                 batch_size=cfg['batch_size'],
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                     indx[split_point:]),
                                                 pin_memory=True, num_workers=cfg['num_workers'])
    # Get the testloader
    hsi_test_dataset = HSIDataset(cfg['dataset_dir'], cfg['test_txtfiles'],
                                  cfg['classes'], cfg['n_cutoff_imgs'])
    test_dataloader = torch.utils.data.DataLoader(hsi_test_dataset,
                                                  batch_size=1,  # restrict to 1 as we have a function to display one image
                                                  pin_memory=True, num_workers=cfg['num_workers'])

    # Return the dataloaders
    return train_dataloader, val_dataloader, test_dataloader


def init_obj(module_name, module_args, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    assert all([k not in module_args for k in
                kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def convert_prob2seg(y_prob, classes):
    """
    Convert the class-probability image into the segmentation image
    :param y_prob: class-probability image with a size of n_classes x H x W
    :param classes: list of classes in image y
    :return: 2D array, a segmentation image with a size of H x W
    """
    y_class = np.argmax(y_prob, axis=0)
    y_seg = np.zeros((y_prob.shape[1], y_prob.shape[2]))

    for k, class_label in enumerate(classes):
        # Find indices in y_class whose pixels are k
        indx = np.where(y_class == k)
        if len(indx[0] > 0):
            y_seg[indx] = class_label

    return np.int8(y_seg)


def compute_confusion_matrix(y_gt, y_pr, classes=[0, 1, 2, 3, 4]):
    """
    :param y_gt: array of size (batchsize, 1, H, W)
    :param y_pr: array of size (batchsize, n_classes, H, W)
    :param classes: list of class labels
    :return: confusion matrix of the y_gt and segmentation of y_pr
    """
    cm = 0
    for k in range(y_gt.shape[0]):
        # Convert the current y_pr in to the segmentation
        y_prk = convert_prob2seg(np.squeeze(y_pr[k, ...]), classes).flatten()

        # Get the kth ground-truth segmentaion
        y_gtk = np.squeeze(y_gt[k, ...]).flatten()

        # Sum up the confusion matrix
        cm = cm + confusion_matrix(y_gtk, y_prk, classes)

    return cm

def compute_eval_from_cm(cm):
    true = cm.sum(1)
    pred = cm.sum(0)
    tp = np.diag(cm)
    recall = tp / np.maximum(1.0, true)
    precision = tp / np.maximum(1.0, pred)
    dice = (2*(precision * recall)/(precision + recall)).mean()

    pixel_acc = tp.sum() / true.sum()
    mean_acc = (tp / np.maximum(1.0, true)).mean()
    IoU_array = (tp / np.maximum(1.0, pred + true - tp))
    mean_IoU = IoU_array.mean()

    kappa = cohen_kappa_score(cm)
    return pixel_acc, mean_acc, mean_IoU, IoU_array, dice, kappa

def cohen_kappa_score(confusion):
    r"""Cohen's kappa: a statistic that measures inter-annotator agreement.

    This function computes Cohen's kappa [1]_, a score that expresses the level
    of agreement between two annotators on a classification problem. It is
    defined as

    .. math::
        \kappa = (p_o - p_e) / (1 - p_e)

    where :math:`p_o` is the empirical probability of agreement on the label
    assigned to any sample (the observed agreement ratio), and :math:`p_e` is
    the expected agreement when both annotators assign labels randomly.
    :math:`p_e` is estimated using a per-annotator empirical prior over the
    class labels [2]_.

    Read more in the :ref:`User Guide <cohen_kappa>`.

    Parameters
    ----------
    weights : str, optional
        Weighting type to calculate the score. None means no weighted;
        "linear" means linear weighted; "quadratic" means quadratic weighted.


    Returns
    -------
    kappa : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.
    """
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    w_mat = np.ones([n_classes, n_classes], dtype=np.int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k

def test_running_time(dump_input, model, comet):
    model.eval()
    reps = 100
    dump_input = dump_input.cuda()
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((reps, 1))
    with torch.no_grad():
        for _ in range(50):
            model(dump_input)
        print("Start testing inference time...")
        for rep in range(reps):
            starter.record()
            _ = model(dump_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    #infer_time = total_time/reps
    infer_time = np.sum(timings) / reps
    infer_time /= 1000

    print("Inference time : {} seconds {} FPS".format(infer_time,1/infer_time))
    comet.log_metric(f'inference_time', infer_time)


def test_running_time_with_wrapper(dump_input, model, comet, n_heads):
    from models.sgr_wrapper import SGR_Wrapper
    from thop import profile, clever_format
    model.eval()

    infer_time = [0] * n_heads
    reps = 500
    dump_input = dump_input.cuda()
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for i in range(n_heads):
            sub_model = SGR_Wrapper(sgr_model=model.module, t_head=i+1).cuda()
            sub_model.eval()
            flops, params = profile(sub_model.to('cuda:0'), inputs=(dump_input.to('cuda:0'),), verbose=False)
            macs, params = clever_format([flops, params], "%.3f")
            print(f"Model {n_heads} FLOPS: {macs}, PARAMS: {params}")
            comet.log_other(f'Model {n_heads} trainable parameters', params)
            comet.log_other(f'Model {n_heads} Floating point operations per second (FLOPS)', flops)
            comet.log_other(f'Model {n_heads} Multiply accumulates per second (MACs)', macs)

            for _ in range(100):
                sub_model(dump_input)
            print(f"Start testing inference time for model {i+1}...")
            timings = np.zeros((reps, 1))
            for rep in range(reps):
                starter.record()
                _ = sub_model(dump_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
            infer_time[i] = np.sum(timings) / rep
            print(timings.mean())

    for i in range(n_heads):
        print("Inference time : {} seconds {} FPS".format(infer_time[i],1/(infer_time[i]/1000)))
        comet.log_metric(f'inference_time_{i}', infer_time[i])
        

# ------------------------------------------------------------------------------
# Main function for testing
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # file_name = '../../../02.data/hsi_data/200716_botanic_garden/200716_mp_0.raw'
    # file_name = '../../../02.data/hsi_data/200716_botanic_garden/200716_mp_26.raw'
    file_name = '../data/UOW-HSI/000426.raw'
    # file_name = '../data/UOW-HSI/000247.raw'
    # file_name = 'U:/02-Data/UOW-HSI/000426.raw'
    #
    # file_name = 'U:/02-Data/UOW-HSI/000399.raw'
    # file_name = 'U:/02-Data/UOW-HSI/000425.raw'
    # file_name = 'U:/02-Data/UOW-HSI/000570.raw'
    # file_name = 'U:/02-Data/UOW-HSI/000002.raw'
    bmp_file = file_name[:-4] + '.bmp'
    # Read hsi raw data
    img, wave_lengths = hsi_read_data(file_name)
    rgb = hsi_img2rgb(img)    # get pseudo-color image

    # Read ground-truth image
    bmp = Image.open(bmp_file)
    bmp = np.array(bmp.getdata()).reshape(bmp.size[1], bmp.size[0])

    colors = ['black', 'green', 'blue', 'red', 'yellow']
    cmap = mpl.colors.ListedColormap(colors[np.min(bmp):np.max(bmp)+1])
    fig = plt.figure(figsize=(3, 2))
    fig.subplots_adjust(bottom=0.5)
    plt.imshow(bmp, cmap=cmap)
    plt.show()
    # # Set colormap
    # colors = ['black', 'green', 'blue', 'red', 'yellow']
    # cmap = mpl.colors.ListedColormap(colors[np.min(bmp):np.max(bmp)+1])
    # # cmap.set_over('0.25')
    # # cmap.set_under('0.75')
    # bounds = [0, 1, 2, 3, 4]
    # labels = ['Other', 'Plant', 'Soil', 'Creature', 'Yellow']
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #
    # # Plot images
    # fig = plt.figure(figsize=(9, 2))
    # fig.subplots_adjust(bottom=0.5)
    # plt.subplot(131)
    # plt.imshow(rgb)
    # plt.title('Pseudo-color image')
    #
    # # Plot the ground-truth image
    # ax = plt.subplot(132)
    # im = ax.imshow(bmp, cmap=cmap)
    # plt.title('Ground-truth image')
    # fig.colorbar(im, ax=ax)
    #
    # # Plot the Float-converted ground-truth image
    # ax = plt.subplot(133)
    # im = ax.imshow(np.float32(bmp[np.newaxis, ...])[0, ...], cmap=cmap)
    # plt.title('Float-converted ground-truth image')
    # fig.colorbar(im, ax=ax)
    #
    # plt.show()
