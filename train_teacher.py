# -*- coding: utf-8 -*-
"""
Created on 18/08/2020 7:41 pm

@author: Hieu Phan, UOW
"""
# Standard library imports
import os
import time
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Third party imports
from comet_ml import Experiment
import torch
from torchsummary import summary
from tqdm import tqdm
from thop import profile, clever_format
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# Local application imports
import utils.losses
import utils.metrics

import models.hsi_net
from utils.utils import show_visual_results, create_exp_dir, test_running_time_with_wrapper, \
    get_experiment_dataloaders, init_obj, compute_confusion_matrix, compute_eval_from_cm


def debug(cfg):
    """
    Set the network based on the configuration spicified in the .yml file
    :param cfg: dict of parameters that are specified in the .yml config file
    :param comet: comet logger object
    :return:
    """
    # Set random seeds for reproducibility
    init_seeds(cfg['seed'])

    # Use GPU is available, otherwise use CPU
    # device, n_gpu_ids = prepare_device()
    device = torch.device('cuda:0')
    # device = 'cpu'

    # Create the model
    m_params = cfg['model_params']
    model = getattr(models.hsi_net, m_params['name'])(n_bands=m_params['n_bands'],
                                                      classes=m_params['classes'],
                                                      nf_enc=m_params['nf_enc'],
                                                      nf_dec=m_params['nf_dec'],
                                                      do_batchnorm=m_params['do_batchnorm'],
                                                      # asp_ocr=m_params['asp'],
                                                      n_heads=m_params['n_heads'],
                                                      max_norm_val=None,
                                                      encoder_name=m_params['encoder_name'])


    # Get dataloaders
    t_params = cfg['train_params']
    t_params['classes'] = m_params['classes']
    train_loader, val_loader, test_loader = get_experiment_dataloaders(cfg['train_params'])
    profile(model.to('cuda:0'), inputs=(test_loader.dataset[0]['input'].unsqueeze(0).to('cuda:0'),), verbose=False)

    model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)
    saved = torch.load(cfg['train_params']['save_dir'] + '/best_model.pth')
    model.load_state_dict(saved)
    print("finish loading model")


# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------
def main(cfg, comet):
    """
    Set the network based on the configuration spicified in the .yml file
    :param cfg: dict of parameters that are specified in the .yml config file
    :param comet: comet logger object
    :return:
    """
    # Set random seeds for reproducibility
    init_seeds(cfg['seed'])

    # Use GPU is available, otherwise use CPU
    #device, n_gpu_ids = prepare_device()
    device = torch.device('cuda:0')
    #device = 'cpu'

    # Create the model
    m_params = cfg['model_params']
    model = getattr(models.hsi_net, m_params['name'])(n_bands=m_params['n_bands'],
                                                      classes=m_params['classes'],
                                                      nf_enc=m_params['nf_enc'],
                                                      nf_dec=m_params['nf_dec'],
                                                      do_batchnorm=m_params['do_batchnorm'],
                                                      n_heads=m_params['n_heads'],
                                                      max_norm_val=None,
                                                      encoder_name=m_params['encoder_name'])

    # Get dataloaders
    t_params = cfg['train_params']
    t_params['classes'] = m_params['classes']
    train_loader, val_loader, test_loader = get_experiment_dataloaders(cfg['train_params'])
    #
    # # Log the trainable parameters
    x = test_loader.dataset[0]['input'].unsqueeze(0)
    if 'cuda' in device.type:
        x = x.to('cuda:0')
    # print(model.device)
    # print(x.device)
    flops, params = profile(model.to('cuda:0'), inputs=(x.to('cuda:0'),), verbose=False)
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPS: {}, PARAMS: {}".format(flops, params))
    comet.log_other('Model trainable parameters', params)
    comet.log_other('Floating point operations per second (FLOPS)', flops)
    comet.log_other('Multiply accumulates per second (MACs)', macs)

    model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)


    # Define an optimiser
    print(f"Optimiser for model weights: {cfg['optimizer']['type']}")
    optimizer = init_obj(cfg['optimizer']['type'],
                         cfg['optimizer']['args'],
                         torch.optim, model.parameters())
    best_performance = 0
    last_epoch = 0
    if cfg['resume']:
        model_state_file = os.path.join(cfg['train_params']['save_dir'],
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_performance = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    # Metrics
    metric = getattr(utils.metrics, cfg['metric'])(activation='softmax2d')

    # # Loss function
    loss_fn = getattr(utils.losses, cfg['loss'])(base_feat_w = m_params['feat_weight'],
                                                 base_resp_w = m_params['resp_weight'],
                                                 student_loss_w = m_params['student_weight'],
                                                )

    # Training
    training(cfg['train_params'], optimizer, model, train_loader, val_loader,
             loss_fn, metric, device, last_epoch, best_performance, comet)

    # Testing
    print("\nThe training has completed. Testing the model now...")

    #Load the best model
    saved = torch.load(cfg['train_params']['save_dir'] + '/best_model.pth')
    model.load_state_dict(saved)

    
    #testing(model, m_params['classes'], test_loader, metric, device, 
    #        m_params['n_heads'], comet, cfg['train_params']['save_dir'])
    test_running_time_with_wrapper(x, model, comet, m_params['n_heads'])
    comet.log_asset(cfg['train_params']['save_dir'] + '/best_model.pth')


def train_epoch(optimizer, model, train_loader, loss_fn, metric, device, epoch, epoch_iters, max_iters):
    """
    Train an epoch for the dataset. Logic for training an epoch:
        1. Get a minibatch for training
        2. Compute the forward path
        3. Compute the loss and update the weight
        4. Evaluate train performance
        5. Store the loss and metric
        6. Display losses
    :param optimizer: optimizer for the model
    :param model: network model
    :param train_loader: training dataloader
    :param loss_fn: loss function for the model
    :param metric: metric for computing the performance
    :param device: device used for training (cpu or gpu)
    :return: train the network model for the
    """
    cur_iters = epoch * epoch_iters
    model.train()
    pbar = tqdm(train_loader, ncols=80, desc='Training')
    running_loss = 0
    running_performance = 0
    for step, minibatch in enumerate(pbar):
        optimizer.zero_grad()  # clear the old gradients
        # 1. Get a minibatch data for training
        x, y_oht, y_seg = minibatch['input'], minibatch['ground_truth_onehot'], \
                          minibatch['ground_truth_seg']
        x = x.to(device)            # of size (batchsize, n_bands, H, W)
        y_seg = y_seg.to(device)    # of size (batchsize, 1, H, W)
        #y_oht = y_oht.to(device)    # of size (batchsize, n_classes, H, W)

        # 2. Compute the forward pass
        f_results, o_results = model(x)             # of size (batchsize, n_classes, H, W

        # 3. Compute loss, then update weights
        loss_fn.update_kd_loss_params(iters=cur_iters+step, max_iters=max_iters)
        loss = loss_fn(f_results,o_results, y_seg)
        loss.backward()  # calculate gradients
        optimizer.step()  # update weights
        
        o_results = [F.interpolate(o, size=(y_seg.size(-2), y_seg.size(-1)), mode='nearest') for o in o_results]
        # 4. Evaluate train performance
        performance = metric(o_results[-1], y_seg)

        # 5. Store the loss and metric
        running_loss = running_loss + loss.item()
        running_performance = running_performance + performance.item()

        # 6. Display losses
        result = "{}: {:.4}".format('Train loss', loss)
        pbar.set_postfix_str(result)

    # Compute the average loss and performance
    avg_loss = running_loss / len(train_loader)
    avg_performance = running_performance / len(train_loader)

    # Return the loss and performance (average-over-epoch values) of the training set
    return avg_loss, avg_performance


def val_epoch(model, val_loader, loss_fn, metric, device, n_classes=5):
    """
    Validation an epoch of validation set
    :param model: network model
    :param val_loader: validation dataloader
    :param loss_fn: loss function for the model
    :param metric: metric for computing the performance
    :param device: device used for evaluation (cpu or gpu)
    :return: average loss and performance of the validation set
    """
    model.eval()  # set model to eval mode
    pbar = tqdm(val_loader, ncols=80, desc='Validating')
    running_loss = 0
    running_performance = 0
    cm = 0
    with torch.no_grad():  # declare no gradient operations
        for step, minibatch in enumerate(pbar):
            # 1. Get a minibatch data for validation
            x, y_oht, y_seg = minibatch['input'], minibatch['ground_truth_onehot'], \
                                 minibatch['ground_truth_seg']
            x = x.to(device)            # of size (batchsize, n_bands, H, W)
            y_seg = y_seg.to(device)    # of size (batchsize, 1, H, W)
            y_oht = y_oht.to(device)    # of size (batchsize, n_classes, H, W)

            # 2. Compute the forward pass
            f_results, o_results = model(x)  # of size (batchsize, n_classes, H, W)

            # 3. Compute loss, then update weights
            #loss_fn.update_kd_loss_params(iters=cur_iters + step, max_iters=max_iters)
            loss = loss_fn(f_results, o_results, y_seg)

            o_results = [F.interpolate(o, size=(y_seg.size(-2), y_seg.size(-1)), mode='nearest') for o in o_results]

            # 4. Evaluate test performance
            performance = metric(o_results[-1], y_seg)

            # 5. Store the loss and performance
            running_loss = running_loss + loss.item()
            running_performance = running_performance + performance.item()

            cm = cm + compute_confusion_matrix(y_gt=y_seg.detach().cpu().numpy(),
                                                y_pr=o_results[-1].detach().cpu().numpy(),
                                               classes=list(range(n_classes)))

            # 6. Display losses
            result = "{}: {:.4}".format('Val loss', loss)
            pbar.set_postfix_str(result)

        # Compute the average loss and performance
        avg_loss = running_loss / len(val_loader)
        #avg_performance = running_performance / len(val_loader)
        pixel_acc, mean_acc, mean_IoU, IoU_array, _, _ = compute_eval_from_cm(cm)

        # Return the loss and performance (average-over-epoch values) of the evaluation set
        return avg_loss, mean_IoU


def training(train_cfg, optimizer, model, train_loader, val_loader, loss_fn,
             metric, device, last_epoch, best_performance, comet=None):
    """

    :param train_cfg: training configuration dict
    :param optimizer: optimizer for training
    :param model: network model
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param loss_fn: loss function
    :param metric: metric for computing the performance
    :param device: device used for training (cpu or gpu)
    :param comet: comet-ml logger
    :return:
    """
    # Number of iterations
    epoch_iters = np.int(train_loader.dataset.__len__() /
                         cfg['train_params']['batch_size'] / len(cfg['gpu_id']))
    max_iters = epoch_iters * cfg['train_params']['n_epochs']


    n_epochs = train_cfg['n_epochs']
    #best_model_indicator = 10000000
    not_improved_epochs = 0
    #scheduler = StepLR(optimizer, step_size=25, gamma=0.8)
    for epoch in range(last_epoch, n_epochs):
        print(f"\nTraining epoch {epoch + 1}/{n_epochs}")
        print("-----------------------------------")
        # Train a epoch
        with comet.train():
            train_loss, train_performance = train_epoch(optimizer, model,
                                                        train_loader, loss_fn,
                                                        metric, device, epoch=epoch, epoch_iters=epoch_iters, max_iters=max_iters)
            comet.log_metric('loss', train_loss, epoch=epoch + 1)
            comet.log_metric('performance', train_performance, epoch=epoch + 1)

        # Validate a epoch
        with comet.validate():
            val_loss, val_performance = val_epoch(model, val_loader, loss_fn,
                                                  metric, device)
            comet.log_metric('loss', val_loss, epoch=epoch + 1)
            comet.log_metric('performance', val_performance, epoch=epoch + 1)
        #scheduler.step(epoch)
        print(f"\nSummary of epoch {epoch + 1}:")
        print(f"Train loss: {train_loss:.4f}, "
              f"Train performance: {train_performance:.4f}")

        print(f"Val loss: {val_loss:.4f}, "
              f"Val performance: {val_performance:.4f} - Best: {best_performance:.4f}")
        print('=> saving checkpoint to {}'.format(
            train_cfg['save_dir'] + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_performance,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(train_cfg['save_dir'], 'checkpoint.pth.tar'))

        # Save best model only
        #if val_loss < best_model_indicator:
        if val_performance > best_performance:
            print(f'Model exceeds prev best score'
                  f'({val_performance:.4f} > {best_performance:.4f}). Saving it now.')
            #best_model_indicator = val_loss
            best_performance = val_performance

            # Write the model
            torch.save(model.state_dict(), train_cfg['save_dir'] + '/best_model.pth')

            not_improved_epochs = 0  # reset counter
        else:
            if not_improved_epochs > train_cfg['early_stop']:  # early stopping
                print(f"Stopping training early because it has not improved for "
                      f"{train_cfg['early_stop']} epochs.")
                break
            else:
                not_improved_epochs = not_improved_epochs + 1


def testing(model, classes, test_loader, metric, device, n_heads, comet, save_dir):
    """
    Test the model with the test dataset
    :param model: network model
    :param test_loader: testing dataloader
    :param metric: metric for computing the performance
    :param device: device used for testing (cpu or gpu)
    :param comet: comet logger object
    :return:
    """
    vis_path = f'{save_dir}/visual'
    os.makedirs(vis_path,exist_ok=True)
    model.eval()  # set model to eval mode
    pbar = tqdm(test_loader, ncols=80, desc='Testing')
    running_performance = [0] * n_heads
    cm = [0] * n_heads
    dices = [0] * n_heads
    acc = [0] * n_heads
    kappas = [0] * n_heads
    aa = [0] * n_heads
    count_kappa = [0] * n_heads
    #dice_coeff = utils.metrics.Dice()
    with torch.no_grad():  # declare no gradient operations, and namespacing in comet
        for step, minibatch in enumerate(pbar):
            # 1. Get a minibatch data for testing
            x, y_oht, y_seg, name = minibatch['input'], minibatch['ground_truth_onehot'], \
                                 minibatch['ground_truth_seg'], minibatch['name'][0]
            x = x.to(device)            # of size (batchsize, n_bands, H, W)
            y_seg = y_seg.to(device)    # of size (batchsize, 1, H, W)
            y_oht = y_oht.to(device)    # of size (batchsize, n_classes, H, W)

            # 2. Compute the forward pass
            f_results, o_results = model(x)  # of size (batchsize, n_classes, H, W)
            o_results = [F.interpolate(o, size=(y_seg.size(-2), y_seg.size(-1)), mode='nearest') for o in o_results]
            for i in range(n_heads):
                # 3. Compute the performance of the testing minibatch
                performance = metric(o_results[i], y_seg)

                # 4. Store the performance
                running_performance[i] = running_performance[i] + performance.item()

                # # 5. Show visual results
                test_loader.dataset.save_pred(y_oht, sv_path=f'{vis_path}', name=f'{name}_gt.png')
                test_loader.dataset.save_pred(o_results[i],sv_path=f'{vis_path}',name=f'{name}_clf{i}.png')

                # 6. Compute the confusion matrix
                cm[i] = cm[i] + compute_confusion_matrix(y_seg.detach().cpu().numpy(),
                                                   o_results[i].detach().cpu().numpy(),
                                                   classes=classes)

                # 7. Test dice
                dices[i] += utils.metrics.dice_coeff(o_results[i], y_seg)

                # 8. Compute Accuracy
                acc[i] += utils.metrics.accuracy(o_results[i], y_seg)

                # 9. Compute Avg Accuracy
                aa[i] += utils.metrics.average_accuracy(o_results[i], y_seg)

                # 10. Compute Kappa
                #k = utils.metrics.kappa(y_pr, y_seg)
                k = utils.metrics.kappa(o_results[i], y_seg)
                if not np.isnan(k):
                    kappas[i] += k
                    count_kappa[i] += 1
            #kappa += utils.metrics.kappa(y_pr, y_seg)

    for i in range(n_heads):
        pixel_acc, mean_acc, mean_IoU, IoU_array, dice, kappa = compute_eval_from_cm(cm[i])
        # Compute the average performance of the test set
        avg_performance = running_performance[i] / len(test_loader)
        avg_dice = dices[i] / len(test_loader)

        # Add the average performance value into the log_metric of comet object
        print(f"Testing performance clf {i}: {avg_performance:.4f}")
        comet.log_metric(f'test_performance_{i}', mean_IoU)

        print(f"Testing dice clf {i}: {avg_dice:.4f}")
        comet.log_metric(f'test_dice_{i}', dice)

        # pos = cm[0].sum(1)
        # res = cm[0].sum(0)
        # tp = np.diag(cm[0])
        # pixel_acc = tp.sum() / pos.sum()
        # mean_acc = (tp / np.maximum(1.0, pos)).mean()
        # IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # mean_IoU = IoU_array.mean()
        print(f"Testing all-mIoU clf {i}: {mean_IoU}")
        print(f"Testing all-acc clf{i}: {pixel_acc}")
        print(f"Testing all-OA clf {i}: {mean_acc}")
        print(f"IoU Array {i}", IoU_array)
        comet.log_metric(f"test_iou_array_{i}", IoU_array)


        # Log the confusion matrix
        comet.log_confusion_matrix(matrix=cm[i], labels=classes)

        avg_acc = acc[i] / len(test_loader)
        print(f"Overall Accuracy clf {i}: {avg_acc:.4f}")
        comet.log_metric(f"test_oa_{i}", pixel_acc)

        avg_aa = aa[i] / len(test_loader)
        print(f"Average Accuracy Clf {i}: {avg_aa:.4f}")
        comet.log_metric(f"test_aa_{i}", mean_acc)

        avg_kappa = kappas[i] / count_kappa[i]
        print(f"Average Kappa Clf {i}: {avg_kappa:.4f}")
        comet.log_metric(f"test_kappa_{i}", kappa)

def init_seeds(seed):
    """

    :param seed:
    :return:
    """
    # Setting seeds
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# def convert_prob2seg_tensor(y_prob, classes):
#     y_class = np.argmax(y_prob, axis=0)
#     y_seg = np.zeros((y_prob.shape[1], y_prob.shape[2]))

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Main file')
    args.add_argument('--config', default='configs/base.yml', type=str,
                      help='config file path (default: None)')
    args.add_argument('--debug', default=0, type=int,
                      help='debug mode? (default: 0')
    cmd_args = args.parse_args()

    assert cmd_args.config is not None, "Please specify a config file"

    # Configuring comet-ml logger
    api_key_path = "./configs/comet-ml-key.txt"
    if os.path.isfile(api_key_path) and os.access(api_key_path, os.R_OK):
        f = open(api_key_path, "r")
        comet_key = f.read()
        f.close()
    else:
        raise FileNotFoundError(
            'You need to create a textfile containing only the comet-ml api key. '
            'The full path should be ./configs/comet-ml-key.txt')

    comet = Experiment(api_key=comet_key,
                       project_name="uow-hsi-kdm",
                       workspace="hieuphan",
                       disabled=False,
                       auto_metric_logging=False)

    # Read experiment configurations
    nn_config_path = cmd_args.config
    with open(nn_config_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    if cmd_args.debug == 1:
        cfg['train_params']['dataset_dir'] = '/home/hieu/research/data/UOW-HSI'
        cfg['train_params']['num_workers'] = 0
        cfg['batch_size'] = 1
        cfg['warmup_epochs'] = 1
        cfg['debug_mode'] = 1
        print('DEBUG mode')
        save_dir = 'experiments/test-folder'
        create_exp_dir(save_dir, visual_folder=True)
    elif cfg['train_params']['save_dir'] == '':
        # If not debug, we create a folder to store the model weights and etc
        save_dir = f'experiments/{cfg["name"]}-{time.strftime("%Y%m%d-%H%M%S")}'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        save_dir = f"experiments/{cfg['train_params']['save_dir']}"

    cfg['train_params']['save_dir'] = save_dir
    comet.set_name('%s-%depochs' % (cfg['name'], cfg['train_params']['n_epochs']))
    comet.log_asset(nn_config_path)
    comet.add_tags(cfg['tags'])
    #
    main(cfg, comet)
    #debug(cfg)
