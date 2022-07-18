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

# Local application imports
import utils.losses
import utils.metrics

import models.hsi_net
from models.hsi_net import HSINet, HSINet1c, SGR_Net
from utils.utils import show_visual_results, create_exp_dir, prepare_device, \
    get_experiment_dataloaders, init_obj, compute_confusion_matrix, test_running_time


def adjust_learning_rate(optimizer, base_lr, epoch, step_size):
    """Decay the LR by 0.8 every 30 epochs, reset the lr to the base_lr after 'resets' number of epochs"""
    #interval_id = epoch // step_size
    left_epochs = epoch % step_size
    lr = base_lr * (0.8 ** (left_epochs // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def learning_rate(base_lr, epoch, stepsize, decay_steps=80):
    """Decay the LR by 0.8 every 30 epochs, reset the lr to the base_lr after 'resets' number of epochs"""
    lr = base_lr * (0.8** (epoch%stepsize)//decay_steps)
    return lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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
    teacher_params = cfg['teacher_params']
    teacher_model = getattr(models.hsi_net, teacher_params['name'])(n_bands=teacher_params['n_bands'],
                                                      classes=teacher_params['classes'],
                                                      nf_enc=teacher_params['nf_enc'],
                                                      nf_dec=teacher_params['nf_dec'],
                                                      do_batchnorm=teacher_params['do_batchnorm'],
                                                      n_heads=teacher_params['n_heads'],
                                                      max_norm_val=None,
                                                      rates=teacher_params['rates'])

    student_params = cfg['student_params']
    if ('name' not in student_params) or \
            ('name' in student_params and student_params['name'] == 'HSINet'):
        student_model = HSINet(n_bands=student_params['n_bands'],
                       classes=student_params['classes'],
                       nf_enc=student_params['nf_enc'],
                       nf_dec=student_params['nf_dec'],
                       do_batchnorm=student_params['do_batchnorm'],
                       max_norm_val=None)
    else:
        student_model = eHSINet(segnet=student_params['name'],
                        n_bands=student_params['n_bands'],
                        classes=len(student_params['classes']),
                        do_batchnorm=student_params['do_batchnorm'],
                        max_norm_val=None)
    # Get dataloaders
    t_params = cfg['train_params']
    t_params['classes'] = student_params['classes']
    train_loader, val_loader, test_loader = get_experiment_dataloaders(cfg['train_params'])
    #
    # # Log the trainable parameters
    x = test_loader.dataset[0]['input'].unsqueeze(0)
    if 'cuda' in device.type:
        x = x.to('cuda:0')
    # print(model.device)
    # print(x.device)
    flops, params = profile(student_model.to('cuda:0'), inputs=(x.to('cuda:0'),), verbose=False)
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPS: {}, PARAMS: {}".format(flops, params))
    comet.log_other('Student model trainable parameters', params)
    comet.log_other('Floating point operations per second (FLOPS)', flops)
    comet.log_other('Multiply accumulates per second (MACs)', macs)

    test_running_time(x,student_model,comet)
    student_model = torch.nn.DataParallel(student_model).cuda()
    student_model = student_model.to(device)

    flops, params = profile(teacher_model.to('cuda:0'), inputs=(x.to('cuda:0'),), verbose=False)
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPS: {}, PARAMS: {}".format(flops, params))
    comet.log_other('Teacher model trainable parameters', params)
    comet.log_other('Floating point operations per second (FLOPS)', flops)
    comet.log_other('Multiply accumulates per second (MACs)', macs)

    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    teacher_model = teacher_model.to(device)
    saved = torch.load(teacher_params['pretrained_file'])
    teacher_model.load_state_dict(saved)


    # Define an optimiser
    print(f"Optimiser for model weights: {cfg['optimizer']['type']}")
    optimizer = init_obj(cfg['optimizer']['type'],
                         cfg['optimizer']['args'],
                         torch.optim, student_model.parameters())
    print(cfg['optimizer']['args'])

    best_performance = 0
    last_epoch = 0
    if cfg['resume']:
        model_state_file = os.path.join(cfg['train_params']['save_dir'],
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_performance = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    optimizer = init_obj(cfg['optimizer']['type'],
                         cfg['optimizer']['args'],
                         torch.optim, student_model.parameters())

    # Metrics
    metric = getattr(utils.metrics, cfg['metric'])(activation='softmax2d')

    # # Loss function
    loss_fn = getattr(utils.losses, cfg['loss'])(feat_w=student_params['feat_weight'],
                                                 resp_w=student_params['resp_weight'])

    # Training
    training(cfg, optimizer, student_model, teacher_model, train_loader, val_loader,
             loss_fn, metric, device, last_epoch, best_performance, comet)

    # Testing
    print("\nThe training has completed. Testing the model now...")

    # Load the best model
    saved = torch.load(cfg['train_params']['save_dir'] + '/best_model.pth')
    student_model.load_state_dict(saved)
    testing(student_model, student_params['classes'], test_loader, metric, device, comet, cfg['train_params']['save_dir'])

    comet.log_asset(cfg['train_params']['save_dir'] + '/best_model.pth')

    test_running_time(test_loader.dataset[0]['input'].unsqueeze(0),student_model,comet)


def train_epoch(optimizer, student_model, teacher_model, train_loader, loss_fn, metric, device):
    """
    Train an epoch for the dataset. Logic for training an epoch:
        1. Get a minibatch for training
        2. Compute the forward path
        3. Compute the loss and update the weight
        4. Evaluate train performance
        5. Store the loss and metric
        6. Display losses
    :param optimizer: optimizer for the model
    :param student_model: network model
    :param train_loader: training dataloader
    :param loss_fn: loss function for the model
    :param metric: metric for computing the performance
    :param device: device used for training (cpu or gpu)
    :return: train the network model for the
    """
    #cur_iters = epoch * epoch_iters
    student_model.train()
    teacher_model.eval()
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
        student_f, student_o = student_model(x)             # of size (batchsize, n_classes, H, W

        # Compute teacher output
        with torch.no_grad():
            teacher_f, teacher_o = teacher_model(x)
        # 3. Compute loss, then update weights
        #loss_fn.update_kd_loss_params(iters=cur_iters+step, max_iters=max_iters)
        loss = loss_fn(student_o, teacher_o, student_f, teacher_f, y_seg)
        #loss = loss_fn(f_results,o_results, y_seg)
        loss.backward()  # calculate gradients
        optimizer.step()  # update weights

        # 4. Evaluate train performance
        performance = metric(student_o, y_seg)

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


def val_epoch(student_model, teacher_model, val_loader, loss_fn, metric, device):
    """
    Validation an epoch of validation set
    :param model: network model
    :param val_loader: validation dataloader
    :param loss_fn: loss function for the model
    :param metric: metric for computing the performance
    :param device: device used for evaluation (cpu or gpu)
    :return: average loss and performance of the validation set
    """
    student_model.eval()  # set model to eval mode
    pbar = tqdm(val_loader, ncols=80, desc='Validating')
    running_loss = 0
    running_performance = 0
    with torch.no_grad():  # declare no gradient operations
        for step, minibatch in enumerate(pbar):
            # 1. Get a minibatch data for validation
            x, y_oht, y_seg = minibatch['input'], minibatch['ground_truth_onehot'], \
                                 minibatch['ground_truth_seg']
            x = x.to(device)            # of size (batchsize, n_bands, H, W)
            y_seg = y_seg.to(device)    # of size (batchsize, 1, H, W)

            # 2. Compute the forward pass
            # 2. Compute the forward pass
            student_f, student_o = student_model(x)  # of size (batchsize, n_classes, H, W

            # Compute teacher output
            with torch.no_grad():
                teacher_f, teacher_o = teacher_model(x)
            # 3. Compute loss, then update weights
            # loss_fn.update_kd_loss_params(iters=cur_iters+step, max_iters=max_iters)
            loss = loss_fn(student_o, teacher_o, student_f, teacher_f, y_seg)

            # 4. Evaluate test performance
            performance = metric(student_o, y_seg)

            # 5. Store the loss and performance
            running_loss = running_loss + loss.item()
            running_performance = running_performance + performance.item()

            # 6. Display losses
            result = "{}: {:.4}".format('Val loss', loss)
            pbar.set_postfix_str(result)

        # Compute the average loss and performance
        avg_loss = running_loss / len(val_loader)
        avg_performance = running_performance / len(val_loader)

        # Return the loss and performance (average-over-epoch values) of the evaluation set
        return avg_loss, avg_performance


def training(cfg, optimizer, student_model, teacher_model, train_loader, val_loader, loss_fn,
             metric, device, last_epoch, best_performance, comet=None):
    """

    :param train_cfg: training configuration dict
    :param optimizer: optimizer for training
    :param student_model: network model
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param loss_fn: loss function
    :param metric: metric for computing the performance
    :param device: device used for training (cpu or gpu)
    :param comet: comet-ml logger
    :return:
    """
    # epoch_iters = np.int(train_loader.dataset.__len__() /
    #                      cfg['train_params']['batch_size'] / len(cfg['gpu_id']))
    # max_iters = epoch_iters * cfg['train_params']['n_epochs']

    train_cfg = cfg['train_params']
    n_epochs = train_cfg['n_epochs']
    n_networks = cfg['teacher_params']['n_heads']
    resets = n_epochs // n_networks
    #best_model_indicator = 10000000
    not_improved_epochs = 0
    for epoch in range(last_epoch, n_epochs):
        #adjust_learning_rate(optimizer=optimizer, base_lr=cfg['optimizer']['args']['lr'], epoch=epoch, step_size=resets)

        print(f"\nTraining epoch {epoch + 1}/{n_epochs}")
        print("-----------------------------------")
        # Train a epoch
        with comet.train():
            train_loss, train_performance = train_epoch(optimizer, student_model, teacher_model,
                                                        train_loader, loss_fn,
                                                        metric, device)
            comet.log_metric('loss', train_loss, epoch=epoch + 1)
            comet.log_metric('performance', train_performance, epoch=epoch + 1)

        # Validate a epoch
        with comet.validate():
            val_loss, val_performance = val_epoch(student_model, teacher_model, val_loader, loss_fn,
                                                  metric, device)
            comet.log_metric('loss', val_loss, epoch=epoch + 1)
            comet.log_metric('performance', val_performance, epoch=epoch + 1)

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
            'state_dict': student_model.state_dict(),
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
            torch.save(student_model.state_dict(), train_cfg['save_dir'] + '/best_model.pth')

            not_improved_epochs = 0  # reset counter
        else:
            if not_improved_epochs > train_cfg['early_stop']:  # early stopping
                print(f"Stopping training early because it has not improved for "
                      f"{train_cfg['early_stop']} epochs.")
                break
            else:
                not_improved_epochs = not_improved_epochs + 1
        lr = get_lr(optimizer)
        print("Learning rate", lr)

def testing(model, classes, test_loader, metric, device, comet, save_dir):
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
    running_performance = 0
    cm = 0
    dice = 0
    acc = 0
    kappa = 0
    aa = 0
    count_kappa = 0
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
            _, o = model(x)  # of size (batchsize, n_classes, H, W)
            # 3. Compute the performance of the testing minibatch
            performance = metric(o, y_seg)

            # 4. Store the performance
            running_performance = running_performance + performance.item()

            # # 5. Show visual results
            test_loader.dataset.save_pred(y_oht, sv_path=f'{vis_path}', name=f'{name}_gt.png')
            test_loader.dataset.save_pred(o,sv_path=f'{vis_path}',name=f'{name}_pred.png')

            # 6. Compute the confusion matrix
            cm = cm + compute_confusion_matrix(y_seg.detach().cpu().numpy(),
                                               o.detach().cpu().numpy(),
                                               classes=classes)

            # 7. Test dice
            dice += utils.metrics.dice_coeff(o, y_seg)

            # 8. Compute Accuracy
            acc += utils.metrics.accuracy(o, y_seg)

            # 9. Compute Avg Accuracy
            aa += utils.metrics.average_accuracy(o, y_seg)

            # 10. Compute Kappa
            #k = utils.metrics.kappa(y_pr, y_seg)
            k = utils.metrics.kappa(o, y_seg)
            if not np.isnan(k):
                kappa += k
                count_kappa += 1
            #kappa += utils.metrics.kappa(y_pr, y_seg)

    # Compute the average performance of the test set
    avg_performance = running_performance / len(test_loader)
    avg_dice = dice / len(test_loader)

    # Add the average performance value into the log_metric of comet object
    print(f"Testing performance: {avg_performance:.4f}")
    comet.log_metric(f'test_performance', avg_performance)

    print(f"Testing dice: {avg_dice:.4f}")
    comet.log_metric(f'test_dice', avg_dice)

    # Log the confusion matrix
    comet.log_confusion_matrix(matrix=cm, labels=classes)

    avg_acc = acc / len(test_loader)
    print(f"Overall Accuracy: {avg_acc:.4f}")
    comet.log_metric(f"test_oa", avg_acc)

    avg_aa = aa / len(test_loader)
    print(f"Average Accuracy: {avg_aa:.4f}")
    comet.log_metric(f"test_aa", avg_aa)

    avg_kappa = kappa / count_kappa
    print(f"Average Kappa Clf: {avg_kappa:.4f}")
    comet.log_metric(f"test_kappa", avg_kappa)

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
                       project_name="uow-hsi",
                       workspace="hieuphan",
                       disabled=bool(cmd_args.debug),
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

    main(cfg, comet)
