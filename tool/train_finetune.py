import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import functools
from torch.nn import init
import collections

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from collections import OrderedDict

from util import dataset, transform, config
from util import finetune_dataset as dataset
from util import finetune_transform as transform
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, batchPSNRandSSIMGPU, find_free_port, intersectionAndUnionCPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == 'iterative_derain_seg':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        pass
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    args = get_parser()
    # check(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.seg_loss_type == 'ce':
        if args.ohem:
            min_kept = int(args.batch_size // len(
                args.train_gpu) * args.train_h * args.train_w // 16)
            seg_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                                   min_kept=min_kept,
                                                   use_weight=False)
        else:
            seg_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    else:
        raise NotImplementedError

    if args.derain_loss_type == 'mse':
        derain_criterion = nn.MSELoss()
    else:
        raise NotImplementedError

    if args.arch == 'iterative_derain_seg':
        from model.dic_arch_derainseg_fineutne import DIC
        model = DIC(args,
                    derain_criterion=derain_criterion,
                    seg_criterion=seg_criterion,
                    is_train=True)
        modules_ori = [model.seg_net]
        modules_new = [model.block, model.first_block, model.conv_in,
                       model.conv_out, model.derain_final_conv]
        modules_fix = [model.edge_net]
    else:
        raise NotImplementedError

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 0))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 1))
    for module in modules_fix:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 0))
    args.index_split_1 = 1
    args.index_split_2 = 6
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.pretrained:
        if main_process():
            logger.info("=> Loading derain first weight from '{}'\n "
                        "and '{}'\n seg weight from '{}'".format(args.derain_first_pretrained_path,
                                                                 args.derain_last_pretrained_path,
                                                                 args.seg_pretrained_path))
        load_derain_and_seg(model, args)
        if main_process():
            logger.info("=> Loaded derain first weight from '{}'\n "
                        "and '{}'\n seg weight from '{}'".format(args.derain_first_pretrained_path,
                                                                 args.derain_last_pretrained_path,
                                                                 args.seg_pretrained_path))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        # transform.RandScale([args.scale_min, args.scale_max]),
        # transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        # transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    train_data = dataset.SemData(split='train', data_root=args.data_root, rain_data_root=args.rain_data_root,
                                 data_list=args.train_list, transform=train_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)
    # if args.evaluate:
    #     val_transform = transform.Compose([
    #         transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean, ignore_label=args.ignore_label),
    #         transform.ToTensor(),
    #         transform.Normalize(mean=mean, std=std)])
    #     val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
    #     if args.distributed:
    #         val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    #     else:
    #         val_sampler = None
    #     val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train, psnr_train, ssim_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('psnr_train', psnr_train, epoch_log)
            writer.add_scalar('ssim_train', ssim_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            # if epoch_log / args.save_freq > 2:
            #     deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
            #     os.remove(deletename)
        # if args.evaluate:
        #     loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
        #     if main_process():
        #         writer.add_scalar('loss_val', loss_val, epoch_log)
        #         writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
        #         writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
        #         writer.add_scalar('allAcc_val', allAcc_val, epoch_log)


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    derain_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    list_multiply = lambda x, y: x * y
    assert len(args.seg_loss_step_weight) == args.num_steps

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (clear_label, rain_input) in enumerate(train_loader):
        data_time.update(time.time() - end)

        clear_label = clear_label.cuda(non_blocking=True)
        rain_input = rain_input.cuda(non_blocking=True)
        derain_output, derain_losses = model(rain_input, clear_label)
        derain_losses = map(list_multiply, derain_losses, args.derain_loss_step_weight)
        derain_sum_loss = sum(derain_losses)
        if not args.multiprocessing_distributed:
            derain_sum_loss = torch.mean(derain_sum_loss)
        loss = args.derain_loss_weight * derain_sum_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = rain_input.size(0)
        if args.multiprocessing_distributed:
            derain_sum_loss, loss = derain_sum_loss.detach() * n, \
                                    loss * n  # not considering ignore pixels
            count = clear_label.new_tensor([n], dtype=torch.long)
            dist.all_reduce(derain_sum_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            derain_sum_loss, loss = derain_sum_loss / n, loss / n

        # intersection, union, target = intersectionAndUnionCPU(seg_output, seg_label, args.classes, args.ignore_label)
        psnr, ssim = batchPSNRandSSIMGPU(derain_output, clear_label)
        # if args.multiprocessing_distributed:
        #     dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        psnr_meter.update(psnr), ssim_meter.update(ssim)

        # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        accuracy = 0
        psnr_val = psnr_meter.val
        ssim_val = ssim_meter.val
        derain_loss_meter.update(derain_sum_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        for index in range(0, args.index_split_1):
            optimizer.param_groups[index]['lr'] = current_lr * 0
        for index in range(args.index_split_1, args.index_split_2):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        for index in range(args.index_split_2, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 0
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'DerainLoss {derain_loss_meter:.4f} '
                        'SegLoss {seg_loss_meter:.4f} '
                        'Loss {loss_meter:.4f} '
                        'Accuracy {accuracy:.4f}.'
                        'PSNR {psnr_val:.2f}.'
                        'SSIM {ssim_val:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                      batch_time=batch_time,
                                                      data_time=data_time,
                                                      remain_time=remain_time,
                                                      derain_loss_meter=derain_loss_meter.val,
                                                      seg_loss_meter=seg_loss_meter.val,
                                                      loss_meter=loss_meter.val,
                                                      accuracy=accuracy,
                                                      psnr_val=psnr_val,
                                                      ssim_val=ssim_val))

        if main_process():
            writer.add_scalar('derain_loss_train_batch', derain_loss_meter.val, current_iter)
            writer.add_scalar('seg_loss_train_batch', seg_loss_meter.val, current_iter)
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            # writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            # writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
            writer.add_scalar('psnr_train_batch', psnr_val, current_iter)
            writer.add_scalar('ssim_train_batch', ssim_val, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    mIoU = 0
    mAcc = 0
    allAcc = 0
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
        logger.info(
            'Train result at epoch [{}/{}]: PSNR/SSIM {:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, psnr_meter.avg,
                                                                                           ssim_meter.avg))
    return loss_meter.avg, mIoU, mAcc, allAcc, psnr_meter.avg, ssim_meter.avg



# def validate(val_loader, model, criterion):
#     if main_process():
#         logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     loss_meter = AverageMeter()
#     intersection_meter = AverageMeter()
#     union_meter = AverageMeter()
#     target_meter = AverageMeter()
#
#     model.eval()
#     end = time.time()
#     for i, (input, target) in enumerate(val_loader):
#         data_time.update(time.time() - end)
#         input = input.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)
#         output = model(input)
#         if args.zoom_factor != 8:
#             output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
#         loss = criterion(output, target)
#
#         n = input.size(0)
#         if args.multiprocessing_distributed:
#             loss = loss * n  # not considering ignore pixels
#             count = target.new_tensor([n], dtype=torch.long)
#             dist.all_reduce(loss), dist.all_reduce(count)
#             n = count.item()
#             loss = loss / n
#         else:
#             loss = torch.mean(loss)
#
#         output = output.max(1)[1]
#         intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
#         if args.multiprocessing_distributed:
#             dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
#         intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
#         intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
#
#         accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
#         loss_meter.update(loss.item(), input.size(0))
#         batch_time.update(time.time() - end)
#         end = time.time()
#         if ((i + 1) % args.print_freq == 0) and main_process():
#             logger.info('Test: [{}/{}] '
#                         'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
#                         'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
#                         'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
#                         'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
#                                                           data_time=data_time,
#                                                           batch_time=batch_time,
#                                                           loss_meter=loss_meter,
#                                                           accuracy=accuracy))
#
#     iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
#     accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
#     mIoU = np.mean(iou_class)
#     mAcc = np.mean(accuracy_class)
#     allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
#     if main_process():
#         logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
#         for i in range(args.classes):
#             logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
#         logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
#     return loss_meter.avg, mIoU, mAcc, allAcc


def _load_func(m, d):
    d_copy = collections.OrderedDict()
    for key in d.keys():
        if 'module.' in key:
            d_copy[key.replace('module.', '')] = d[key]
    res = m.module.load_state_dict(d_copy, strict=False)

    res_str = ''
    if len(res.missing_keys) != 0:
        res_str += 'missing_keys: ' + ', '.join(res.missing_keys)
    if len(res.unexpected_keys) != 0:
        res_str += 'unexpected_keys: ' + ', '.join(res.unexpected_keys)
    if len(res_str) == 0:  # strictly fit
        res_str = 'Strictly loaded!'
    return res_str


def load_derain_and_seg(model, args):
    model = _net_init(model)

    if os.path.isfile(args.edge_pretrained_path):
        print("=> loading edge checkpoint '{}'".format(args.edge_pretrained_path))
        checkpoint = torch.load(args.edge_pretrained_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded edge checkpoint '{}'".format(args.edge_pretrained_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.edge_pretrained_path))

    if args.derain_first_pretrained_path is not None:
        print('===> Loading Derain model from %s' %
              args.derain_first_pretrained_path)
        checkpoint = torch.load(args.derain_first_pretrained_path)
        res_str = _load_func(model, checkpoint)
        print(res_str)

    if args.derain_last_pretrained_path is not None:
        print('===> Loading Derain model from %s' %
              args.derain_last_pretrained_path)
        checkpoint = torch.load(args.derain_last_pretrained_path)
        res_str = _load_func(model, checkpoint)
        print(res_str)

    if args.seg_pretrained_path is not None:
        print('===> Loading SegNet model from %s' %
              args.seg_pretrained_path)
        model.module.seg_net = load_seg_model(model.module.seg_net, args)


def load_seg_model(model, args, is_restore=False):
    t_start = time.time()
    state_dict = torch.load(args.seg_pretrained_path, map_location=torch.device('cpu'))
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        print('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        print('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    print(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model


def _net_init(model, init_type='kaiming'):
    print('==> Initializing the network using [%s]' % init_type)
    return init_weights(model, init_type)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
        return net
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
        return net
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
        return net
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            # print('initializing [%s] ...' % classname)
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            # print('initializing [%s] ...' % classname)
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            # print('initializing [%s] ...' % classname)
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [1.4297, 1.4805, 1.4363, 3.365, 2.6635, 1.4311, 2.1943, 1.4817,
                 1.4513, 2.1984, 1.5295, 1.6892, 3.2224, 1.4727, 7.5978, 9.4117,
                 15.2588, 5.6818, 2.2067])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = torch.sort(mask_prob)
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, ignore_label, gamma=2.0, alpha=0.25,
                 reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = (target.ne(self.ignore_label)).float()
        target = mask * target
        onehot = target.view(b, -1, 1)

        # TODO: use the pred instead of pred_sigmoid
        max_val = (-pred_sigmoid).clamp(min=0)

        pos_part = (1 - pred_sigmoid) ** self.gamma * (
                pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + (
                (-max_val).exp() + (-pred_sigmoid - max_val).exp()).log())

        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(
            dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


if __name__ == '__main__':
    main()
