import datetime
import logging
import os
import time

import torch as T
import wandb
from nn_isomorphics.utils.utils import (AverageMeter, MetricLogger,
                                        ProgressMeter, SmoothedValue, Summary,
                                        accuracy, ifnot_create)

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


def validate_imagenet(val_dl, model):
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(len(val_dl), [batch_time, acc1, acc5],
                             prefix=f"Test: ")

    # Get the initialization test time
    end = time.time()

    model.to(DEVICE)

    with T.no_grad():
        for i, (images, targets) in enumerate(val_dl):
            # Inference
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            output = model(images)

            # measure accuracy and record loss
            top1, top5 = accuracy(output, targets, topk=(1, 5))
            acc1.update(top1[0].item(), images.size(0))
            acc5.update(top5[0].item(), images.size(0))

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if i % 100 == 0:
                progress.display(i + 1)

    # print metrics
    progress.display_summary()


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    epoch,
    config,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s",
                            SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
            metric_logger.log_every(data_loader, config['print_freq'],
                                    header)):
        start_time = time.time()
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        if config['clip_grad_norm'] is not None:
            T.nn.utils.clip_grad_norm_(model.parameters(),
                                       config['clip_grad_norm'])
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size /
                                             (time.time() - start_time))

    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.loss.global_avg


def evaluate(model, criterion, data_loader, config, log_suffix=""):
    model.to(DEVICE)
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with T.inference_mode():
        for image, target in metric_logger.log_every(data_loader,
                                                     config['print_freq'],
                                                     header):
            image = image.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    # gather the stats from all processes
    logging.info(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.loss.global_avg


def train_imagenet(
    model,
    criterion,
    train_dl,
    val_dl,
    config,
    optimizer,
    lr_scheduler,
):
    print("Start training")
    start_time = time.time()

    model.to(DEVICE)
    for epoch in range(config['start_epoch'], config['epochs']):
        train_acc1, train_acc5, train_loss = train_one_epoch(
            model, criterion, optimizer, train_dl, epoch, config)
        lr_scheduler.step()

        eval_acc1, eval_acc5, eval_loss = evaluate(model, criterion, val_dl,
                                                   config)

        wandb.log({
            'train_acc1': train_acc1,
            'train_acc5': train_acc5,
            'train_loss': train_loss,
            'eval_acc1': eval_acc1,
            'eval_acc5': eval_acc5,
            'eval_loss': eval_loss
        })

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        # "epoch": epoch,
        "config": config,
    }

    T.save(checkpoint, ifnot_create(os.path.join(config['model_path'])))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
