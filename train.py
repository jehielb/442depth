import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
import torchvision.transforms
from tensorboardX import SummaryWriter
from PIL import Image
from model import PTModel as Model
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize_depth
import gc

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run') # 20 epoch default previously
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()

    # Create model
    model = Model().cuda()
    print('Model created.')

    # # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()

    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            # breakpoint()
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
            depth = torchvision.transforms.Resize((320, 240))(depth)

            # Normalize depth
            depth_n = DepthNorm( depth )

            # Predict
            # 320, 240
            output = model(image)

            l_depth = l1_criterion(output.squeeze(), depth_n)
            # breakpoint()
            l_ssim = torch.clamp((1 - ssim(output.squeeze().unsqueeze(0), depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
            # l_ssim = 0
            loss = (1.0 * l_ssim) + (0.1 * l_depth)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 5 == 0:
                LogProgress(model, writer, test_loader, niter)
            

        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image2 = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    image = torchvision.transforms.Resize((320, 240))(image2)
    depth = torchvision.transforms.Resize((320, 240))(depth)
    # breakpoint()
    # breakpoint()
    output = DepthNorm(model(image2)).squeeze()                 # TODO: might need to get rid of DepthNorm call
    # breakpoint()
    for i in range(output.shape[0]):
        if epoch == 0: writer.add_image(f'Train.1.Image-{i}', vutils.make_grid(image.data[i], nrow=6, normalize=True), epoch)
        if epoch == 0: writer.add_image(f'Train.2.Depth-{i}', colorize_depth(depth.data, i, 'viridis'), epoch) 
        writer.add_image(f'Train.3.Ours-{i}', colorize_depth(output.data, i, colormap='viridis'), epoch)
        writer.add_image(f'Train.3.Diff-{i}', colorize_depth(depth.data-output.data, i, colormap='viridis'), epoch)
    del image
    del depth
    del output
    gc.collect()

if __name__ == '__main__':
    main()



            # Compute the loss
            # # breakpoint()
            # output = torch.squeeze(output, dim=1)
            # output_np = output.detach().cpu().numpy()
            # # Transpose dimensions if necessary (e.g., from CHW to HWC)
            # output_np = np.transpose(output_np, (1, 2, 0))
            # # Rescale pixel values to the range [0, 255]
            # output_np = (output_np * 255).astype(np.uint8)
            # # Convert NumPy array to PIL image
            # output_img = Image.fromarray(output_np)
            # # Save the image to a file
            # output_img.save('output_image.png')