import time
import os
import torch
import copy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from options import get_train_options
from datasets import UnalignedDataset
from models import CycleGANModel
from utils import ImagePool, get_scheduler

def main():
    opt = get_train_options()
    
    # Dataset
    dataset = UnalignedDataset(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads),
        drop_last=False
    )
    dataset_size = len(dataset)
    print(f'The number of training images = {dataset_size}')

    # Validation Dataset
    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'val' # using test or val folder
    try:
        val_dataset = UnalignedDataset(val_opt)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads),
            drop_last=True
        )
        val_dataset_size = len(val_dataset)
        print(f'The number of validation images = {val_dataset_size}')
        has_val = True
    except Exception as e:
        print(f'Could not load validation dataset: {e}')
        has_val = False

    # Tensorboard initialization
    writer_train = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name, 'train'))
    if has_val:
        writer_val = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name, 'val'))

    # Model definition
    model = CycleGANModel(opt)
    
    # Schedulers
    schedulers = [get_scheduler(optimizer, opt) for optimizer in model.optimizers]

    # Image Pools (for discriminator)
    fake_A_pool = ImagePool(opt.pool_size)
    fake_B_pool = ImagePool(opt.pool_size)

    total_iters = 0

    # Training loop
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        epoch_losses_sum = {}
        num_train_batches = 0

        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # Setup input
            model.set_input(data)
            
            # Forward & Backward G
            model.forward()
            
            # G parameters update
            # temporarily freeze D
            for netD in [model.netD_A, model.netD_B]:
                for param in netD.parameters():
                    param.requires_grad = False
            
            model.optimizer_G.zero_grad()
            model.backward_G()
            model.optimizer_G.step()

            # D parameters update
            # unfreeze D
            for netD in [model.netD_A, model.netD_B]:
                for param in netD.parameters():
                    param.requires_grad = True

            fake_B = fake_B_pool.query(model.fake_B)
            # Need to update model state so backward_D_A uses queried pool image
            model.fake_B = fake_B
            model.optimizer_D.zero_grad()
            model.backward_D_A()
            
            fake_A = fake_A_pool.query(model.fake_A)
            # Need to update model state so backward_D_B uses queried pool image
            model.fake_A = fake_A
            model.backward_D_B()
            
            model.optimizer_D.step()

            # Accumulate losses per epoch
            losses = {
                'G_A': model.loss_G_A.item(),
                'G_B': model.loss_G_B.item(),
                'Cyc_A': model.loss_cycle_A.item(),
                'Cyc_B': model.loss_cycle_B.item(),
                'D_A': model.loss_D_A.item(),
                'D_B': model.loss_D_B.item(),
            }
            if opt.lambda_identity > 0.0:
                losses['idt_A'] = model.loss_idt_A.item()
                losses['idt_B'] = model.loss_idt_B.item()
            if opt.lambda_perceptual > 0.0:
                losses['VGG'] = model.loss_perceptual.item()
            if opt.lambda_arcface > 0.0:
                losses['Arc'] = model.loss_arcface.item()
                
            for k, v in losses.items():
                epoch_losses_sum[k] = epoch_losses_sum.get(k, 0.0) + v
            num_train_batches += 1

            # Print losses
            if total_iters % opt.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                
                message = f'(epoch: {epoch}, iters: {epoch_iter}, time: {t_comp:.3f}, data: {t_data:.3f}) '
                for k, v in losses.items():
                    message += f'{k}: {v:.3f} '
                print(message)

            if total_iters % opt.save_latest_freq == 0:
                print(f'saving the latest model (epoch {epoch}, total_iters {total_iters})')
                model.save_networks('latest')

            iter_data_time = time.time()

        # Log epoch average training losses to TensorBoard
        if num_train_batches > 0:
            for k, v in epoch_losses_sum.items():
                avg_v = v / num_train_batches
                writer_train.add_scalar(f'Loss/{k}', avg_v, epoch)

        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation loop
        if has_val and epoch % opt.val_freq == 0:
            print(f'Running validation at epoch {epoch}...')
            val_losses_sum = {}
            num_val_batches = 0
            for i, data in enumerate(val_dataloader):
                model.set_input(data)
                model.forward()
                losses = model.compute_val_losses()
                for k, v in losses.items():
                    val_losses_sum[k] = val_losses_sum.get(k, 0.0) + v
                num_val_batches += 1
            
            if num_val_batches > 0:
                print(f'[Validation Epoch {epoch}] ', end='')
                for k, v in val_losses_sum.items():
                    avg_v = v / num_val_batches
                    writer_val.add_scalar(f'Loss/{k}', avg_v, epoch)
                    print(f'{k}: {avg_v:.3f} ', end='')
                print()

        # Update learning rates
        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec')
        for scheduler in schedulers:
            scheduler.step()
        
        lr = model.optimizers[0].param_groups[0]['lr']
        print(f'learning rate = {lr:.7f}')


if __name__ == '__main__':
    main()
