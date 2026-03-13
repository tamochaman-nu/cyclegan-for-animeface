import optuna
import time
import os
import torch
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from options import get_train_options
from datasets import UnalignedDataset
from models import CycleGANModel
from utils import ImagePool, get_scheduler

def objective(trial, base_opt):
    opt = copy.deepcopy(base_opt)
    
    # Suggest hyperparameters
    opt.lambda_A = trial.suggest_float('lambda_A', 1.0, 20.0, log=True)
    opt.lambda_B = trial.suggest_float('lambda_B', 1.0, 20.0, log=True)
    opt.lambda_identity = trial.suggest_float('lambda_identity', 0.1, 1.0)
    opt.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    opt.lambda_perceptual = trial.suggest_float('lambda_perceptual', 0.01, 1.0, log=True)
    opt.n_blocks_g = trial.suggest_int('n_blocks_g', 3, 12, step=3)
    
    # Override settings for fast search
    opt.n_epochs = opt.n_epochs_optuna
    opt.n_epochs_decay = 0
    
    # Create a unique name for this trial to save checkpoints and tensorboard logs
    trial_name = f"{opt.name}_trial_{trial.number}"
    opt.name = trial_name
    
    # Set tensorboard dir
    tb_dir = os.path.join(opt.tensorboard_dir, trial_name)
    writer_train = SummaryWriter(log_dir=os.path.join(tb_dir, 'train'))
    
    print(f"--- Starting Trial {trial.number} ---")
    print(f"Hyperparameters: lambda_A={opt.lambda_A:.3f}, lambda_B={opt.lambda_B:.3f}, lambda_identity={opt.lambda_identity:.3f}, lambda_perceptual={opt.lambda_perceptual:.3f}, lr={opt.lr:.6f}, n_blocks_g={opt.n_blocks_g}")
    
    # Dataset
    dataset = UnalignedDataset(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads),
        drop_last=False
    )
    
    # Validation Dataset
    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'val'
    has_val = False
    try:
        val_dataset = UnalignedDataset(val_opt)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads),
            drop_last=False
        )
        has_val = True
        writer_val = SummaryWriter(log_dir=os.path.join(tb_dir, 'val'))
    except Exception as e:
        print(f'Could not load validation dataset for optuna: {e}')

    # Model definition
    model = CycleGANModel(opt)
    schedulers = [get_scheduler(optimizer, opt) for optimizer in model.optimizers]

    fake_A_pool = ImagePool(opt.pool_size)
    fake_B_pool = ImagePool(opt.pool_size)

    best_val_loss = float('inf')
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_losses_sum = {}
        num_train_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Trial {trial.number} Epoch {epoch}/{opt.n_epochs}")
        for i, data in enumerate(pbar):
            model.set_input(data)
            model.forward()
            
            # G update
            for netD in [model.netD_A, model.netD_B]:
                for param in netD.parameters():
                    param.requires_grad = False
            
            model.optimizer_G.zero_grad()
            model.backward_G()
            model.optimizer_G.step()
            
            # D update
            for netD in [model.netD_A, model.netD_B]:
                for param in netD.parameters():
                    param.requires_grad = True

            fake_B = fake_B_pool.query(model.fake_B)
            model.fake_B = fake_B
            model.optimizer_D.zero_grad()
            model.backward_D_A()
            
            fake_A = fake_A_pool.query(model.fake_A)
            model.fake_A = fake_A
            model.backward_D_B()
            
            model.optimizer_D.step()

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
                losses['VGG_A'] = model.loss_perceptual_A.item()
                losses['VGG_B'] = model.loss_perceptual_B.item()
            if opt.lambda_arcface > 0.0:
                losses['Arc_A'] = model.loss_arcface_A.item()
                losses['Arc_B'] = model.loss_arcface_B.item()
                
            for k, v in losses.items():
                epoch_losses_sum[k] = epoch_losses_sum.get(k, 0.0) + v
            num_train_batches += 1
            
            if i % 10 == 0:
                pbar.set_postfix({
                    'G_A': f"{losses['G_A']:.2f}",
                    'D_A': f"{losses['D_A']:.2f}",
                    'Cyc_A': f"{losses['Cyc_A']:.2f}"
                })
            
        if num_train_batches > 0:
            for k, v in epoch_losses_sum.items():
                avg_v = v / num_train_batches
                writer_train.add_scalar(f'Loss/{k}', avg_v, epoch)
                
        # Validation loop
        current_val_loss = 0.0
        if has_val:
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
                for k, v in val_losses_sum.items():
                    avg_v = v / num_val_batches
                    writer_val.add_scalar(f'Loss/{k}', avg_v, epoch)
                    
                # Calculate metric to minimize: e.g. Cyc_A + Cyc_B + G_A + G_B
                current_val_loss = (val_losses_sum['Cyc_A'] + val_losses_sum['Cyc_B'] + 
                                    val_losses_sum['G_A'] + val_losses_sum['G_B']) / num_val_batches
                
                # Report intermediate value to Optuna for pruning
                trial.report(current_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        for scheduler in schedulers:
            scheduler.step()

        if has_val and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            
    # Depending on what we want to optimize, return best val loss, or the last loss if no val.
    if has_val:
        return best_val_loss
    else:
        # fallback if no valset
        final_train_loss = (epoch_losses_sum['Cyc_A'] + epoch_losses_sum['Cyc_B'] + 
                            epoch_losses_sum['G_A'] + epoch_losses_sum['G_B']) / num_train_batches
        return final_train_loss


def main():
    opt = get_train_options()
    
    # Setup Optuna study with sqlite storage
    db_url = "sqlite:///optuna.db"
    study_name = opt.name
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=db_url, 
        load_if_exists=True,
        direction="minimize"
    )
    
    study.optimize(lambda trial: objective(trial, opt), n_trials=opt.optuna_trials)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == '__main__':
    main()
