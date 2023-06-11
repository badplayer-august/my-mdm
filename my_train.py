# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
from time import time
import argparse
import numpy as np
from tqdm import tqdm

from models.model import DiT_models
from diffusion import create_diffusion
from data_loaders.get_data import get_dataset_loader


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiT_models[args.model](
        joint_size=args.joint_size, 
        motion_size=args.motion_size,
        encode_type=args.encode_type
    ).to(device)
    diffusion = create_diffusion(timestep_respacing="", learn_sigma=False)  # default: 1000 steps, linear noise schedule


    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    loader = get_dataset_loader(batch_size=64, num_frames=60)

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}...")
        cnt = 0
        for x, conds in tqdm(loader):
            x = x.reshape(x.shape[0], x.shape[1], x.shape[3])
            x = x.to(device)
            y = np.array(conds['y']['text'])
            mask = conds['y']['mask'].to(device)
            mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[3])
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, mask, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{args.results_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    checkpoint = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "args": args
    }
    checkpoint_path = f"{args.results_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    print("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S")
    parser.add_argument("--joint-size", type=int, default=263)
    parser.add_argument("--motion-size", type=int, default=196)
    parser.add_argument("--encode-type", type=str, choices=['clip', 'bert'], default='clip')
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--ckpt-every", type=int, default=5_0000)
    args = parser.parse_args()
    main(args)
