import torch

def save_checkpoint(model, optimizer, filename="./models/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def __load_checkpoint(checkpoint_file, model, optimizer, lr, DEVICE):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def load_model(
        CHECKPOINT_GENERATOR_B, CHECKPOINT_GENERATOR_A, CHECKPOINT_DISCRIMINATOR_B, CHECKPOINT_DISCRIMINATOR_A, gen_B, gen_A, disc_B, disc_A, opt_gen, opt_disc, LEARNING_RATE, DEVICE, LOAD_MODEL=False
):
    if LOAD_MODEL:
        __load_checkpoint(
            CHECKPOINT_GENERATOR_B,
            gen_B,
            opt_gen,
            LEARNING_RATE,
            DEVICE
        )
        __load_checkpoint(
            CHECKPOINT_GENERATOR_A,
            gen_A,
            opt_gen,
            LEARNING_RATE,
            DEVICE
        )
        __load_checkpoint(
            CHECKPOINT_DISCRIMINATOR_B,
            disc_B,
            opt_disc,
            LEARNING_RATE,
            DEVICE
        )
        __load_checkpoint(
            CHECKPOINT_DISCRIMINATOR_A,
            disc_A,
            opt_disc,
            LEARNING_RATE,
            DEVICE
        )


