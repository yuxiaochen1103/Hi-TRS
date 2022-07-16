from torch.optim.lr_scheduler import LambdaLR
def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):

    lr_init = optimizer.defaults["lr"]
    if not (lr_init >= lr_end):
         raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            if lr_init > lr_end:
                lr_range = lr_init - lr_end
                decay_steps = num_training_steps - num_warmup_steps
                pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
                decay = lr_range * pct_remaining ** power + lr_end
                return decay / lr_init  # as LambdaLR multiplies by lr_init
            else:
                return lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)