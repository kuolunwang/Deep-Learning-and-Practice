#!/usr/bin/env python3

def teacher_forcing_ratio(epoch, epochs):
    """
    epochs : totoal epochs
    epochs : current epoch
    """

    # from 1.0 to 0.0
    teacher_forcing_ratio = 1. - (1. / (epochs - 1)) * (epoch - 1) 
    return teacher_forcing_ratio

def get_kld_weight(kl_cost_type, epochs, epoch, threshold):
    """
    kl_cost_type : 'monotonic' or 'cycle'
    epochs : totoal epochs
    epochs : current epoch
    threshold :
        montonic -> threshold for 0.0 to 1.0
        cycle -> each threshold repeat
    """

    if(kl_cost_type == 'monotonic'):
        return (0.25 / (threshold - 1)) * (epoch - 1) if epoch < threshold else 0.25

    elif(kl_cost_type == 'cycle'):
        epoch %= threshold 
        return (0.25 / (threshold - 1)) * (epoch - 1) if epoch is not 0 else 0.25

    else:
        raise("kl_cost_type not exist")