from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnnealing


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER in ['adam_onecycle','adam_cosineanneal']:
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))
        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        betas = optim_cfg.get('BETAS', (0.9, 0.99))
        betas = tuple(betas)
        optimizer_func = partial(optim.Adam, betas=betas)
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    elif optim_cfg.OPTIMIZER in ['adam_onecycle_split']:

        def split_layers_by_names(model, layer_names, frozen_layers=[]):
            def children(m: nn.Module):
                return list(m.children())

            def num_children(m: nn.Module) -> int:
                return len(children(m))
            def check_children(model, layer_names, prefix):
                res_list = []
                remain_list = []
                check_flag = False
                for layer in children(model):
                    if prefix + layer.__class__.__name__ in layer_names:
                        res_list += [layer]
                        check_flag = True
                    else:
                        res, remain, check_flag_tmp = check_children(layer, layer_names, prefix + layer.__class__.__name__ + '.')
                        if check_flag_tmp:
                            res_list += res
                            remain_list += remain
                        else:
                            remain_list += [layer]
                        check_flag = check_flag or check_flag_tmp

                return res_list, remain_list, check_flag
            flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
            # get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
            layers_list = children(model)
            remain_list = []
            res_list = []
            for layer in layers_list:
                layer_name = layer.__class__.__name__
                if layer_name in frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    remain_list += flatten_model(layer)
                elif layer_name in layer_names:
                    current_layer_list = flatten_model(layer)
                    res_list += [nn.Sequential(*current_layer_list)]
                else:
                    # remain_list += flatten_model(layer)
                    res_list_tmp, remain_list_tmp, check_flag = check_children(layer, layer_names, layer_name + '.')
                    res_list_tmp_flatten = []
                    remain_list_tmp_flatten = []
                    for layer_tmp in res_list_tmp:
                        res_list_tmp_flatten += flatten_model(layer_tmp)
                    for layer_tmp in remain_list_tmp:
                        remain_list_tmp_flatten += flatten_model(layer_tmp)
                    if len(res_list_tmp_flatten) > 0:
                        res_list += [nn.Sequential(*res_list_tmp_flatten)]
                    if len(remain_list_tmp_flatten) > 0:
                        remain_list += remain_list_tmp_flatten
            res_list += [nn.Sequential(*remain_list)]
            return res_list
        # flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        # get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        betas = optim_cfg.get('BETAS', (0.9, 0.99))
        betas = tuple(betas)
        # optimizer_func = partial(optim.Adam, betas=betas)
        if optim_cfg.get('OPT_TYPE', 'adam') == 'adam':
            optimizer_func = partial(optim.Adam, betas=betas)
        elif optim_cfg.get('OPT_TYPE', 'adam') == 'AdamW':
            optimizer_func = partial(optim.AdamW, betas=betas)
        else:
            raise NotImplementedError
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, split_layers_by_names(model, optim_cfg.SPLIT_CFG.get('MODEL_NAME', []), optim_cfg.SPLIT_CFG.get('FROZEN_LAYERS', [])), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step: 
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle_split':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START, optim_cfg.SPLIT_CFG.get('SCALE', [])
        )
    elif optim_cfg.OPTIMIZER == 'adam_cosineanneal':
        lr_scheduler = CosineAnnealing(
            optimizer, total_steps, total_epochs, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.PCT_START, optim_cfg.WARMUP_ITER
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
