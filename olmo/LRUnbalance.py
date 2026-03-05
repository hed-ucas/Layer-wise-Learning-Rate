import torch
import sys, os

import torch.nn as nn
import math
import pandas as pd
import numpy as np
from operator import itemgetter
import os
import matplotlib.pyplot as plt
import openpyxl
from openpyxl import Workbook
from galore_utils import training_utils

def safe_log10(x, epsilon=1e-10):
    if x <= 0:
        return math.log10(epsilon)
    return math.log10(x)

def record_lrs_to_excel(name, value, excel_path='alpha.xlsx', global_step=0):
    row = [global_step] + value

    if global_step<=10:
        if os.path.exists(excel_path):
            os.remove(excel_path)

        wb = Workbook()
        ws = wb.active

        header = ['global_step'] + name
        ws.append(header)
        ws.append(row)
        wb.save(excel_path)
    else:
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active
        ws.append(row)
        wb.save(excel_path)


class layerTempbalance(object):
    def __init__(self, 
                    global_rank,
                    net, 
                    args, 
                    use_modulewise_lr,
                    alpha_positively_with_lr=True,
                    EVALS_THRESH=0.00001,
                    bins=100, 
                    conv_norm=0.5,
                    pl_fitting='median',
                    xmin_pos=2,
                    filter_zeros=False,
                    remove_first_layer=True,
                    remove_last_layer=True,
                    eigs_thresh=50,
                    esd_metric_for_tb='alpha',
                    assign_func='tb_linear_map',
                    lr_min_ratio=0.5,
                    lr_max_ratio=1.5,
                    batchnorm=True,
                    batchnorm_type='name',
                    layernorm=False,
                    sigmoid_alpha = 4, 
                    swanlab_name=None
                    ):
        """init function
        Args:
            net (nn.module):             net to train
            EVALS_THRESH (float, ):      threshold to filter small eigenvalue. Defaults to 0.00001.
            bins (int, int):             ESD bins. Defaults to 100.
            conv_norm (float, ):         conv norm. Defaults to 0.5.
            pl_fitting (str, ):          powerlaw fitting method. Defaults to median, ['median', 'goodness-of-fit', 'fix-finger']
            xmin_pos (int, ):            set the position of minimum eigenvalue in the tail. Defaults to 2.
            filter_zeros (bool, ):       filter small eigenvalues or not. Defaults to False.
            remove_first_layer (bool, ): whether exclude first layer in TB. Defaults to True.
            remove_last_layer (bool, ): whether exclude last layer in TB. Defaults to True.
            esd_metric_for_tb (str, ): metric for TB scheduling. Defaults to 'alpha'.
            assign_func (str, ):         learning rate assignment function. Defaults to 'tb_linear_map'.
            lr_min_ratio (float, ):      learning rate lower bound. Defaults to 0.5.
            lr_max_ratio (float, ):       learning rate upper bound. Defaults to 1.5.
            batchnorm (bool, ):          whether adjust batch norm learning rate using TB. Defaults to True.
            batchnorm_type (str, ):      how to set learning rate for batchnorm layers
            layernorm (bool, ):          whether adjust layer norm learning rate using TB. Defaults to True.
        """
        self.net = net
        self.args = args
        self.use_modulewise_lr = use_modulewise_lr
        self.alpha_positively_with_lr = alpha_positively_with_lr
        self.EVALS_THRESH = EVALS_THRESH
        self.bins = bins
        self.conv_norm = conv_norm
        self.pl_fitting = pl_fitting
        self.xmin_pos = xmin_pos
        self.filter_zeros = filter_zeros
        self.remove_first_layer = remove_first_layer
        self.remove_last_layer = remove_last_layer
        self.eigs_thresh = eigs_thresh
        self.esd_metric_for_tb = esd_metric_for_tb
        self.assign_func = assign_func
        self.lr_min_ratio = lr_min_ratio
        self.lr_max_ratio = lr_max_ratio
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.bn_to_conv = {}
        self.ln_to_linear = {}
        self.sigmoid_alpha = sigmoid_alpha
        self.swanlab_name = swanlab_name
        self.embed_name = args.optimizer.embed_name
        self.head_name = args.optimizer.head_name
        
        if global_rank==0:
            self.excel_folder = os.path.join(os.getcwd(), 'hot_excel')
            self.excel_file = os.path.join(self.excel_folder, f'{self.swanlab_name}.xlsx')
            if not os.path.exists(self.excel_folder):
                os.makedirs(self.excel_folder)
            if not os.path.exists(self.excel_file):
                columns = ['Step', 'attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
                pd.DataFrame(columns=columns).to_excel(self.excel_file, index=False)
        
        if batchnorm and batchnorm_type == 'name':
            # let the batch norm layer change lr corresponding to the layer
            # with the same layer name 
            longname_lst = []
            for name, m in self.net.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    longname_lst.append(name)
            for name, module in self.net.named_modules():
                if isinstance(module, nn.BatchNorm2d) \
                        and name.replace('bn', 'conv') in longname_lst:
                    self.bn_to_conv[name] = name.replace('bn', 'conv')
                    
        elif batchnorm and batchnorm_type == 'order':
            # let the batch norm layer change lr corresponding to the 
            # conv layer before current layer
            longname_lst = []
            type_lst = []
            for name, module in self.net.named_modules():
                if isinstance(module, nn.Conv2d):
                    longname_lst.append(name)
                    type_lst.append('nn.Conv2d')
                if isinstance(module, nn.BatchNorm2d):
                    if type_lst[-1] == 'nn.Conv2d':
                        self.bn_to_conv[name] = longname_lst[-1]
                    longname_lst.append(name)
                    type_lst.append('nn.BatchNorm2d')
        
        if self.layernorm:
            longname_lst = []
            type_lst = []
            for name, module in self.net.named_modules():
                if isinstance(module, nn.Linear):
                    longname_lst.append(name)
                    type_lst.append('nn.Linear')
                if isinstance(module, nn.LayerNorm):
                    if type_lst[-1] == 'nn.Linear':
                        self.ln_to_linear[name] = longname_lst[-1]
                    longname_lst.append(name)
                    type_lst.append('nn.LayerNorm')
            
        
    def build_optimizer_param_group(self, args, optimizer=None, untuned_lr=0.1, initialize=True, alpha_metric='weight'):
        """build the parameter group for optimizer

        Args:
            untuned_lr (float, ): global learning rate that is not tuned. Defaults to 0.1.
            initialize (bool, ): if True, build a list of dictionary, if False, build a list of learning rate . Defaults to True.

        Returns:
            _type_: _description_
        """
        metrics = self.net_esd_estimator(optimizer, alpha_metric)
        layer_stats = pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})
        layer_stats.loc[layer_stats['eigs_num'] == -1, self.esd_metric_for_tb] = layer_stats[self.esd_metric_for_tb].max()
        layer_stats.loc[layer_stats['eigs_num'] == -1, 'eigs_num'] = layer_stats['eigs_num'].max()

        if self.remove_first_layer:
            layer_stats = layer_stats.drop(labels=0, axis=0)
            layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb])))
        if self.remove_last_layer:
            layer_stats = layer_stats.drop(labels=len(layer_stats) - 1, axis=0)
            layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb])))
        
        layer_stats = layer_stats[layer_stats['eigs_num'] >= self.eigs_thresh]
        layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb])))

        metric_scores = np.array(layer_stats[self.esd_metric_for_tb])
        scheduled_lr = self.get_layer_temps(assign_func=self.assign_func, 
                                            metric_scores=metric_scores, 
                                            untuned_lr=untuned_lr,
                                            layer_stats=layer_stats)
        if args.optimizer.embed_lr > 0:
            scheduled_lr[0] = args.optimizer.embed_lr
            scheduled_lr[-1] = args.optimizer.embed_lr

        layer_stats['scheduled_lr'] = scheduled_lr 

        if args.optimizer.name.lower() == "muon":
            if args.embed_lr > 0:
                layer_stats.loc[layer_stats["longname"].str.contains(self.embed_name, na=False), "scheduled_lr"] = args.optimizer.embed_lr
                layer_stats.loc[layer_stats["longname"].str.contains(self.head_name, na=False), "scheduled_lr"] = args.optimizer.embed_lr
            else:
                layer_stats.loc[layer_stats["longname"].str.contains(self.embed_name, na=False), "scheduled_lr"] = args.optimizer.learning_rate * args.LLR.lr_max_ratio
                layer_stats.loc[layer_stats["longname"].str.contains(self.head_name, na=False), "scheduled_lr"] = args.optimizer.learning_rate * args.LLR.lr_max_ratio

        layer_name_to_tune = list(layer_stats['longname'])
        opt_params_groups = []
        params_to_tune_ids = []
        layer_count = 0
        for name, module in self.net.named_modules():
            if name in layer_name_to_tune: 
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_lr = layer_stats[layer_stats['longname'] == name]['scheduled_lr'].item()
                if initialize:
                    # append a dictionary for initialize optimizer
                    if args.optimizer.name.lower() == "muon" and (self.embed_name in name or self.head_name in name):
                        opt_params_groups.append({'params': module.parameters(), 'initial_lr': args.optimizer.learning_rate})
                    else:
                        opt_params_groups.append({'params': module.parameters(), 'initial_lr': untuned_lr})
                else:
                    # append tuned learning rate 
                    opt_params_groups.append(scheduled_lr)
                layer_count += 1
            elif self.batchnorm \
                and isinstance(module, nn.BatchNorm2d) \
                    and name in self.bn_to_conv \
                        and self.bn_to_conv[name] in layer_name_to_tune:
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_lr = layer_stats[layer_stats['longname'] == self.bn_to_conv[name]]['scheduled_lr'].item()
                if initialize:
                    # append a dictionary for initialize optimizer
                    if args.optimizer.name.lower() == "muon" and (self.embed_name in name or self.head_name in name):
                        opt_params_groups.append({'params': module.parameters(), 'initial_lr': args.optimizer.learning_rate})
                    else:
                        opt_params_groups.append({'params': module.parameters(), 'initial_lr': untuned_lr})
                else:
                    # append tuned learning rate 
                    opt_params_groups.append(scheduled_lr)
                layer_count += 1
            
            elif self.layernorm \
                and isinstance(module, nn.LayerNorm) \
                    and name in self.ln_to_linear \
                        and self.ln_to_linear[name] in layer_name_to_tune:
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_lr = layer_stats[layer_stats['longname'] == self.ln_to_linear[name]]['scheduled_lr'].item()
                if initialize:
                    if args.optimizer.name.lower() == "muon" and (self.embed_name in name or self.head_name in name):
                        opt_params_groups.append({'params': module.parameters(), 'initial_lr': args.optimizer.learning_rate})
                    else:
                        opt_params_groups.append({'params': module.parameters(), 'initial_lr': untuned_lr})
                else:
                    opt_params_groups.append(scheduled_lr)
                layer_count += 1
        
        if initialize:
            untuned_params = \
                filter(lambda p: id(p) not in params_to_tune_ids, self.net.parameters())
            opt_params_groups.append({'params': untuned_params, 'initial_lr': args.optimizer.learning_rate})
            return opt_params_groups, layer_count, layer_stats
        else:
            return opt_params_groups, layer_count, layer_stats

    def calculate_lr_after_steps(self, args, optimizer, layer_count, layer_stats, global_steps, num_steps=10):
        temp_param_groups = []
        for pg in optimizer.param_groups:
            temp_pg = {k: v for k, v in pg.items() if k != 'params'}
            temp_pg['params'] = pg['params'] 
            temp_param_groups.append(temp_pg)
        
        temp_optimizer = type(optimizer)(temp_param_groups, **optimizer.defaults)
        
        temp_scheduler = training_utils.get_scheculer(
            optimizer=temp_optimizer,
            scheduler_type=args.scheduler.name,
            num_training_steps=args.stop_at,
            warmup_steps=args.scheduler.t_warmup,
            min_lr_ratio=args.scheduler.alpha_f,
            last_epoch=global_steps-1)
        
        if global_steps == 1:
            num_linear_steps = num_steps-1
        else:
            num_linear_steps = num_steps

        for _ in range(num_linear_steps):
            temp_scheduler.step()


        
        lr_after_steps = [pg['lr'] for pg in temp_optimizer.param_groups]
        
        param_group_to_layer_name = {}
        
        for pg_idx, param_group in enumerate(optimizer.param_groups[:layer_count]):
            if len(param_group['params']) > 0:
                first_param = next(iter(param_group['params']))
                for name, param in self.net.named_parameters():
                    if param is first_param:
                        module_name = name.rsplit('.', 1)[0]
                        if module_name in self.bn_to_conv:
                            module_name = self.bn_to_conv[module_name]
                        elif module_name in self.ln_to_linear:
                            module_name = self.ln_to_linear[module_name]
                        param_group_to_layer_name[pg_idx] = module_name
                        break
        
        column_name = f'lr_after_{num_steps}_steps'
        if column_name not in layer_stats.columns:
            layer_stats[column_name] = None
        
        for pg_idx, layer_name in param_group_to_layer_name.items():
            if pg_idx < len(lr_after_steps):
                matching_rows = layer_stats[layer_stats['longname'] == layer_name]
                if len(matching_rows) > 0:
                    layer_stats_idx = matching_rows.index[0]
                    layer_stats.loc[layer_stats_idx, column_name] = lr_after_steps[pg_idx]
        
        return layer_stats

    def step(self, args, optimizer, untuned_lr, linear_steps, step_count=None, rank0=True, alpha_metric='weight'):
        opt_params_groups, layer_count, layer_stats = \
            self.build_optimizer_param_group(args=args, optimizer=optimizer, untuned_lr=untuned_lr, initialize=False, alpha_metric=alpha_metric)
        if rank0==True:
            record_lrs_to_excel(list(layer_stats.longname), list(layer_stats.alpha_test), './alpha_test/'+args.swanlab.name+'.xlsx', step_count)
            print(f"Step {step_count}: Layer-wise LR alpha Log")
            # layer_stats.to_excel('./alpha_test/'+args.swanlab_name+'-step-'+str(step_count)+'.xlsx', index=False)
        for index, param_group in enumerate(optimizer.param_groups):
            if index <= layer_count - 1: 
                param_group['initial_lr'] = opt_params_groups[index]
            else:
                param_group['initial_lr'] = args.optimizer.learning_rate        
        
        lr_before_steps = [pg['lr'] for pg in optimizer.param_groups]
        layer_stats = self.calculate_lr_after_steps(args, optimizer, layer_count, layer_stats, step_count, num_steps=linear_steps)
        
        column_name = f'lr_after_{linear_steps}_steps'
        target_lrs = layer_stats[column_name].tolist()
        target_lrs.extend([min(target_lrs)] * (len(optimizer.param_groups) - len(target_lrs)))
        
        lr_diff = [target_lrs[i] - lr_before_steps[i] for i in range(len(target_lrs))]
            
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['initial_lr'] = target_lrs[index]
        
        
        if step_count == 1:
            linear_steps = linear_steps
        else:
            linear_steps = linear_steps+1
        linear_scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type="linear_to_target",
            lr_before_steps=lr_before_steps,
            target_lrs=target_lrs,
            start_step=step_count,
            num_steps=linear_steps,
            last_epoch=step_count-1
        )

        return opt_params_groups, layer_count, linear_scheduler

    def plot_layer_lr(self, optimizer, step_count):
        main_dir = 'unbalanced_figure'
        save_dir = os.path.join(main_dir, self.swanlab_name)
        os.makedirs(save_dir, exist_ok=True)
        
        layer_lrs = []
        layer_names = []

        '''
        print("\nLayer index to module name mapping:")
        print("-----------------------------------")
        for i, param_group in enumerate(optimizer.param_groups):
            layer_lrs.append(param_group['initial_lr'])
            if i == len(optimizer.param_groups) - 1:
                layer_names.append('Untuned')
                print(f"Layer {i}: Untuned parameters")
            else:
                layer_names.append(f'{i+1}')
                # Get module name from parameters
                for param in param_group['params']:
                    for name, param_ref in self.net.named_parameters():
                        if param_ref is param:
                            print(f"Layer {i}: {name}")
                            break
        '''

        for i, param_group in enumerate(optimizer.param_groups):
            layer_lrs.append(param_group['initial_lr'])
            if i == len(optimizer.param_groups) - 1:
                layer_names.append('Untuned')
            else:
                layer_names.append(f'{i+1}')
        
        plt.figure(figsize=(15, 6))
        bars = plt.bar(range(len(layer_lrs)), layer_lrs)

        label_interval = max(2, len(layer_lrs) // 8) 
        for idx, bar in enumerate(bars):
            if idx % label_interval == 0 or idx == len(bars)-1: 
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1e}',
                        ha='center', va='bottom', rotation=45,
                        fontsize=8) 

        correlation_type = "Positive" if self.alpha_positively_with_lr else "Negative"
        plt.title(f'Layer-wise Weight Decay Distribution\n({correlation_type} Correlation)',
                pad=20, fontsize=14)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Weight Decay', fontsize=12)
        
        total_layers = len(layer_lrs)
        if total_layers > 20:
            show_interval = max(2, total_layers // 8)
            xticks_pos = list(range(0, total_layers-show_interval, show_interval))
            if total_layers - 1 - xticks_pos[-1] >= show_interval//2:
                xticks_pos.append(total_layers - 1)
            
            plt.xticks(xticks_pos, 
                    [layer_names[i] for i in xticks_pos],
                    rotation=45, 
                    ha='right') 
            plt.xticks(range(len(layer_lrs)), 
                    layer_names,
                    rotation=45, 
                    ha='right') 

        plt.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        plt.subplots_adjust(bottom=0.2)

        correlation_type = "pos" if self.alpha_positively_with_lr else "neg"
        save_path = os.path.join(save_dir, f'lr_distribution_{correlation_type}_step_{step_count}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
                
    def net_esd_estimator(
            self,
            optimizer=None,
            alpha_metric='weight', 
            verbose=False):
        """evaluate the ESD of the conv nets
        Args:
            verbose: 
        Returns:
            _type_: _description_
        """

        if verbose:
            print("=================================")
            print(f"pl_fitting: {self.pl_fitting}, xmin_pos: {self.xmin_pos}, conv_norm: {self.conv_norm}, filter_zeros: {self.filter_zeros}")
            print("=================================")
        # iterate through layers

        results = {
        'gradnorm': [],
        'gradnorm_d_weightnorm': [],
        'fnorm':[],
        'spectral_norm': [],
        'entropy': [],
        'stable_rank': [],
        'alphahat':[],
        'alpha':[],
        'alpha_test':[],
        'longname':[],
        'eigs':[],
        'eigs_num':[]
        }
        for name, m in self.net.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding): 
                
                # matrix = m.weight.data.clone().to(torch.float)
                if alpha_metric == 'weight' or m.weight.grad is None:
                    matrix = m.weight.data.clone().to(torch.float)
                elif alpha_metric == 'grad':
                    matrix = m.weight.grad.data.clone().to(torch.float)
                elif alpha_metric == 'momentum':
                    for _, param in m.named_parameters(prefix=name):
                        if param.requires_grad and param in optimizer.state:
                            state = optimizer.state[param]
                            if 'momentum_buffer' in state:
                                matrix = state['momentum_buffer'].clone().to(torch.float)   
                                break
                            elif 'exp_avg' in state: 
                                matrix = state['exp_avg'].clone().to(torch.float) 
                                break
                            else:
                                raise NotImplementedError  

                if isinstance(m, nn.Conv2d):
                    matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(self.conv_norm)
                    matrix = matrix.transpose(1, 2).transpose(0, 1)
                eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
                eigs, _ = torch.sort(eigs, descending=False) 
                spectral_norm = eigs[-1].item() 
                fnorm = torch.sum(eigs).item() 
                stable_rank = fnorm / (spectral_norm + 1e-8)
                entropy = self.matrix_entropy(torch.sqrt(eigs))

                if m.weight.grad is not None:
                    grad_norm = m.weight.grad.data.norm(2).item()
                    gradnorm_div_weightnorm = grad_norm / (spectral_norm + 1e-8)
                else:
                    grad_norm = 10 
                    gradnorm_div_weightnorm = 10

                if self.filter_zeros:
                    nz_eigs = eigs[eigs > self.EVALS_THRESH]
                    N = len(nz_eigs)
                    if N == 0:
                        nz_eigs = eigs
                        N = len(nz_eigs)
                else:
                    nz_eigs = eigs
                    N = len(nz_eigs)

                log_nz_eigs  = torch.log(nz_eigs) 

                if self.pl_fitting == 'median': #
                    i = int(len(nz_eigs) / self.xmin_pos) 
                    xmin = nz_eigs[i] 
                    n = float(N - i) 
                    seq = torch.arange(n).cuda()
                    final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i]) 
                    final_D = torch.max(torch.abs(
                                1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                            )) 
                else:
                    alphas = torch.zeros(N-1)
                    Ds     = torch.ones(N-1)
                    if self.pl_fitting == 'fix-finger':
                        hist_nz_eigs = torch.log10(nz_eigs) 
                        min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                        counts = torch.histc(hist_nz_eigs, self.bins, min=min_e, max=max_e)
                        boundaries = torch.linspace(min_e, max_e, self.bins + 1) 
                        h = counts, boundaries
                        ih = torch.argmax(h[0])
                        xmin2 = 10 ** h[1][ih] 
                        xmin_min = torch.log10(0.95 * xmin2) 
                        xmin_max = 1.5 * xmin2 
                    
                    for i, xmin in enumerate(nz_eigs[:-1]): 
                        if self.pl_fitting == 'fix-finger': 
                            if xmin < xmin_min:
                                continue
                            if xmin > xmin_max:
                                break

                        n = float(N - i) 
                        seq = torch.arange(n).cuda() 
                        alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                        alphas[i] = alpha 
                        if alpha > 1: 
                            Ds[i] = torch.max(torch.abs(
                                1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                            ))

                    min_D_index = torch.argmin(Ds) 
                    final_alpha = alphas[min_D_index]
                    final_D = Ds[min_D_index]
                
                final_alpha = final_alpha.item()
                final_D = final_D.item()
                final_alphahat=final_alpha*safe_log10(spectral_norm) 
                final_alphahat=math.log(1.0 + math.exp(final_alphahat))

                results['alpha_test'].append(final_alpha)
                results['longname'].append(name) 
                
                if self.embed_name in name or self.head_name in name: 
                    results['eigs_num'].append(-1)
                    results['alpha'].append(-1)
                    results['alphahat'].append(-1)

                    results['gradnorm'].append(-1)
                    results['gradnorm_d_weightnorm'].append(-1)
                    results['fnorm'].append(-1)
                    results['spectral_norm'].append(-1)
                    results['entropy'].append(-1)
                    results['stable_rank'].append(-1)
                    results['eigs'].append(-1)
                else:
                    results['eigs_num'].append(len(eigs))
                    results['alpha'].append(final_alpha)
                    results['alphahat'].append(final_alphahat)
                
                    results['gradnorm'].append(grad_norm)
                    results['gradnorm_d_weightnorm'].append(gradnorm_div_weightnorm)
                    results['fnorm'].append(fnorm)
                    results['spectral_norm'].append(spectral_norm)
                    results['entropy'].append(entropy.detach().cpu().item())
                    results['stable_rank'].append(stable_rank)
                    results['eigs'].append(eigs.detach().cpu().numpy())
        return results
    
    def matrix_entropy(self, svals):
        EPSILON = 6e-05

        rank = torch.count_nonzero(svals > EPSILON) 
        evals = svals*svals 
        p = evals / torch.sum(evals) + EPSILON 
        entropy = -torch.sum(p * torch.log(p)) / torch.log(torch.tensor(rank.detach().cpu().numpy() + EPSILON, dtype=torch.float))
        return entropy

    def get_layer_temps(self, assign_func, metric_scores, untuned_lr, layer_stats):
        n = len(metric_scores) 
        idx = [i for i in range(n)]
        temps = np.array([untuned_lr] * n) 

        # Get the layer names from layer_stats
        layer_metrics = {}
        for idx, name in enumerate(layer_stats['longname']):
            try:
                layer_name = name.split('.')[2]  # Gets 'layer1' from 'module.layer1.0.conv1'
                if layer_name not in layer_metrics:
                    layer_metrics[layer_name] = []
                layer_metrics[layer_name].append(metric_scores[idx])  # Use index to get corresponding score
            except:
                layer_name = -1  # Gets 'layer1' from 'module.layer1.0.conv1'
                if layer_name not in layer_metrics:
                    layer_metrics[layer_name] = []
                layer_metrics[layer_name].append(metric_scores[idx])  # Use index to get corresponding score


        if self.alpha_positively_with_lr: 
            if assign_func == 'tb_linear_map':
                lr_range = [self.lr_min_ratio * untuned_lr,  self.lr_max_ratio * untuned_lr]
                score_range = [min(metric_scores),  max(metric_scores)]
                temps = np.interp(metric_scores, score_range, lr_range) 
            elif assign_func == 'tb_sqrt':
                temps = np.sqrt(metric_scores)/np.sum(np.sqrt(metric_scores)+1e-8) * n * untuned_lr
            elif assign_func == 'tb_log2':
                temps = np.log2(metric_scores)/np.sum(np.log2(metric_scores)+1e-8) * n * untuned_lr
            elif assign_func == 'sigmoid':
                mean = np.mean(metric_scores)
                std = np.std(metric_scores) + 1e-8
                norm = (metric_scores - mean) / std
                theta = 2 / (1 + np.exp(-self.sigmoid_alpha * norm))
                temps = theta * untuned_lr
            elif assign_func == 'layer_linear_map':
                # Process each layer separately
                temps = np.zeros_like(metric_scores)
                for layer_name, layer_scores in layer_metrics.items():
                    layer_scores = np.array(layer_scores)
                    lr_range = [self.lr_min_ratio * untuned_lr,  self.lr_max_ratio * untuned_lr]
                    score_range = [min(layer_scores),  max(layer_scores)]
                    layertemps = np.interp(layer_scores, score_range, lr_range) 

                    layer_indices = [i for i, name in enumerate(layer_stats['longname']) 
                                    if name.split('.')[2] == layer_name]
                    for idx, temptheta in zip(layer_indices, layertemps):
                        temps[idx] = temptheta
            elif assign_func == 'layer_sqrt':
                # Process each layer separately
                temps = np.zeros_like(metric_scores)
                for layer_name, layer_scores in layer_metrics.items():
                    layer_scores = np.array(layer_scores)
                    n = len(layer_scores)
                    layertemps = np.sqrt(layer_scores)/np.sum(np.sqrt(layer_scores)+1e-8) * n * untuned_lr
                    layer_indices = [i for i, name in enumerate(layer_stats['longname']) 
                                    if name.split('.')[2] == layer_name]
                    for idx, temptheta in zip(layer_indices, layertemps):
                        temps[idx] = temptheta
            elif assign_func == 'layer_log2':
                # Process each layer separately
                temps = np.zeros_like(metric_scores)
                for layer_name, layer_scores in layer_metrics.items():
                    layer_scores = np.array(layer_scores)
                    n = len(layer_scores)
                    layertemps = np.log2(layer_scores)/np.sum(np.log2(layer_scores)+1e-8) * n * untuned_lr
                    layer_indices = [i for i, name in enumerate(layer_stats['longname']) 
                                    if name.split('.')[2] == layer_name]
                    for idx, temptheta in zip(layer_indices, layertemps):
                        temps[idx] = temptheta
            elif assign_func == 'layerwise_sigmoid':
                # Process each layer separately
                temps = np.zeros_like(metric_scores)
                for layer_name, layer_scores in layer_metrics.items():
                    layer_scores = np.array(layer_scores)
                    # Compute layer-specific mean and std
                    layer_mean = np.mean(layer_scores)
                    layer_std = np.std(layer_scores) + 1e-8
                    # Normalize within layer
                    layer_norm = (layer_scores - layer_mean) / layer_std
                    # Apply sigmoid
                    layer_theta = 2 / (1 + np.exp(-self.sigmoid_alpha * layer_norm))
                    # Assign back to original positions
                    layer_indices = [i for i, name in enumerate(layer_stats['longname']) 
                                    if name.split('.')[2] == layer_name]
                    for idx, theta in zip(layer_indices, layer_theta):
                        temps[idx] = theta * untuned_lr
            else:
                raise NotImplementedError
        else: 
            if assign_func == 'tb_linear_map':
                lr_range = [self.lr_max_ratio * untuned_lr, self.lr_min_ratio * untuned_lr] 
                score_range = [min(metric_scores),  max(metric_scores)] 
                temps = np.interp(metric_scores, score_range, lr_range) 
            elif assign_func == 'tb_sqrt':
                temps = np.sum(np.sqrt(metric_scores))/(np.sqrt(metric_scores)*n) * untuned_lr
            elif assign_func == 'tb_log2': 
                temps = (np.sum(np.log2(metric_scores)) * n)/np.log2(metric_scores) * untuned_lr
            elif assign_func == 'tb_step':
                idxes = np.argsort(-metric_scores) 
                unsort_temps = [untuned_lr * (self.lr_min_ratio + (self.lr_max_ratio - self.lr_min_ratio) * i / n) for i in range(n)] 
                temps = [value for _, value in sorted(list(zip(idxes, unsort_temps)), key=itemgetter(0))] 
            else:
                raise NotImplementedError
        return temps