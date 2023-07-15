import wandb
import numpy as np
hyper_config={
    'strategy':{
        'values':[5,6],
    },
    'learning_rate':{
        'distribution':'log_uniform_values',
        'min':1e-4,
        'max':0.5,
    },
    'layer_num':{
        #q:from 2 to 64 in log distribution
        'distribution':'q_log_uniform_values',
        'q':2,
        'min':4,
        'max':64,
    },
    'depth':{
        #q:from 2 to 64 in log distribution
        'distribution':'q_log_uniform_values',
        'q':4,
        'min':4,
        'max':32,
    },  
}
def main(config=None):
    with wandb.init(
        project = 'distillation',
        name = 'banana_milk',
        config = config,
    ):
        pass

def wb_option(hyper_config, sweep_name, project_name):
    sweep_config={
        'method':'random',
        'metric':{
            #metric：目标是最大化acc。
            'name':'val-acc',
            'goal':'maximize',
        },
    }
    sweep_config['parameters'] = hyper_config
    sweep_config['name'] = sweep_name
    sweeper = wandb.sweep(sweep_config, project = project_name)
    return sweeper

if __name__=='__main__':
    print(callable(main))