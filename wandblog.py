import wandb
import json

with open("./wandb_config.json",'r') as w:
    cfg = json.load(w)

print(cfg)
class Logger:
    def __init__(self,config):
        if cfg['use_wandb']:
            wandb.init(project=cfg['project_name'],entity=cfg['entity'],name=cfg['name'], config=config )
            wandb.define_metric('train_step')
            wandb.define_metric('train_epoch')
            wandb.define_metric('val_epoch')
            wandb.define_metric('val_step')
            
            wandb.define_metric('train/loss',step_metric = 'train_step')
            wandb.define_metric('train/acc',step_metric = 'train_epoch')
            wandb.define_metric('val/*', step_metric = 'val_epoch')


    def log(self, args):
        if cfg['use_wandb']:
            wandb.log(**args)
            
        else:
            pass