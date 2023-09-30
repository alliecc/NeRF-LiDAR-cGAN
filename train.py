
import configparser
import argparse
import os
from src.trainer import Trainer

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_id',  type=str, required=True)
    parser.add_argument('--name_config',  type=str,  required=True)
    parser.add_argument('--name_data', choices=['clean', 'noisy'], type=str, required=True)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    print(args)

    cfg = configparser.ConfigParser()
    cfg.read(os.path.join('configs', args.name_config))
    cfg = cfg['DEFAULT']

    print(cfg)
    cfg['log_id'] = str(args.log_id)
    cfg['name_data'] = str(args.name_data)

    trainer = Trainer(cfg, eval_only=args.eval_only)
    trainer.run()
