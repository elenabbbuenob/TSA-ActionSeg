import os
import yaml
from easydict import EasyDict

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def create_config(config_file_exp):
    root_dir = "output/"

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Output path for the dataset
    base_dir = os.path.join(root_dir, cfg['db_name'])
    mkdir_if_missing(base_dir)

    cfg['base_dir'] = base_dir

    return cfg 
