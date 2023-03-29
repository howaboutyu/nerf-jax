import yaml
from dataclasses import dataclass


@dataclass
class Config:

    dataset_name: str
    dataset_path: str
    dataset_type: str


    scale: float = 1.0
    near: float = 2.0
    far: float = 6.0
    L_position: int = 10 
    L_direction: int = 4 
    use_hvs: bool = True
    num_samples_coarse: int = 64
    num_samples_fine: int = 128 

    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 0.0001
    optimizer: str = 'adam'

    ckpt_dir: str = 'checkpoints'



def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_config(config_file):
    config = load_config(config_file)
    return Config(**config)

