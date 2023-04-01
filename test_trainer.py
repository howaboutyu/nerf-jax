import pytest
import yaml


from nerf_config import get_config

from trainer import train_and_evaluate 


def test_get_config():
    config_path = 'configs/lego.yaml'

    with open(config_path) as file:
        config = yaml.safe_load(file)

    config = get_config(config_path)

    assert config.dataset_name == 'lego'


def test_train_and_evaluate():
    config_path = 'configs/lego.yaml'

    nerf_config = get_config(config_path)

    nerf_config.epochs = 1
    nerf_config.log_every = 1   
    nerf_config.batch_size = 8 
    nerf_config.num_devices = 1
    nerf_config.max_steps = 4


    nerf_config.ckpt_dir = '/tmp/test_train_and_evaluate'

    train_and_evaluate(nerf_config)

    assert True
