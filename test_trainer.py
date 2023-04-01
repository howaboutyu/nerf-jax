import pytest
import yaml
import os




from nerf_config import get_config

from trainer import train_and_evaluate 


def test_get_config():
    config_path = 'configs/lego.yaml'

    with open(config_path) as file:
        config = yaml.safe_load(file)

    config = get_config(config_path)

    assert config.dataset_name == 'lego'


def test_train_and_evaluate():
    """Test that the training loop runs without errors"""

    os.system('make get_nerf_example_data')

    config_path = 'configs/lego.yaml'

    nerf_config = get_config(config_path)

    nerf_config.scale = 0.02
    nerf_config.epochs = 1
    nerf_config.log_every = 1   
    nerf_config.batch_size = 8 
    nerf_config.num_devices = 1
    nerf_config.steps_per_eval = 1
    nerf_config.max_steps = 4
    nerf_config.ckpt_dir = '/tmp/test_train_and_evaluate'

    train_and_evaluate(nerf_config)

    assert True
