
import tensorflow as tf
import jax
from absl import app, flags

from datasets import dataset_factory
from nerf_config import get_config, NerfConfig
from trainer import train_and_evaluate

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", None, "config file path")
flags.mark_flag_as_required("config_path")


def main(argv):

    # Remove GPUs for tf 
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Load config from yaml file
    config = get_config(FLAGS.config_path)

    # Train and evaluate
    train_and_evaluate(config)


if __name__ == '__main__':
    app.run(main)
