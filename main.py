import tensorflow as tf
import jax
from absl import app, flags

from datasets import dataset_factory
from nerf_config import get_config, NerfConfig
from trainer import train_and_evaluate
from renderer import render_nerf

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", None, "config file path")
flags.DEFINE_string("mode", "train_and_eval", "can be train_and_eval or render")
# for render
flags.DEFINE_string("render_output_folder", None, "where to output the rendered images")
flags.DEFINE_bool(
    "display_render", False, "whether to show the rendered images using cv2.imshow"
)

flags.mark_flag_as_required("config_path")


def main(argv):
    # Remove GPUs for tf
    tf.config.experimental.set_visible_devices([], "GPU")

    # Load config from yaml file
    config = get_config(FLAGS.config_path)

    if FLAGS.mode == "train_and_eval":
        # Train and evaluate
        train_and_evaluate(config)
    else:
        # Render NeRF
        render_nerf(
            config,
            render_output_folder=FLAGS.render_output_folder,
            display_render=FLAGS.display_render,
        )


if __name__ == "__main__":
    app.run(main)
