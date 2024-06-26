# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Train the DCASE2020 FUSS baseline source separation model."""

import argparse
import os
import sys

cur_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(cur_path)
sys.path.append(os.path.join(parent_path, 'dcase2020_fuss_baseline'))

import tensorflow.compat.v1 as tf

from train import data_io
from train import model
from train import train_with_estimator


def main():
    parser = argparse.ArgumentParser(
        description='Train the DCASE2020 FUSS+DESED baseline separation model.')
    parser.add_argument(
        '-dd', '--data_dir', help='Data directory.',
        required=True)
    parser.add_argument(
        '-md', '--model_dir', help='Directory for checkpoints and summaries.',
        required=True)
    args = parser.parse_args()

    hparams = model.get_model_hparams()
    hparams.sr = 48000.0
    hparams.signal_names = ['DESED_background', 'DESED_foreground',
                            'FUSS_mixture']
    hparams.signal_types = hparams.signal_names

    roomsim_params = {
        'num_sources': len(hparams.signal_names),
        'num_receivers': 1,
        'num_samples': int(hparams.sr * 3.0),
    }
    tf.logging.info('Params: %s', roomsim_params.values())

    feature_spec = data_io.get_roomsim_spec(**roomsim_params)
    inference_spec = data_io.get_inference_spec()

    train_list = os.path.join(
        args.data_dir, 'FUSS_DESED_2_train_mixture_bg_fg_list.txt')
    validation_list = os.path.join(
        args.data_dir, 'FUSS_DESED_2_validation_mixture_bg_fg_list.txt')

    params = {
        'feature_spec': feature_spec,
        'inference_spec': inference_spec,
        'hparams': hparams,
        'io_params': {'parallel_readers': tf.data.experimental.AUTOTUNE,
                      'num_samples': int(hparams.sr * 3.0),
                      'combine_by_class': True,
                      'fixed_classes': ['BG_DSD', 'FG_DSD', 'FUSS_mixture']},
        'input_data_train': train_list,
        'input_data_eval': validation_list,
        'model_dir': args.model_dir,
        'train_batch_size': 3,
        'eval_batch_size': 3,
        'train_steps': 20000000,
        'eval_suffix': 'validation',
        'eval_examples': 800,
        'save_checkpoints_secs': 600,
        'save_summary_steps': 1000,
        'keep_checkpoint_every_n_hours': 4,
        'write_inference_graph': True,
        'randomize_training': True,
    }
    tf.logging.info(params)
    train_with_estimator.execute(model.model_fn, data_io.input_fn, **params)


if __name__ == '__main__':
    main()
