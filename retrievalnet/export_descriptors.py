import numpy as np
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
from retrievalnet.models import get_model  # noqa: E402
from retrievalnet.datasets import get_dataset  # noqa: E402
from retrievalnet.settings import EXPER_PATH, DATA_PATH  # noqa: E402


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    args = parser.parse_args()

    export_name = args.export_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    seqs = config['data']['test_seq']

    if not isinstance(seqs, list):
        seqs = [seqs]

    if Path(EXPER_PATH, export_name).exists():
        checkpoint_path = Path(EXPER_PATH, export_name)
        if 'weights' in config:
            checkpoint_path = Path(checkpoint_path, config['weights'])
    else:
        checkpoint_path = Path(DATA_PATH, 'weights', config['weights'])

    with get_model(config['model']['name'])(
            data_shape={'image': [None, None, None, config['model']['image_channels']]},
            **config['model']) as net:
        net.load(str(checkpoint_path))

        for seq in tqdm(seqs):
            output_dir = Path(EXPER_PATH, 'outputs/{}/{}/'.format(export_name, seq))
            output_dir.mkdir(parents=True, exist_ok=True)

            config['data']['test_seq'] = seq
            dataset = get_dataset(config['data']['name'])(**config['data'])
            test_set = dataset.get_test_set()

            pbar = tqdm()
            while True:
                try:
                    data = next(test_set)
                except dataset.end_set:
                    break
                descriptor = net.predict(data, keys='descriptor')
                filepath = Path(output_dir, '{}.npz'.format(
                        data['name'].decode('utf-8')))
                np.savez_compressed(filepath, descriptor=descriptor)
                pbar.update(1)
            pbar.close()
