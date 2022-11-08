import logging

import h5py

import data_digest

log = logging.getLogger(__name__)


def run(path):
    log.info(f'Create summary of all recordings in {path}')
    with h5py.File(f'{path}Summary.hdf5', 'w') as h5f:
        data_digest.digest_folder(path, h5f)
