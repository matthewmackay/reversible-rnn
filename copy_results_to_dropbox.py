# File: copy_results_to_dropbox.py
#
# Usage: python copy_results_to_dropbox.py --base_dir=.
#
# Optionally set up a cron job as follows, to copy results to Dropbox every 5 minutes:
#        */5 * * * * /home/pvicol/anaconda3/bin/python /home/pvicol/rev-rnn-public/copy_results_to_dropbox.py --base_dir=/home/pvicol/rev-rnn-public/nmt >> /home/pvicol/cronjob.log 2>&1

import os
import ipdb
import shutil
import argparse
from fnmatch import fnmatch, filter


def copydir(source, dest, patterns):
    """Copy a directory structure overwriting existing files"""
    for root, dirs, files in os.walk(source):
        if not os.path.isdir(root):
            os.makedirs(root)

        keep_files = [name for name in files if any([name.endswith(pattern) for pattern in patterns])]  # THIS GOES WITH ".txt"

        for file in keep_files:
            rel_path = root.replace(source, '').lstrip(os.sep)
            dest_path = os.path.join(dest, rel_path)

            if not os.path.isdir(dest_path):
                os.makedirs(dest_path)

            shutil.copyfile(os.path.join(root, file), os.path.join(dest_path, file))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility to Copy Experiment Results to Dropbox')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base directory from which to recursively copy files to Dropbox (keeping directory structure)')
    args = parser.parse_args()

    DROPBOX_BASE_DIR = '/home/pvicol/Dropbox'
    FROM_BASE_DIR = args.base_dir
    TO_BASE_DIR = os.path.join(DROPBOX_BASE_DIR, os.path.basename(FROM_BASE_DIR))
    file_types_to_copy = ['args.yaml', 'iteration_log_loss', 'log_perp', 'mem_log', 'result.txt',
                          'test_pred.txt', 'test_stdout.txt', 'val_pred.txt', 'val_stdout.txt']
    copydir(FROM_BASE_DIR, TO_BASE_DIR, patterns=file_types_to_copy)
