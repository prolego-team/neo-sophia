""" Script to download Volumes 1-10 of Title 12 """
import os

import click
import requests

from tqdm import tqdm

import examples.project as project

opj = os.path.join


@click.command()
@click.option(
    '--overwrite', '-o', is_flag=True, show_default=True,
    help='Overwrite files if they are already downloaded')
def main(overwrite):
    """ """

    save_dir = opj(project.DATASETS_DIR_PATH, 'Title-12')
    os.makedirs(save_dir, exist_ok=True)
    print(f'\nDownloading Title 12 Volumes to {save_dir} ...\n')

    base_url = 'https://www.govinfo.gov/content/pkg/CFR-2022-title12-vol[VOL]/pdf/CFR-2022-title12-vol[VOL].pdf'

    for volume in range(1, 11):
        url = base_url.replace('[VOL]', str(volume))
        filename = f'Title-12-Volume-{volume}.pdf'
        filepath = opj(save_dir, filename)

        if not os.path.exists(filepath) or overwrite:
            print(f'Downloading {url}')
            req = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(req.content)
                print(f'Saved {filename}')
        else:
            print('Skipping', url)


if __name__ == '__main__':
    main()

