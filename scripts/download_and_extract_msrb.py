""" Simple script for extracting text from a PDF """
# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Cameron Fabbri
import os

import fitz
import click

import neosophia.llmtools.pdf_utils as pu

opj = os.path.join

@click.command()
@click.option('--start_page', '-s', default=16)
@click.option('--end_page', '-e', default=-1)
@click.option('--data_dir', '-d', default='data/')
@click.option('--delimiter', default=r"^\([a-zA-z0-9]\)")

def main(start_page, end_page, data_dir, delimiter):
    """ """

    url = 'https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf'
    filename = 'MSRB-Rule-Book-Current-Version.pdf'

    file_in = opj(data_dir, filename)

    if not os.path.exists(file_in):
        print('Downloading MSRB to data/ ...')
        os.system(f'wget -P {data_dir} {url}')
        print('Done')

    # If only extracting one page, still have to give a range
    if start_page == end_page:
        end_page += 1

    sections, num_pages = pu.extract_text_by_paragraphs(
        file_in, delimiter, start_page, end_page)

    if end_page == -1:
        end_page = num_pages

    file_out = opj(data_dir, f'MSRB-Text-{start_page}-{end_page}.txt')

    with open(file_out, 'w') as f:
        for section in sections:
            section = ' '.join(section.strip().split('\n'))
            f.write(f'{section}\n')

    print(f'Wrote {len(sections)} sections to {file_out}')


if __name__ == '__main__':
    main()

