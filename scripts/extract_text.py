""" Simple script for extracting text from a PDF """
# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Cameron Fabbri
import fitz
import click

import src.pdf_utils as pu


@click.command()
@click.option(
    '--file_in', '-fi', default='data/MSRB-Rule-Book-Current-Version.pdf')
@click.option('--file_out', '-fo', default='data/MSRB-Text.txt')
@click.option('--start_page', '-s', default=16)
@click.option('--end_page', '-e', default=-1)
@click.option('--delimiter', '-d', default='\n\n')
def main(file_in, file_out, start_page, end_page, delimiter):
    """ """

    # If only extracting one page, still have to give a range
    if start_page == end_page:
        end_page += 1

    sections = pu.extract_text_by_paragraphs(
        file_in, delimiter, start_page, end_page)

    with open(file_out, 'w') as f:
        for section in sections:
            section = ' '.join(section.strip().split('\n'))
            f.write(f'{section}\n')

    print(f'Wrote {len(sections)} sections to {file_out}')


if __name__ == '__main__':
    main()

