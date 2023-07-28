"""
"""
import sys
import argparse

import PyPDF2


def main(argv):

    args = parse(argv)

    if args.start == args.end:
        args.end += 1

    with open(args.file_in, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        pages = reader.pages[args.start:args.end]
        writer = PyPDF2.PdfWriter()
        for page in pages:
            writer.add_page(page)
        with open(args.file_out, 'wb') as output_file:
            writer.write(output_file)


def parse(argv):
    parser = argparse.ArgumentParser(argv)
    parser.add_argument('--file_in', required=True, type=str)
    parser.add_argument('--file_out', required=True, type=str)
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--end', required=True, type=int)
    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    main(sys.argv)
