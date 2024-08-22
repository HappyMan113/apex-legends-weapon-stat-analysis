import glob
import os
import sys
from multiprocessing import Pool, cpu_count
from typing import Iterable

import ffmpeg

ANNOTATED_SUFFIX = '.annotated'


def main():
    input_args = sys.argv[1:]
    default_extension = '.mp4'
    if len(input_args) == 0:
        print(f'Must specify a filenames and/or directories of {default_extension} files to '
              'annotate.', file=sys.stderr)
        sys.exit(1)

    input_filenames_iters: list[Iterable[str]] = []
    for input_arg in input_args:
        input_arg = input_arg.strip(' "')
        if len(input_arg) == 0:
            continue

        if not os.path.exists(input_arg):
            print(f'Path {input_arg} does not exist. Skipping.', file=sys.stderr)

        if os.path.isdir(input_arg):
            input_filenames = (
                filename
                for filename in glob.iglob(os.path.join(input_arg, f'*{default_extension}'))
                if not _is_annotated_filename(filename))
        else:
            input_filenames = input_args
        input_filenames_iters.append(input_filenames)

    max_num_threads = max(1, cpu_count() // 8)
    overwrite = False
    with Pool(max_num_threads) as pool:
        pool.starmap(annotate_frame_and_time,
                     ((input_filename, overwrite)
                      for input_filenames in input_filenames_iters
                      for input_filename in input_filenames))

    print(f'Done')


def _to_annotated_filename(filename: str):
    if not isinstance(filename, str):
        raise ValueError(f'filename {filename} is not a string.')
    name_no_ext, ext = os.path.splitext(filename)
    output_filename = f'{name_no_ext}{ANNOTATED_SUFFIX}{ext}'
    return output_filename


def _is_annotated_filename(filename: str):
    if not isinstance(filename, str):
        raise ValueError(f'filename {filename} is not a string.')
    name_no_ext, _ = os.path.splitext(filename)
    return name_no_ext.endswith(ANNOTATED_SUFFIX)


def annotate_frame_and_time(input_filename: str, overwrite: bool = True):
    if _is_annotated_filename(input_filename):
        print(f'File {input_filename} is already annotated. Skipping.')
        return

    output_filename = _to_annotated_filename(input_filename)

    if os.path.exists(output_filename):
        if not overwrite:
            print(f'{output_filename} already exists. Skipping.')
            return
        os.remove(output_filename)

    print(f'Annotating "{input_filename}"...')
    font_filename = os.path.abspath(os.path.join('resources', 'Apex-Medium.ttf'))
    stream = ffmpeg.input(input_filename)

    # Draw text. See https://kkroening.github.io/ffmpeg-python/#ffmpeg.drawtext.
    boxborderw = 8
    common_kwargs: dict[str] = dict(
        x=f'w*.97-tw-{boxborderw}',
        fontfile=font_filename,
        escape_text=False,
        fontcolor='White',
        box=1,
        boxcolor='Navy',
        expansion='normal',
        boxborderw=boxborderw,
        fontsize=36)
    stream = ffmpeg.drawtext(stream, text='Frame: %{frame_num}', y='h*0.88-lh', **common_kwargs)

    stream = ffmpeg.output(stream, output_filename)
    ffmpeg.run(stream, overwrite_output=True, quiet=True)
    print(f'Done annotating "{input_filename}".')


if __name__ == '__main__':
    main()
