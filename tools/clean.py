#!/usr/bin/env python

import argparse
import sys
import nbformat
from nbdev.export import *
from nbdev.clean import *
from fastcore.all import *

_re_header = re.compile(r'^#+\s+\S+')
_re_clean  = re.compile(r'^\s*#\s*clean\s*')

def is_header_cell(cell): return _re_header.search(cell['source']) is not None
def is_clean_cell(cell): return _re_clean.search(cell['source']) is not None

_re_questionnaire = re.compile(r'^#+\s+Questionnaire')

def get_stop_idx(cells):
    i = 0
    while i < len(cells) and _re_questionnaire.search(cells[i]['source']) is None: i+=1
    return i

def clean_tags(cell):
    if is_header_cell(cell): return cell
    for attr in ["id", "caption", "alt", "width", "hide_input", "hide_output", "clean"]:
        cell["source"] = re.sub(r'#\s*' + attr + r'.*?($|\n)', '', cell["source"])
    return cell

def proc_nb(fname, dest_path, verify_only=False):
    """Create a cleaned version of the notebook in fname in the dest_path folder.

    return True if the current file in the dest folder needs to be modified.
    """
    nb = read_nb(fname)
    i = get_stop_idx(nb['cells'])
    nb['cells'] = [clean_tags(c) for j,c in enumerate(nb['cells']) if
                   c['cell_type']=='code' or is_header_cell(c) or is_clean_cell(c) or j >= i]
    clean_nb(nb, clear_all=True)

    clean_dest = dest_path/fname.name

    try:
        existing_nb_clean = read_nb(clean_dest)
    except FileNotFoundError:
        existing_nb_clean = None

    if nb == existing_nb_clean:
        return False
    else:
        print (f'{clean_dest} is not up to date!')
        if not verify_only:
            print (f'  ==> Modifying {clean_dest}.')
            with open(clean_dest, 'w') as f:
                nbformat.write(nb, f, version=4)
        return True

def proc_all(path='.', dest_path='clean', verify_only=False):
    """Process all the notebooks to create cleaned version in the dest_path folder.

    return True if a modification is needed.
    """
    path,dest_path = Path(path),Path(dest_path)
    fns = [f for f in path.iterdir() if f.suffix == '.ipynb' and not f.name.startswith('_')]
    need_cleaning = [proc_nb(fn, dest_path=dest_path, verify_only=verify_only) for fn in fns]

    return any(need_cleaning)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Create clean versions of the notebooks, without the prose.')

    parser.add_argument(
        '--verify-only',
        dest='verify_only',
        action='store_true',
        help='Only verify if the clean folder is up to date. Used for CI.',
    )

    args = parser.parse_args()

    exit_code = 1 if proc_all(verify_only=args.verify_only) else 0

    sys.exit(exit_code)
