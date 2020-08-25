#!/usr/bin/python3

import sys
import os.path
import sqlite3


# process-samples.py label files...
#
# For each sound file given, insert record into labels table with given label.
# If sound file needs to be converted, the original will be renamed with .orig
# extension.

# all paths to sound files are relative to REPO_BASE
home=os.path.expanduser("~")
REPO_BASE = os.path.join(home, 'rudimetrics', 'trdata', 'repo')
REPO_DB = os.path.join(REPO_BASE, 'repo.db')


def main():

    if len(sys.argv) < 3:
        print('label and at least one sound file required on command line')
        sys.exit(0)


    label = sys.argv[1]

    # verify full absolute path of any given sound file is under base_path
    for fn in sys.argv[2:]:
        check_path_is_under_repo_if_abs(fn)

    conn = sqlite3.connect(REPO_DB)

    # query existing labels, warn if new


def check_path_is_under_repo_if_abs(fn):
    pass
    #raise Exception


if __name__ == '__main__':
    main()

