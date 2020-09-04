#!/usr/bin/python3

import os.path
import sqlite3
import argparse


# import-samples.py label ac files...
#
# For each sound file given, insert record into labels table with given label.

# all paths to sound files are relative to REPO_BASE
home=os.path.expanduser("~")
REPO_BASE = os.path.join(home, 'rudimetrics', 'trdata', 'repo')
REPO_DB = os.path.join(REPO_BASE, 'repo.db')

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", help='delete records from samples table', action="store_true", default=False)
    ap.add_argument("-l", "--label", dest="label", help="use label for all files", required=True)
    ap.add_argument("--impure", action="store_true", default=False,
                      help='flag files in repo so some distortions can be skipped later')
    ap.add_argument("--background", action="store_true", dest="background", help="sample is a background", default=False)
    ap.add_argument("files", nargs='*', help='list of files to register in repo')

    args = ap.parse_args()

    tableName = 'samples'
    if args.background:
        tableName = 'backgrounds'

    conn = sqlite3.connect(REPO_DB) # default autocommit ... apparently not

    if args.wipe:
        c = conn.cursor()
        c.execute('DELETE FROM {}'.format(tableName))
        conn.commit()
        print('deleted {} rows from table {}'.format(c.rowcount, tableName))
        c.close()

    sql = 'INSERT INTO {} (filepath,labels,effects) VALUES (?,?,?)'.format(tableName)
    impure = 0
    if args.impure:
        impure = 1
    c = conn.cursor()

    for fn in args.files:
        print(fn)
        c.execute(sql, (fn, args.label, impure))

    c.close()
    conn.commit()
    conn.close()

if __name__ == '__main__':
    main()

