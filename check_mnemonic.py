import os
import sys


def check_mnemonic(path):
    for subpath in os.listdir(path):
        complete_subpath = os.path.join(path, subpath)
        if complete_subpath.endswith('.py') and \
            os.path.abspath(os.path.join(path, complete_subpath)) != os.path.abspath(__file__):
            with open(complete_subpath, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if 'FIXME' in lines[i] or 'TODO' in lines[i] or 'XXX' in lines[i]:
                        print('file', complete_subpath, 'line', i + 1, 'content', lines[i].strip())
        elif os.path.isdir(complete_subpath):
            check_mnemonic(complete_subpath)


if __name__ == "__main__":
    check_mnemonic(sys.argv[1])

