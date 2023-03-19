import os
import sys


add = input('Run "git add -A"? Remember to run "git pull" first! y/N: ')
if add == 'y':
    ret = os.system('git add -A')
    commit = input('notes of "git commit -m": ')
    os.system(f'git commit -m "{commit}"')
    branch = input('branch name of "git push origin": ')
    os.system(f'git push origin {branch}')
else:
    print('Aborting...')

