import os
import sys


drive_letter = sys.argv[1]
target_dir = sys.argv[2]

if not drive_letter.endswith(':'):
    drive_letter.append(':')
if not os.path.exists(target_dir):
    os.system(f'sudo mkdir {target_dir}')
elif len(os.listdir(target_dir)) > 0:
    print(f'[ERROR] a disk has already been mounted to {target_dir}')
    sys.exit(0)
os.system(f'sudo mount -t drvfs {drive_letter} {target_dir}')
print(f'disk {drive_letter} mounted to {target_dir}')

