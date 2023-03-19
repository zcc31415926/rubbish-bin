import subprocess
import os


os.system('ipconfig.exe > .ip.txt')
with open('.ip.txt', 'r', encoding='gbk') as f:
    lines = f.readlines()
os.system('rm .ip.txt')

i = 0
while i < len(lines):
    if 'WSL' in lines[i]:
        break
    else:
        i += 1
assert i < len(lines), '[ERROR] no WSL IPs in ipconfig.exe'

target_lines = lines[i :]
i = 0
while i < len(target_lines):
    if 'IPv4' in target_lines[i]:
        break
    else:
        i += 1
assert i < len(lines), '[ERROR] no WSL IPv4 in ipconfig.exe'

target_content = target_lines[i].strip().split(':')[-1]
ip = target_content.strip()
print(ip)

