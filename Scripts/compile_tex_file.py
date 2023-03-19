import os
import sys


texname = sys.argv[1]
bibname = sys.argv[2]

count = 0
for i in range(len(texname) - 1, -1, -1):
    if texname[i] == '.':
        count = i
        break
texname_wo_suffix = texname[: count]

engine = 'pdflatex'
with open(texname_wo_suffix + '.tex', 'r') as f:
    line = f.readline()
    if 'ctexart' in line or 'beamer' in line:
        engine = 'xelatex'

if os.path.exists(bibname):
    os.system(f'{engine} {texname_wo_suffix}.tex')
    os.system(f'bibtex   {texname_wo_suffix}.aux')
    os.system(f'{engine} {texname_wo_suffix}.tex')
    os.system(f'{engine} {texname_wo_suffix}.tex')
else:
    os.system(f'{engine} {texname_wo_suffix}.tex')

os.system('rm *.log')
os.system('rm *.nav')
os.system('rm *.out')
os.system('rm *.snm')
os.system('rm *.toc')
os.system('rm *.aux')
os.system('rm *.blg')
# os.system('rm *.bbl')

