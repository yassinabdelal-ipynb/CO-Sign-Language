"""Audit the Git publication set (not ignored local data/model assets)."""
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
result=subprocess.run(['git','ls-files','--cached','-z'],cwd=root,capture_output=True,check=True)
paths=[Path(p.decode('utf-8')) for p in result.stdout.split(b'\0') if p]
errors=[]
if not paths: errors.append('Nothing tracked/staged for publication')
for rel in paths:
    path=root/rel
    if rel.parts[0] in {'data','weights','local','.venv','runs'} or path.suffix.lower() in {'.pt','.pth','.dat','.mp4','.pem','.key'}:
        errors.append(f'Private/binary asset in publication: {rel}')
    if path.stat().st_size>10_000_000: errors.append(f'Oversized publication file: {rel}')
    if path.suffix.lower() in {'.png','.jpg'}: continue
    text=path.read_text(encoding='utf-8')
    if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',text):
        errors.append(f'Potential credential: {rel}')
    if re.search(r'(?:C:|D:)[\\/](?:Users|Hack-nation)',text,re.I): errors.append(f'Personal absolute path: {rel}')
    if path.suffix=='.ipynb':
        notebook=json.loads(text)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                if cell.get('outputs') or cell.get('execution_count') is not None: errors.append('Notebook outputs not cleared')
                compile(''.join(cell['source']),str(rel),'exec')
print(json.dumps({'files_checked':len(paths),'errors':errors},indent=2))
sys.exit(1 if errors else 0)
