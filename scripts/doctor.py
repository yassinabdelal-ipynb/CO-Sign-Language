"""Read-only runtime inventory. Does not open a camera or download models."""
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys

result={'os':platform.platform(),'python':sys.version.split()[0],
    'packages':{},'tools':{name:bool(shutil.which(name)) for name in ['git','cl','cmake','nvcc','nvidia-smi']}}
for name in ['torch','numpy','Pillow','mediapipe','dlib','opencv-contrib-python','ultralytics','tensorflow']:
    try: result['packages'][name]=metadata.version(name)
    except metadata.PackageNotFoundError: result['packages'][name]=None
try:
    import torch
    result['cuda']={'available':torch.cuda.is_available(),'runtime':torch.version.cuda,
        'devices':[{'name':torch.cuda.get_device_name(i),'vram_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]}
except ImportError: result['cuda']={'available':False,'reason':'torch missing'}
if shutil.which('nvidia-smi'):
    probe=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    result['driver_query']=probe.stdout.strip() if probe.returncode==0 else probe.stderr.strip()
print(json.dumps(result,indent=2))
