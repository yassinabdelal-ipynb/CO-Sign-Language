"""Bounded synthetic-input inference check. Does not measure clinical accuracy."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from realtime_detection import load_model, predict
from isef_preprocess import training_array


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repetitions',type=int,default=10)
    parser.add_argument('--output',type=Path,default=Path('local/checkpoint-verification.json'))
    args=parser.parse_args()
    if not 0<=args.warmup<=100 or not 1<=args.repetitions<=1000:
        parser.error('Require 0<=warmup<=100, 1<=repetitions<=1000')
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    torch.manual_seed(2025); torch.set_num_threads(4)
    rng=np.random.default_rng(2025)
    image=Image.fromarray(rng.integers(0,256,(150,350),dtype=np.uint8))
    setup_start=time.perf_counter()
    model,names=load_model(args.checkpoint,args.device)
    if args.device=='cuda': torch.cuda.synchronize()
    setup_ms=(time.perf_counter()-setup_start)*1000
    tensor=torch.from_numpy(training_array(image)).to(args.device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(tensor)
            predict(model,image,args.device)
        if args.device=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        samples=[]; device_samples=[]
        for _ in range(args.repetitions):
            if args.device=='cuda':
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(); model(tensor); b.record(); b.synchronize()
                device_samples.append(a.elapsed_time(b))
            start=time.perf_counter(); output=predict(model,image,args.device)
            samples.append((time.perf_counter()-start)*1000)
        assert np.isfinite(output).all() and output.shape==(6,)
        numerical={'finite':True,'output_shape':[6]}
        if args.device=='cuda':
            gpu=output.copy(); model.cpu()
            cpu=predict(model,image,'cpu')
            numerical.update(max_absolute_probability_difference=float(np.max(np.abs(cpu-gpu))),atol=.0001,rtol=.001,
                cpu_cuda_allclose=bool(np.allclose(cpu,gpu,atol=.0001,rtol=.001)))
            if not numerical['cpu_cuda_allclose']: raise ValueError('CPU/CUDA mismatch')
    root=Path(__file__).resolve().parents[1]
    report={'schema_version':1,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['realtime_detection.py','isef_preprocess.py','scripts/verify_checkpoint.py']},
        'input':'procedurally generated uint8 grayscale noise; not evaluation data',
        'seed':2025,'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'classes':names,'parameters':sum(p.numel() for p in model.parameters()),
        'device':args.device,'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        'vram_bytes':torch.cuda.get_device_properties(0).total_memory if args.device=='cuda' else None,
        'os':platform.platform(),'python':platform.python_version(),
        'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','Pillow']},
        'cuda_runtime':torch.version.cuda,'warmup':args.warmup,'warmup_scope':'both model-only and full preprocessing/inference requests','repetitions':args.repetitions,
        'setup_ms':setup_ms,'request_ms':samples,'device_ms':device_samples,
        'request_median_ms':statistics.median(samples),'request_min_ms':min(samples),'request_max_ms':max(samples),
        'request_stdev_ms':statistics.stdev(samples) if len(samples)>1 else 0,
        'numerical':numerical,'failed_samples':0,
        'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        'caveats':'Laptop clocks, power and other activity uncontrolled. Model-only CUDA events and request preprocessing/transfer/output time measured separately. No camera, face detector or eye-state gate measured. No accuracy inferred.'}
    try:
        report['driver']=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    except OSError: report['driver']=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'device':args.device,'numerical':numerical,'median_request_ms':report['request_median_ms'],'output':str(args.output)}))


if __name__=='__main__': main()
