"""Render readable white-background previews of original Canva graph images.

Original bytes stay in originals/. This script does not extract, fit or invent
numerical data. Transparent source pixels are composited by Matplotlib.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root = Path(__file__).resolve().parents[1] / 'docs/figures/canva'
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for entry in manifest['figures']:
    source = root / entry['original']
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry['original_sha256']:
        raise ValueError('Original image changed: '+entry['original'])
    with Image.open(source) as image:
        width,height = image.size
        fig = plt.figure(figsize=(width/100,height/100),dpi=100,facecolor='white')
        ax = fig.add_axes([0,0,1,1])
        ax.imshow(image,interpolation='nearest')
        ax.set_axis_off()
        fig.savefig(root/entry['preview'],dpi=100,facecolor='white',transparent=False)
        plt.close(fig)
print(f'Rendered {len(manifest["figures"])} historical graph previews; original hashes verified.')
