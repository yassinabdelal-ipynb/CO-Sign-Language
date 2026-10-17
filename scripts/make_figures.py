"""Reproduce graphs from provided numerical artifacts; no reconstructed samples."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white',
    'axes.labelcolor':'#222222','text.color':'#222222','svg.fonttype':'none'})
history=json.loads((ROOT/'docs/evidence/history.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
epochs=np.arange(1,len(history['loss'])+1)
for ax,metric,label in zip(axes,['accuracy','loss'],['Accuracy','Cross-entropy loss']):
    ax.plot(epochs,history[metric],color='#222222',marker='o',markersize=3,label='Training')
    ax.plot(epochs,history['val_'+metric],color='#666666',linestyle='--',marker='s',markersize=3,label='Validation')
    ax.set(xlabel='Epoch',ylabel=label,xticks=epochs)
    ax.grid(axis='y',color='#dddddd',linewidth=.5); ax.legend(frameon=False)
fig.suptitle('Supplied training run · validation used for checkpoint selection',fontsize=11)
for ext in ['svg','png']: fig.savefig(OUT/f'training-history.{ext}',dpi=180)
plt.close(fig)

artifact=json.loads((ROOT/'docs/evidence/confusion_matrix.json').read_text())
cm=np.asarray(artifact['counts']); names=artifact['classes']
fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
ax.imshow(cm,cmap='Greys',vmin=0,vmax=cm.max())
for row in range(len(names)):
    for column in range(len(names)):
        value=cm[row,column]
        ax.text(column,row,str(value),ha='center',va='center',color='white' if value>cm.max()/2 else '#222222')
ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,
    xlabel='Predicted class',ylabel='Recorded class',title='Supplied validation confusion matrix · not rerun')
for ext in ['svg','png']: fig.savefig(OUT/f'confusion-matrix.{ext}',dpi=180)
plt.close(fig)

def diagram(filename,labels):
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    ax.set(xlim=(0,11),ylim=(0,6)); ax.axis('off')
    for i,(title,note) in enumerate(labels):
        column=i//4; row=i%4; x=.2+column*3.7; y=4.7-row*1.35
        box=FancyBboxPatch((x,y),3.25,.95,boxstyle='round,pad=0.04,rounding_size=.08',
                           edgecolor='#444444',facecolor='#f4f4f4',linewidth=.9)
        ax.add_patch(box); ax.text(x+.12,y+.65,title,fontsize=10,weight='bold')
        ax.text(x+.12,y+.28,note,fontsize=8)
        if i+1<len(labels):
            if row<3: start,end=(x+1.6,y),(x+1.6,y-.35)
            else: start,end=(x+3.3,y+.48),(x+3.6,5.15)
            ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#444444','connectionstyle':'angle3'})
    for ext in ['svg','png']: fig.savefig(OUT/f'{filename}.{ext}',dpi=180)
    plt.close(fig)

diagram('collection-flow',[
    ('1  Camera frame','Request 30 FPS; record observed rate'),
    ('2  Face detection','MediaPipe 0.10.21 · no face → reject'),
    ('3  Facial landmarks','dlib 68-point predictor · explicit asset'),
    ('4  Eye rectangle','Choose image-left/right; padded crop'),
    ('5  Collection image','Grayscale · nearest resize 350 × 150'),
    ('6  Optional augmentation','Keep original + seeded noise derivative'),
    ('7  Eye-state gate','YOLO weights missing; manual mode labeled'),
    ('8  Save or reject','Closed retained only for blink label'),
    ('9  Private session','PNG + per-frame JSONL + metadata'),
    ('10  Split by participant/session','Keep derivatives in their parent split'),
    ('11  Model input','PIL grayscale · nearest 224 × 224 · /255'),
    ('12  Inference / training','No random augmentation during inference')])

diagram('model-flow',[
    ('1  Eye tensor','N × 1 × 224 × 224 · float32 /255'),
    ('2  Patches','7 × 7 patches · 1,024 tokens'),
    ('3  Embedding','49 → 64 projection + learned positions'),
    ('4  Encoder × 12','LayerNorm → attention → residual'),
    ('5  Multi-head attention','8 heads × key dimension 64 = 512 inner'),
    ('6  Feed-forward residual','64 → 128 → 64 · GELU · dropout 0.1'),
    ('7  Pool and classify','LayerNorm → mean → MLP → 6 logits'),
    ('8  Runtime decoder','Confidence + dwell + center re-arm'),
    ('9  Commands','26 letters + 3 phrases · new mapping'),
    ('10  Parameter count','1,879,878 · fixed-token ViT'),
    ('11  Intention classifier','Original code absent · not active'),
    ('12  Scope','Direction prediction ≠ clinical validation')])
print('Rendered training, confusion, collection and model figures.')
