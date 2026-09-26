"""Export the completed development interventions as static research figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import session as s
from full_resources import Budget


def main(args):
    def read(name):return json.loads((s.HERE/(name+'.json')).read_text())
    full=read('EXPANDED_DECOMPOSITION')['summaries']['other_neuron_orbits']
    endpoint=read('ENDPOINT_RECEPTION')['summaries']['other_neuron_orbits']
    control=read('SUPPORT_CONTROLS')['summaries']['all']['top_32']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    def panel(ax,labels,entries,title):
        mean=np.array([v['factor_group_mean'] for v in entries]);bounds=np.array([v['group_bootstrap_95_percent'] for v in entries])
        ax.bar(np.arange(len(labels)),mean,color='#3274a1',width=.65)
        ax.errorbar(np.arange(len(labels)),mean,yerr=np.stack([mean-bounds[:,0],bounds[:,1]-mean]),fmt='none',ecolor='#222222',capsize=4)
        ax.set_xticks(np.arange(len(labels)),labels);ax.axhline(0,color='#555555',linewidth=.6)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel('Mean change in target contrast')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    panel(axes[0,0],['16 pairs','32 pairs','64 pairs','All 384 pairs'],[full[f'top_{k}'] for k in [16,32,64,384]],'A  Fixed hidden units · 304 factor orbits')
    panel(axes[0,1],['Stronger\nposition','Other\nposition','Both\npositions'],[endpoint[k] for k in ['dominant_position','other_position','positions']],'B  One inverse position per hidden unit')
    panel(axes[1,0],['Keys','Values','Keys and values'],[endpoint[k] for k in ['keys','values','keys_and_values']],'C  Reception in the second attention layer')
    panel(axes[1,1],['Opposite target\nsupport','Same target\nsupport'],[control[k] for k in ['opposite','same']],'D  Matched donors · 202 factor orbits')
    fig.suptitle('Development circuit interventions',fontsize=15,fontweight='bold')
    outputs=[]
    for extension in ['png','pdf','svg']:
        path=s.ART/(args.name+'.'+extension);fig.savefig(path,dpi=180);outputs.append(dict(path=str(path.relative_to(s.HERE)),sha256=s.sha(path)))
    plt.close(fig)
    report=dict(status='completed',outputs=outputs,source_sha256=s.sha(Path(__file__)),
        sources={name:s.sha(s.HERE/(name+'.json')) for name in ['EXPANDED_DECOMPOSITION','ENDPOINT_RECEPTION','SUPPORT_CONTROLS']},
        caption='Means give equal weight to canonical factor orbits. Error bars are exploratory 95% bootstrap intervals over those orbits. Panels A–C exclude all 53 factor orbits used in the earlier neuron analysis; panel D uses the preselected matched donors. A hidden-unit pair consists of indices k and k+384. The target contrast compares the valid indices of the paired opposite-support example with those of the destination. These effects are not additive and do not constitute independent confirmation. The selected checkpoint has minimum development accuracy 98.20%.')
    s.write(args.name+'.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='CIRCUIT_FIGURE');args=parser.parse_args()
    budget=Budget(args.name.lower(),60);status='failed_or_partial'
    try:main(args);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
