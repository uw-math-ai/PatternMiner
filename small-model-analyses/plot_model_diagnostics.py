"""Render the recorded P16/P18 assessments without fitting or evaluating new models."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'model_diagnostics_data.json').read_text())
BLUE, ORANGE, GREY, RED = '#2E86AB', '#D18B24', '#b0b0b0', '#C73E1D'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

def finish(fig, name):
    fig.savefig(HERE / f'{name}.pdf', bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)

def grouped(name, counts, labels, names, colors):
    fig, ax = plt.subplots(figsize=(6.4, 2.65))
    fig.subplots_adjust(top=.76, bottom=.20)
    width = .21
    for i, (pairs, label, color) in enumerate(zip(counts, names, colors)):
        x = [j+(i-(len(counts)-1)/2)*width for j in range(2)]
        bars = ax.bar(x, [100*c/n for c,n in pairs], width, label=label, color=color)
        for bar, (c,n) in zip(bars, pairs):
            ax.annotate(f'{c}/{n}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,3), textcoords='offset points', ha='center', fontsize=8)
    ax.set(ylim=(0,113), ylabel='Accuracy (%)', xticks=[0,1], xticklabels=labels)
    ax.set_yticks([0,25,50,75,100])
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.legend(loc='upper center', bbox_to_anchor=(.5,.99), ncol=len(names),
               frameon=False, fontsize=8)
    finish(fig, name)

p16 = DATA['p16']['accuracy_series']
assert p16[1]['counts'] == p16[2]['counts'] == [[30,33],[1,2]]
grouped('kl_model_assessment', [x['counts'] for x in p16],
        ['Training terms', 'Withheld pair'],
        ['Majority baseline', 'Stump using atom count', 'Stump using adjacency'],
        [GREY, BLUE, ORANGE])

p18 = DATA['p18']['accuracy_series']
grouped('immanant_model_assessment', [x['counts'] for x in p18],
        ['Ordinary holdout', 'Mined negatives (reused)'],
        ['Majority baseline', 'Initial stump', 'Revised tree'],
        [GREY, BLUE, ORANGE])

fig, axes = plt.subplots(1,2,figsize=(6.4,2.8),sharey=True)
fig.subplots_adjust(top=.81,bottom=.26,wspace=.17)
for ax, item, title in zip(axes, DATA['p18']['intervention_series'],
                           ['Initial stump: 15 mined negatives', 'Revised tree: 14 routed negatives']):
    pairs = item['correct_counts']
    bars = ax.bar(range(4), [100*c/n for c,n in pairs], color=[GREY,BLUE,ORANGE,RED])
    for bar,(c,n) in zip(bars,pairs):
        ax.annotate(f'{c}/{n}', (bar.get_x()+bar.get_width()/2,bar.get_height()),
                    xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.set(ylim=(0,113),xticks=range(4),
           xticklabels=['Original','Identity\npatch','Unused\ninput','Branch\npatch'])
    ax.set_title(title,fontsize=9)
    ax.tick_params(axis='x',labelsize=8)
    ax.set_yticks([0,25,50,75,100])
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Accuracy (%)')
finish(fig,'immanant_branch_tests')
