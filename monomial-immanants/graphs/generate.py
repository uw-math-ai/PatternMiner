"""Plot the retained E031 classifiers and controls; requires matplotlib."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'research/src'))
from origin_stump import prediction
from origin_tree import predict

BLUE, ORANGE, GREY, RED = '#2E86AB', '#D18B24', '#b0b0b0', '#C73E1D'


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def labeled_bars(ax, positions, counts, width, color, label=None, fontsize=10):
    bars = ax.bar(positions, [100 * c / n for c, n in counts], width, color=color, label=label)
    for bar, (correct, total) in zip(bars, counts):
        ax.annotate(f'{100 * correct / total:.1f}%\n({correct}/{total})',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=fontsize)
    return bars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    files = [ROOT / 'research/artifacts/historical' / f'origin_{name}.json' for name in ('stump', 'tree')]
    original, revision = [json.loads(p.read_text()) for p in files]
    test = [r for r in original['rows'] if r['split'] == 'test']
    hard = original['hard_rows']
    groups = [test, hard]
    functions = [lambda x: prediction(original['model'], x), lambda x: predict(revision['tree'], x)]
    train = [r for r in original['rows'] if r['split'] == 'train']
    majority = int(sum(r['label'] for r in train) > len(train) / 2)
    assert majority == 1
    series = [{'name': 'Training-majority baseline (genuine)',
               'counts': [[sum(r['label'] == majority for r in rows), len(rows)] for rows in groups],
               'color': GREY}]
    for name, fn, record, color in zip(['Initial stump', 'Revised depth-two tree'], functions,
                                      [original, revision], [BLUE, ORANGE]):
        counts = [[sum(fn(r['x']) == r['label'] for r in rows), len(rows)] for rows in groups]
        assert counts == [[record[k]['correct'], record[k]['n']] for k in ('test', 'hard')]
        series.append({'name': name, 'counts': counts, 'color': color})

    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.subplots_adjust(top=.77, bottom=.24)
    handles = []
    for i, item in enumerate(series):
        handles.append(labeled_bars(ax, [j + (i - 1) * .23 for j in range(2)],
                                   item['counts'], .23, item['color']))
    ax.set(ylim=(0, 114), ylabel='Origin classification accuracy (%)', xticks=[0, 1],
           xticklabels=['Ordinary holdout (n = 279)\n268 genuine / 11 abstract',
                        'Mined challenge (n = 15)\nAll abstract; reused after inspection'])
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.suptitle('P18 / E031: ordinary accuracy hides the challenge-set failure', y=.98, fontsize=14)
    fig.legend(handles, [s['name'] for s in series], loc='upper center',
               bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .05, 'Revision adds 12 training examples; the challenge was already inspected.\n'
             'Neither model passes the zero-error gate. The task is origin classification, not connectivity.',
             ha='center', fontsize=10, color='#444444')
    save(fig, output, 'fig1_accuracy_vs_baseline')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.3), sharex=True, sharey=True)
    fig.subplots_adjust(top=.74, bottom=.22, wspace=.10)
    xlim, ylim = (0, 145), (0, 132)
    for i, (ax, fn) in enumerate(zip(axes, functions)):
        ax.set_facecolor('#e9f2f7')
        if i == 0:
            cutoff = original['model']['threshold']
            ax.add_patch(Rectangle((0, 0), cutoff, ylim[1], color='#eeeeee', zorder=0))
            ax.axvline(cutoff, color='#555555', linestyle='--', lw=1)
            rule = 'Stump: abstract if transposition ≤ 37'
        else:
            cutoff, childcut = revision['tree']['threshold'], revision['tree']['left']['threshold']
            ax.add_patch(Rectangle((0, childcut), cutoff, ylim[1] - childcut, color='#eeeeee', zorder=0))
            ax.plot([0, cutoff, cutoff], [childcut, childcut, ylim[1]], color='#555555', linestyle='--', lw=1)
            rule = 'Tree: abstract if transposition ≤ 60.5\nand five-cycle > 2.5'
        layers = [([r for r in test if r['label']], 'Genuine holdout', BLUE, 'o', 18),
                  ([r for r in test if not r['label']], 'Abstract holdout', '#555555', 'x', 35),
                  (hard, 'Mined abstract challenge', ORANGE, '^', 42)]
        for rows, label, color, marker, size in layers:
            ax.scatter([r['x'][0] for r in rows], [r['x'][1] for r in rows],
                       c=color, marker=marker, s=size, label=label, alpha=.85, zorder=3)
        errors = [r for rows in groups for r in rows if fn(r['x']) != r['label']]
        ax.scatter([r['x'][0] for r in errors], [r['x'][1] for r in errors], s=100,
                   facecolors='none', edgecolors=RED, linewidths=1.3, label='Wrong prediction', zorder=4)
        ax.set(xlim=xlim, ylim=ylim, xlabel='Transposition class coordinate', title=rule)
        ax.title.set_fontsize(11)
        if i == 1:
            for row in errors:
                x, y = row['x']
                ax.annotate(f'Error: ({x}, {y})', (x, y), xytext=(12, -24 if y > 110 else 13),
                            textcoords='offset points', fontsize=9, color=RED,
                            arrowprops={'arrowstyle': '-', 'color': RED})
    axes[0].set_ylabel('Five-cycle class coordinate')
    fig.suptitle('P18 / E031: decision boundaries and retained assessment errors', y=.98, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91),
               ncol=4, frameon=False, fontsize=10)
    fig.text(.5, .05, 'Background: grey predicts abstract; pale blue predicts genuine. Points can overlap.\n'
             'The same 279 ordinary and 15 deliberately mined targets appear in both panels;\n'
             'the revised model was assessed on the already-inspected challenge.', ha='center', fontsize=10)
    save(fig, output, 'fig2_decision_boundaries')

    interventions = []
    specs = [(original['interventions'], ['before', 'identity_patch', 'unused_coordinate_control', 'branch_patch'],
              'Stump: 15 mined targets', 'Abstract-source branch'),
             (revision['child_interventions'], ['before', 'identity_patch', 'unused_child_input_control', 'genuine_child_branch_patch'],
              'Revised tree child: 14 routed targets', 'Genuine-source branch')]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.1), sharey=True)
    fig.subplots_adjust(top=.76, bottom=.28, wspace=.12)
    for ax, (records, keys, title, donor) in zip(axes, specs):
        # All assessed intervention targets carry abstract-origin label zero.
        assert all(r['target'] in [h['target'] for h in hard] for r in records)
        counts = [[sum(r[k] == 0 for r in records), len(records)] for k in keys]
        interventions.append({'cohort': title, 'keys': keys, 'correct_counts': counts})
        labeled_bars(ax, range(4), counts, .65, [GREY, BLUE, ORANGE, RED])
        ax.set(ylim=(0, 115), xticks=range(4), title=title,
               xticklabels=['Unmodified', 'Identity\npatch', 'Unused-input\ncontrol', donor.replace(' ', '\n', 1)])
        ax.tick_params(axis='x', labelsize=10)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Correct origin predictions (%)')
    fig.suptitle('P18 / E031: branch interventions reproduce the fitted rules', y=.98, fontsize=14)
    fig.text(.5, .87, 'Patch donors differ: abstract source for the stump, genuine source for the tree child.',
             ha='center', fontsize=11)
    fig.text(.5, .06, 'Direct interventions on decision-tree booleans, with identity and unused-input controls.\n'
             'The tree panel includes only the 14 challenge targets reaching that child.\n'
             'These controls explain model predictions; they do not prove a mathematical criterion.',
             ha='center', fontsize=10)
    save(fig, output, 'fig3_branch_interventions')

    # Macro-average recall makes the original holdout's imbalance explicit.
    class_performance = []
    for name, fn in [('Majority baseline', lambda x: majority),
                     ('Initial stump', functions[0]), ('Revised tree', functions[1])]:
        counts = []
        for label in (0, 1):
            rows = [r for r in test if r['label'] == label]
            counts.append([sum(fn(r['x']) == label for r in rows), len(rows)])
        class_performance.append({'name': name, 'recall_counts': counts,
                                  'balanced_accuracy': sum(c / n for c, n in counts) / 2})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.3), gridspec_kw={'width_ratios': [1.35, 1]}, sharey=True)
    fig.subplots_adjust(top=.74, bottom=.25, wspace=.15)
    for i, (record, color) in enumerate(zip(class_performance, [GREY, BLUE, ORANGE])):
        labeled_bars(axes[0], [j + (i - 1) * .23 for j in range(2)], record['recall_counts'], .23, color, record['name'], fontsize=8)
        height = 100 * record['balanced_accuracy']
        axes[1].bar(i, height, .65, color=color)
        axes[1].annotate(f'{height:.1f}%', (i, height), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11)
    axes[0].set(xticks=[0, 1], xticklabels=['Abstract recall\n(n = 11)', 'Genuine recall\n(n = 268)'],
                ylabel='Recall or balanced accuracy (%)', title='A. Recall within each class')
    axes[1].set(xticks=range(3), xticklabels=['Majority\nbaseline', 'Initial\nstump', 'Revised\ntree'],
                title='B. Equal-weight mean of the two recalls')
    for ax in axes:
        ax.set_ylim(0, 114)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91), ncol=3, frameon=False)
    fig.suptitle('P18 / E031: class balance changes the baseline comparison', y=.98, fontsize=14)
    fig.text(.5, .05, 'Post hoc summary of the same 279 ordinary holdout targets; no refitting or new assessment.\n'
             'The all-abstract mined challenge has no genuine-class recall, so it is not included in the macro-average.\n'
             'The revised tree misses one of 11 abstract targets while retaining all 268 genuine targets.', ha='center', fontsize=10)
    save(fig, output, 'fig4_class_balanced_performance')

    rectangle_file = ROOT / 'research/artifacts/historical/rectangles.json'
    replay_file = ROOT / 'research/artifacts/publication_replay.json'
    frozen = json.loads(rectangle_file.read_text())
    replay = json.loads(replay_file.read_text())['checks']['rectangles']['result']
    assert len(frozen['witnesses']) == len(replay['witnesses']) == 5
    decomposition = []
    for witness, checked in zip(frozen['witnesses'], replay['witnesses']):
        assert witness['shape'] == checked['shape']
        assert len({tuple(t['h']) for t in witness['terms']}) == len(witness['terms'])
        histogram = Counter(t['coefficient'] for t in witness['terms'])
        assert all(c > 0 for c in histogram)
        histogram[0] = replay['hessenberg_functions'] - len(witness['terms'])
        assert sum(histogram.values()) == 132
        assert len(witness['terms']) == checked['positive_terms']
        assert sum(c * n for c, n in histogram.items()) == checked['total_multiplicity']
        decomposition.append({'shape': witness['shape'], 'coefficient_histogram': dict(sorted(histogram.items())),
                              'positive_terms': checked['positive_terms'], 'total_multiplicity': checked['total_multiplicity']})
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(top=.75, bottom=.24)
    bottom = [0] * len(decomposition)
    for coefficient, color in zip(range(4), ['#dddddd', BLUE, ORANGE, '#419D78']):
        heights = [r['coefficient_histogram'].get(coefficient, 0) for r in decomposition]
        ax.bar(range(len(decomposition)), heights, .65, bottom=bottom, color=color,
               label=f'Coefficient {coefficient}')
        bottom = [b + h for b, h in zip(bottom, heights)]
    for i, row in enumerate(decomposition):
        ax.text(i, 135, f'{row["positive_terms"]} positive\nsum = {row["total_multiplicity"]}',
                ha='center', fontsize=10)
    ax.set(ylim=(0, 153), ylabel='Number of Hessenberg indices', xticks=range(len(decomposition)),
           xticklabels=['(' + ','.join(map(str, r['shape'])) + ')' for r in decomposition],
           xlabel='Coefficient partition (all of size 12)')
    ax.set_yticks([0, 33, 66, 99, 132])
    fig.suptitle('P18: exact coefficient distributions for the six-row rectangle', y=.98, fontsize=14)
    fig.legend(loc='upper center', bbox_to_anchor=(.5, .91), ncol=4, frameon=False)
    fig.text(.5, .05, 'All 132 Hessenberg indices per shape; zero terms are included. The skew shape is (2,2,2,2,2,2).\n'
             'These are exact mathematical decompositions, not model predictions or independent size tests.\n'
             'Distinct Hessenberg indices can give identical class functions; finite examples do not replace the proof.',
             ha='center', fontsize=10)
    save(fig, output, 'fig5_exact_decompositions')

    data = {'experiments': ['E031', 'T053 rectangle witnesses'], 'sources': [dict(path=p.relative_to(ROOT).as_posix(),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [*files, rectangle_file, replay_file]],
            'matplotlib_version': matplotlib.__version__, 'majority_training_label': majority,
            'accuracy_series': series, 'stump': original['model'], 'tree': revision['tree'],
            'intervention_series': interventions,
            'class_performance': class_performance, 'exact_decompositions': decomposition,
            'scope': 'Fixed model assessments and exact rectangle witnesses; post hoc summaries, no fresh generalization claim.'}
    (output / 'plot_data.json').write_text(json.dumps(data, indent=2) + '\n')
    print('PASS: five P18 figures, PNG/PDF and underlying counts written to', output)


if __name__ == '__main__':
    main()
