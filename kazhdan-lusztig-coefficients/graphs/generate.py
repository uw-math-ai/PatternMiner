"""Plot E046 and post hoc source-omission refits; requires matplotlib."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'research/src'))
from placement_dataset import best_stumps
BLUE, ORANGE, GREY, RED = '#2E86AB', '#D18B24', '#b0b0b0', '#C73E1D'


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'research/artifacts/publication_replay.json'
    result = json.loads(source.read_text())['checks']['e046']['result']
    train, holdout = result['training_rows'], result['holdout_rows']
    rules = result['best_rules']
    labels = [[r['label'] for r in rows] for rows in (train, holdout)]
    majority = int(sum(labels[0]) > len(train) / 2)
    counts = [[sum(y == majority for y in group), len(group)] for group in labels]
    series = [{'name': 'Training-majority baseline (zero)', 'counts': counts, 'color': GREY}]
    names = ['Rank-one stump', 'Simple-sided stump']
    predictions = []
    for name, (errors, feature, cut, direction), color in zip(names, rules, [BLUE, ORANGE]):
        pred = [[int(direction * row[feature] > direction * cut) for row in rows]
                for rows in (train, holdout)]
        counts = [[sum(a == b for a, b in zip(p, y)), len(y)] for p, y in zip(pred, labels)]
        assert counts[0][0] == len(train) - errors == 30
        series.append({'name': name, 'counts': counts, 'color': color})
        predictions.append(pred[1])
    assert predictions == result['holdout_predictions'] == [[1, 1], [1, 1]]
    assert labels[1] == [0, 1] and majority == 0
    assert all(r['r1'] == 15 and r['simple_sided'] == 1 for r in holdout)

    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150,
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(top=.78, bottom=.22)
    handles = []
    for i, item in enumerate(series):
        xs = [j + (i - 1) * .23 for j in range(2)]
        bars = ax.bar(xs, [100 * c / n for c, n in item['counts']], .23,
                      color=item['color'], label=item['name'])
        handles.append(bars)
        for bar, (correct, total) in zip(bars, item['counts']):
            ax.annotate(f'{100 * correct / total:.1f}%\n({correct}/{total})',
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    ax.set(ylim=(0, 110), ylabel='Accuracy (%)', xticks=[0, 1],
           xticklabels=['Training terms (n = 33)', 'Withheld counterexample pair (n = 2)'])
    ax.set_yticks(range(0, 101, 20))
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    fig.suptitle('P16 / E046: fitted stumps fail to separate the withheld pair', y=.98, fontsize=14)
    fig.legend(handles, [s['name'] for s in series], loc='upper center',
               bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .06, 'Fixed curated terms; the two assessment cases are deliberately selected.\n'
             'The majority label is chosen on training data. These are not population accuracy estimates.',
             ha='center', fontsize=10, color='#444444')
    save(fig, output, 'fig1_accuracy_vs_baseline')

    matrix = [labels[1], *predictions]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=.28, right=.95, top=.80, bottom=.27)
    ax.imshow(matrix, cmap=ListedColormap(['#e2e2e2', BLUE]), vmin=0, vmax=1, aspect='auto')
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            wrong = i > 0 and value != labels[1][j]
            ax.text(j, i, f'{value}' + ('  (error)' if wrong else ''), ha='center', va='center',
                    color='white' if value else '#222222', fontsize=15, weight='bold')
            if wrong:
                ax.add_patch(Rectangle((j - .48, i - .47), .96, .94, fill=False,
                                       edgecolor=RED, linewidth=3))
    ax.set_xticks([0, 1], [f'C016: descent {r["descent"]}' for r in holdout])
    ax.set_yticks([0, 1, 2], ['Exact KL label', *names])
    ax.tick_params(length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle('P16 / E046: identical selected features, different exact labels', y=.97, fontsize=14)
    fig.text(.5, .87, '0 = zero coefficient; 1 = positive coefficient; red outline = wrong prediction',
             ha='center', fontsize=10)
    fig.text(.5, .09, 'Both cases have rank-one count 15 and simple-sided indicator 1.\n'
             'Both fitted rules predict positive; other recorded features can differ.\n'
             'Rules: rank-one count > 12.5; simple-sided indicator > 0.5.', ha='center', fontsize=11)
    save(fig, output, 'fig2_withheld_pair')

    # Post hoc diagnostics on the already published rows, separate from E046's
    # historical assessment. Threshold selection never uses the omitted rows.
    candidates = best_stumps(train)
    assert [list(r) for r in candidates if r[0] == candidates[0][0]] == rules
    features = sorted({r[1] for r in candidates})
    feature_comparison = []
    for feature in features:
        available = [r for r in candidates if r[1] == feature]
        error = available[0][0]
        feature_comparison.append({'feature': feature, 'correct': len(train) - error,
                                   'total': len(train), 'training_errors': error,
                                   'minimizers': [r for r in available if r[0] == error]})
    feature_comparison.sort(key=lambda r: (r['training_errors'], r['feature']))
    fig, ax = plt.subplots(figsize=(10.5, 7))
    fig.subplots_adjust(left=.22, top=.84, bottom=.17, right=.87)
    values = [100 * r['correct'] / r['total'] for r in feature_comparison]
    bars = ax.barh(range(len(values)), values,
                   color=[BLUE if r['training_errors'] == 3 else GREY for r in feature_comparison])
    for bar, row in zip(bars, feature_comparison):
        ax.annotate(f'{row["correct"]}/{row["total"]}',
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', va='center', fontsize=10)
    ax.set(yticks=range(len(values)), yticklabels=[r['feature'] for r in feature_comparison],
           xlim=(0, 105), xlabel='Best training accuracy using this feature alone (%)')
    ax.invert_yaxis()
    baseline = 100 * series[0]['counts'][0][0] / len(train)
    ax.axvline(baseline, color=RED, linestyle=':', label='Training-majority baseline (17/33)')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    fig.suptitle('P16 / E046: exhaustive single-feature threshold comparison', y=.98, fontsize=14)
    fig.text(.5, .91, 'Each feature gets its own best training threshold and polarity; all 14 inputs are shown.',
             ha='center', fontsize=10)
    fig.text(.5, .05, 'Post hoc display of the recorded training task, not a feature-addition experiment.\n'
             'The known C016 pair is not used to fit or rank these rules; training accuracy is not validation.',
             ha='center', fontsize=10)
    save(fig, output, 'fig3_feature_comparison')

    folds = []
    for group in sorted({row['source'] for row in train}):
        fitting = [row for row in train if row['source'] != group]
        assessment = [row for row in train if row['source'] == group]
        candidates = best_stumps(fitting)
        tied = [r for r in candidates if r[0] == candidates[0][0]]
        # Keep every training minimizer, rather than select using assessment.
        scores = [sum(int(direction * r[feature] > direction * cut) == r['label'] for r in assessment)
                  for _, feature, cut, direction in tied]
        baseline_label = int(sum(r['label'] for r in fitting) > len(fitting) / 2)
        assert [list(r[1:]) for r in tied] == [r[1:] for r in rules]
        assert len(set(scores)) == 1
        folds.append({'omitted_source': group, 'fit_rows': len(fitting), 'assessment_rows': len(assessment),
                      'training_minimizers': tied, 'correct_per_minimizer': scores,
                      'assessment_labels': [r['label'] for r in assessment],
                      'training_majority_label': baseline_label,
                      'baseline_correct': sum(r['label'] == baseline_label for r in assessment)})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(top=.77, bottom=.24)
    for offset, color, name, key in [(-.18, GREY, 'Fold-training majority', 'baseline_correct'),
                                      (.18, BLUE, 'Both training-selected stumps', 'correct_per_minimizer')]:
        scores = [r[key][0] if isinstance(r[key], list) else r[key] for r in folds]
        bars = ax.bar([i + offset for i in range(len(folds))],
                      [100 * c / r['assessment_rows'] for c, r in zip(scores, folds)], .36, color=color, label=name)
        for bar, c, row in zip(bars, scores, folds):
            ax.annotate(f'{c}/{row["assessment_rows"]}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    ax.set(ylim=(0, 113), ylabel='Accuracy on omitted source (%)', xticks=range(len(folds)),
           xticklabels=[r['omitted_source'].replace('C014-terminal', 'C014-\nterminal') + f'\n(n = {r["assessment_rows"]})' for r in folds])
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.suptitle('P16: omitting one source leaves the selected rules unchanged', y=.98, fontsize=14)
    fig.legend(loc='upper center', bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    fig.text(.5, .045, 'Post hoc refits on four of the five known source groups; all tied training minimizers retained.\n'
             'This is sensitivity analysis on the original 33 rows, not fresh-data or size generalization.\n'
             'Groups are source labels; cross-source interval equivalence has not been excluded.', ha='center', fontsize=10)
    save(fig, output, 'fig4_source_sensitivity')

    data = {'experiment': 'E046', 'source': source.relative_to(ROOT).as_posix(),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'matplotlib_version': matplotlib.__version__, 'majority_training_label': majority,
            'accuracy_series': series, 'withheld_labels_and_predictions': matrix,
            'withheld_rows': holdout,
            'posthoc_diagnostics': {'date': '2026-09-25', 'feature_comparison': feature_comparison,
                                   'source_omission_refits': folds,
                                   'fit_source': 'research/src/placement_dataset.py:best_stumps',
                                   'fit_source_sha256': hashlib.sha256((ROOT / 'research/src/placement_dataset.py').read_bytes()).hexdigest(),
                                   'scope': 'New descriptive analysis and deterministic refits on existing E046 rows; not historical extra runs or independent confirmation.'},
            'scope': 'Historical visualizations plus separately labeled post hoc diagnostics; no new mathematical examples.'}
    (output / 'plot_data.json').write_text(json.dumps(data, indent=2) + '\n')
    print('PASS: four P16 figures, PNG/PDF and underlying counts written to', output)


if __name__ == '__main__':
    main()
