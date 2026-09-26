"""Replay the fixed E046 experiment and expose all 35 generated feature rows."""
import json
from collections import Counter
from placement_dataset import TRAINING_CORES, labelled_candidates, c016_holdout, best_stumps

def main():
    training=[row for name,pair in TRAINING_CORES.items() for row in labelled_candidates(name,pair)]
    holdout=c016_holdout()
    candidates=best_stumps(training)
    winners=[rule for rule in candidates if rule[0]==candidates[0][0]]
    predictions=[[int(direction*row[feature]>direction*cut) for row in holdout]
                 for _,feature,cut,direction in winners]
    assert len(training)==33 and Counter(row['label'] for row in training)=={0:17,1:16}
    assert winners==[(3,'r1',12.5,1),(3,'simple_sided',0.5,1)]
    assert [row['label'] for row in holdout]==[0,1]
    assert predictions==[[1,1],[1,1]]
    print(json.dumps(dict(status='PASS',training_rows=training,holdout_rows=holdout,
        training_label_histogram=dict(Counter(row['label'] for row in training)),
        best_rules=winners,holdout_predictions=predictions,
        scope='Fixed curated E046 dataset; exact labels recomputed, thresholds refitted. No population generalization claim.'),indent=2))

if __name__=='__main__':main()
