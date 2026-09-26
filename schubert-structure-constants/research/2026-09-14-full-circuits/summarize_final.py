"""Collect the completed final-model results and draw a standalone figure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import session as s
from full_resources import Budget


def read(name):return json.loads((s.HERE/(name+'.json')).read_text())


def main(budget):
    selected=read('SECOND_MODEL_SELECTION');digest=s.sha(s.HERE/selected['checkpoint'])
    assert digest==selected['checkpoint_sha256']
    assessment=read('SECOND_ASSESSMENT');formal=read('SECOND_FORMAL_STUDY_STATUS')
    path=read('FINAL_DEVELOPMENT_DECOMPOSITION');control=read('FINAL_SUPPORT_CONTROLS');scan=read('FINAL_COMPONENT_SCAN');heads=read('DEVELOPMENT_HEAD_CONTROLS')
    assert assessment['status']=='completed'
    assert all(r['status']=='completed' and r['checkpoint_sha256']==digest for r in [path,control,scan,heads])
    primary=path['summaries']['other_neuron_orbits'];ratio=primary['top_32']['factor_group_mean']/primary['top_384']['factor_group_mean']
    ranked=sorted(scan['sites'],key=lambda k:-scan['sites'][k]['all']['factor_group_mean_change'])
    with np.load(s.ART/'DEVELOPMENT_HEAD_CONTROLS.npz') as archive:
        def pair_accuracy(key):
            a=archive[key];return float((a[::2]&a[1::2]).mean())
        head_metrics=dict(selected=heads['selection']['selected'],checks=heads['checks'],
            native_pair_accuracy=pair_accuracy('baseline_good'),
            ablated_selected_pair_accuracy=pair_accuracy('selected_zero_dest_choice'),
            retained_selected_pair_accuracy=pair_accuracy('retain_selected_dest_choice'),
            random_retained_pair_accuracies=[pair_accuracy(f'random_{k:02d}_retain_dest_choice') for k in range(20)],
            all_input_conditions={key:value['all'] for key,value in heads['conditions'].items()},
            role='Development; controls preserve layer counts and widening pairs. All MLPs and residual paths remain during head retention.')
    report=dict(status='completed',checkpoint=selected['checkpoint'],checkpoint_sha256=digest,
        development_criterion=selected['criterion'],independently_qualified=assessment['passes'],
        independent_populations=assessment['populations'],formal_status=formal['status'],
        development_pairs=scan['pairs'],development_factor_orbits=scan['factor_orbits'],
        development_pair_accuracy=scan['unmodified_both_pair_accuracy'],
        fixed_path_other_neuron_orbits=primary,fixed_32_to_full_mean_effect_ratio=ratio,
        same_support_controls=control['summaries']['all'],
        component_ranking=ranked,largest_component_effects={k:scan['sites'][k]['all'] for k in ranked[:12]},
        development_head_controls=head_metrics,
        complete_multiplication_algorithm_recovered=False,
        mathematical_results=['Support component factorization from the frozen primitive leading-form theorem',
            'Polynomial-time evaluation of all Levi necessary length inequalities',
            'An S6 obstruction to sufficiency of all those inequalities, stable under adjoining fixed points',
            'A d+1 component bound for detecting failure of the necessary degree inequalities; sharp in degree one',
            'Every proper Levi ordinary nonvanishing test containing the candidate passes for the S6 obstruction',
            'Eight retained pairs have identical full candidate features and relations but disjoint valid witness sets'],
        limitations=['Empirical accuracy applies to the recorded finite populations and sampling bounds.',
            'Development interventions are adaptive evidence. Single-component effects are not additive.',
            'The mathematical deductions depend on the frozen general proofs and await independent review.'],
        source_sha256=s.sha(Path(__file__)))
    if assessment['passes']:
        assert formal['status']=='completed'
        fp=read('FORMAL_MLP_PATH');fc=read('FORMAL_CONFIRMATION');fd=read('FORMAL_DISCOVERY')
        primary_formal=fp['summaries']['all']['selected_minus_random_16']
        report['formal']=dict(discovery_accuracy=fd['unmodified_both_pair_accuracy'],
            confirmation_accuracy=fc['unmodified_both_pair_accuracy'],heads=fc['selection']['selected'],
            conditions=fc['conditions'],fixed_path_primary=primary_formal,
            fixed_path_primary_positive=primary_formal['group_bootstrap_95_percent'][0]>0)
    assert not (s.HERE/'FINAL_OUTCOME.json').exists();s.write('FINAL_OUTCOME.json',report)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    history=read('second_neighbor_refinement_compact')['history']
    ax=axes[0,0]
    for key,label,color in [('raw_criterion','Current parameters','#b35806'),('moving_average_criterion','Moving average','#2166ac')]:
        ax.plot([h['step'] for h in history],[100*h[key][0] for h in history],marker='o',label=label,color=color)
    ax.axhline(99,color='0.4',ls='--',lw=1);ax.set(xlabel='Updates in final training phase',ylabel='Minimum development accuracy (%)',title='A. All 27 development criteria');ax.legend(fontsize=8)
    ax=axes[0,1];populations=list(assessment['populations'].values());x=np.arange(4)
    ax.plot(x,[100*p['supplied_accuracy'] for p in populations],'o-',label='Individual supplied inputs',color='#2166ac')
    ax.plot(x,[100*p['difficult_pair_all_supplied_accuracy'] if p['difficult_pairs'] else np.nan for p in populations],'s-',label='Both members of difficult pairs',color='#b35806')
    ax.axhline(99,color='0.4',ls='--',lw=1);ax.set(xticks=x,xticklabels=['S7','S8','S9','S10'],ylabel='Accuracy (%)',title='B. Replacement independent assessment');ax.legend(fontsize=8)
    ax=axes[1,0];numbers=[16,32,64,384];means=np.array([primary[f'top_{k}']['factor_group_mean'] for k in numbers])
    ci=np.array([primary[f'top_{k}']['group_bootstrap_95_percent'] for k in numbers]);ax.errorbar(np.arange(4),means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o-',capsize=3,color='#2166ac')
    ax.set(xticks=np.arange(4),xticklabels=['16','32','64','All 384'],xlabel='Fixed hidden-unit pairs',ylabel='Factor-orbit mean contrast change',title='C. Final-model development path')
    ax=axes[1,1];random=np.array(head_metrics['random_retained_pair_accuracies'])*100
    values=[100*head_metrics['native_pair_accuracy'],100*head_metrics['ablated_selected_pair_accuracy'],
            100*head_metrics['retained_selected_pair_accuracy'],float(random.mean())]
    ax.bar(np.arange(4),values,color=['#2166ac','#b35806','#4393c3','#999999'])
    ax.scatter(3+np.linspace(-.17,.17,len(random)),random,s=10,color='black',zorder=3)
    ax.set(ylabel='Difficult-pair accuracy (%)',title='D. Development head controls',ylim=(0,102),
           xticks=np.arange(4),xticklabels=['Native','Ablate\nselected','Retain\nselected','Retain\nrandom'])
    for extension in ['png','pdf','svg']:fig.savefig(s.ART/('FINAL_ANALYSIS_FIGURE.'+extension),dpi=180)
    plt.close(fig)
    sources=['SECOND_MODEL_SELECTION','SECOND_ASSESSMENT','SECOND_FORMAL_STUDY_STATUS','FINAL_DEVELOPMENT_DECOMPOSITION','FINAL_SUPPORT_CONTROLS','FINAL_COMPONENT_SCAN','DEVELOPMENT_HEAD_CONTROLS','second_neighbor_refinement_compact']
    s.write('FINAL_ANALYSIS_FIGURE.json',dict(status='completed',checkpoint_sha256=digest,
        caption='Training, independent accuracy, and final-model development interventions. Panel C uses factor orbits excluded from the earlier neuron ranking; bars show exploratory 95% percentile bootstrap intervals. The ratio of path effects is not a fraction of an algorithm recovered. Panel D retains all MLPs and residual paths; random controls match layer counts and widening pairs, and points show all twenty control sets. Head selection and testing use development data.',
        source_sha256=s.sha(Path(__file__)),input_sha256={name:s.sha(s.HERE/(name+'.json')) for name in sources},
        artifacts={ext:s.sha(s.ART/('FINAL_ANALYSIS_FIGURE.'+ext)) for ext in ['png','pdf','svg']}))
    print(json.dumps(dict(status='completed',independently_qualified=assessment['passes'],formal_status=formal['status'],development_pairs=scan['pairs'],fixed_32_to_full_mean_effect_ratio=ratio)),flush=True)


if __name__=='__main__':
    budget=Budget('summarize_final_results',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
