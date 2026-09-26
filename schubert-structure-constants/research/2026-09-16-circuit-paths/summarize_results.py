"""Summarize retained interventions and render a reproducible research figure."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import circuit_context as s
from pair_circuit_diagnostic import orbit_keys,summarize
from richardson_features import permutations
from circuit_resources import Budget


def read(name):return json.loads((s.HERE/(name+'.json')).read_text())


def main(budget):
    discovery=dict(np.load(s.ART/'DISCOVERY_SCAN.npz'));confirmation=dict(np.load(s.ART/'HEAD_CONFIRMATION.npz'))
    dd=read('DISCOVERY_SCAN');hd=read('HEAD_CONFIRMATION');mlp=read('MLP_CONFIRMATION')
    coal=read('COALITIONS_CONFIRMATION')['results'];endpoint=read('ENDPOINT_CONFIRMATION')['summaries']['all']
    key_sets={name:set(orbit_keys(data['x'])) for name,data in [('discovery',discovery),('confirmation',confirmation)]}
    assert not (key_sets['discovery']&key_sets['confirmation'])
    counts={}
    for name,data,meta in [('discovery',discovery,dd['pairs']),('confirmation',confirmation,read('HEAD_CONFIRMATION_MEMBERSHIP')['pairs'])]:
        good=(data['source_good']&data['dest_good']) if name=='discovery' else (data['baseline_good']&data['source_good'])
        assert np.array_equal(data['mask'][::2],data['mask'][1::2])
        assert not (data['y'][::2]&data['y'][1::2]).any()
        assert np.array_equal(data['x'][::2][data['mask'][::2]],data['x'][1::2][data['mask'][1::2]])
        counts[name]=dict(inputs=len(good),pairs=len(good)//2,factor_orbits=len(key_sets[name]),correct_pairs=int(good.sum())//2,
            pair_populations=dict(Counter(v['population'] for v in meta)),
            original_candidate_features_equal=True,complete_candidate_features_equal=int(data['candidate_features_equal'].sum())//2)
    n=len(confirmation['x']);include=np.ones(n,bool);groups=confirmation['group'];rng=np.random.default_rng(s.SEED+211)
    selected=confirmation['selected_output_change']
    random=np.stack([confirmation[f'random_{i:02d}_output_change'] for i in range(20)]).mean(0)
    head_result=dict(selected=hd['selection']['selected'],
        output_effect=summarize(selected,include,groups,rng),
        random_output_effect=summarize(random,include,groups,rng),
        selected_minus_random=summarize(selected-random,include,groups,rng),
        native_correct_pairs=counts['confirmation']['correct_pairs'],
        ablated_correct_pairs=int(confirmation['selected_zero_dest_choice'].reshape(-1,2).all(1).sum()),
        retained_correct_pairs=int(confirmation['retain_selected_dest_choice'].reshape(-1,2).all(1).sum()))
    # Cases illustrate a fixed old cohort; they are not a newly selected test set.
    source=s.FULL/'artifacts/AUGMENTED_FEATURE_OBSTRUCTION.npz';matched=dict(np.load(source))
    trace=dict(np.load(s.ART/'MATCHED_PATH_TRACE.npz'))
    p,ns=permutations(torch.as_tensor(matched['x'],device='cpu'));p=p.numpy();ns=ns.numpy()
    cases=[]
    for row in range(0,len(p),2):
        size=int(ns[row]);items=[]
        for index in [row,row+1]:
            item=dict(w=p[index,2,:size].tolist(),supplied_index=int(round(float(matched['x'][index,0,1])*size)),
                positive_A_indices=(np.flatnonzero(matched['a'][index])+1).tolist(),
                positive_B_indices=(np.flatnonzero(matched['y'][index])+1).tolist(),
                native_choice=int(trace['baseline_choice'][index])+1,
                path_top32_choice=int(trace['choice_kv1_selected_32'][index])+1,
                path_all_units_choice=int(trace['choice_kv1_selected_384'][index])+1,
                top32_path_later_attention_blocked_choice=int(trace['choice_block_all_later_attention'][index])+1,
                top32_path_later_mlps_blocked_choice=int(trace['choice_block_all_later_mlps'][index])+1)
            assert item['native_choice'] in item['positive_B_indices'];items.append(item)
        cases.append(dict(n=size,u=p[row,0,:size].tolist(),v=p[row,1,:size].tolist(),
            candidate_indices=(np.flatnonzero(matched['mask'][row])+1).tolist(),
            retained_source_rows=matched['source_rows'][row:row+2].tolist(),members=items))
    s.write('MATCHED_CASES.json',dict(role='All eight retained matched development pairs; no case omission',source_sha256=s.sha(source),cases=cases))
    summary=dict(status='completed',checkpoint_sha256=s.sha(s.CHECKPOINT),source_sha256=s.sha(Path(__file__)),
        protocol_amendment_sha256=s.sha(s.HERE/'PROTOCOL_AMENDMENT.json'),cohorts=counts,
        split_check='Canonical factor orbits of discovery and confirmation are disjoint; the earlier fixed neuron-ranking orbits are also excluded, as checked in circuit_path.py.',
        heads=head_result,mlp_primary=mlp['summaries']['all']['selected_minus_random_16'],
        mlp_top32_effect=mlp['summaries']['all']['top_32'],mlp_all_units_effect=mlp['summaries']['all']['top_384'],
        endpoint_effects=endpoint,coalition_results=coal,
        matched_cases=dict(pairs=len(cases),factor_orbits=4,correct_native_pairs=8,
            top32_source_choice_count=int(trace['source_good_kv1_selected_32'].sum()),
            all_units_source_choice_count=int(trace['source_good_kv1_selected_384'].sum()),directed_inputs=len(p)),
        interpretation='A first-MLP path sensitive to inverse positions feeds distributed key/value computation. Later attention and MLP responses contribute. The tested short serial chain does not reproduce the full intervention effect.',
        algebraic_algorithm_identified=False,exact_product_queries=0)
    s.write('RESULTS.json',summary)
    budget.check()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(15.8,4.7),layout='constrained')
    labels=['Inverse positions','Larger endpoint term','Signed difference','Left descent','Rank columns','Relative roots']
    stats=[endpoint['positions'],endpoint['dominant_position'],endpoint['gap'],endpoint['descent'],
           mlp['summaries']['all']['w_ranks'],mlp['summaries']['all']['relative_roots']]
    means=np.array([v['factor_group_mean'] for v in stats]);intervals=np.array([v['group_bootstrap_95_percent'] for v in stats])
    axes[0].barh(np.arange(len(labels)),means,color='#436b95')
    axes[0].errorbar(means,np.arange(len(labels)),xerr=np.stack([means-intervals[:,0],intervals[:,1]-means]),fmt='none',color='black',capsize=2)
    axes[0].set_yticks(np.arange(len(labels)),labels);axes[0].invert_yaxis();axes[0].set_title('A. Input contributions to the fixed MLP path')
    axes[0].set_xlabel('Mean change in score contrast');axes[0].axvline(0,color='gray',lw=.6)
    x=np.arange(8);width=.36
    for offset,mode,color,label in [(-width/2,'keys','#436b95','Keys'),(width/2,'values','#ca8642','Values')]:
        vals=[coal[mode]['shapley'][str(k)] for k in range(8)];mu=np.array([v['factor_group_mean'] for v in vals]);ci=np.array([v['group_bootstrap_95_percent'] for v in vals])
        axes[1].bar(x+offset,mu,width,color=color,label=label)
        axes[1].errorbar(x+offset,mu,yerr=np.stack([mu-ci[:,0],ci[:,1]-mu]),fmt='none',color='black',capsize=2,lw=.7)
    axes[1].set_xticks(x,[f'{i}/{i+8}' for i in range(8)],rotation=45);axes[1].set_xlabel('L1 head pair');axes[1].set_ylabel('Exact Shapley attribution');axes[1].legend(frameon=False)
    axes[1].set_title('B. Key and value contributions')
    full=read('PATH_TRACE_ALL_HEADS')['summaries']['all'];names=['Full L1 transfer','Later MLPs held fixed','Later attention held fixed','Both held fixed','Short serial chain']
    keys=['effect_kv1_selected_32','effect_block_all_later_mlps','effect_block_all_later_attention','effect_block_all_later_attention_and_mlp','effect_serial_mlp1_heads3_mlp3']
    vals=[full[k] for k in keys];mu=np.array([v['factor_group_mean'] for v in vals]);ci=np.array([v['group_bootstrap_95_percent'] for v in vals])
    axes[2].barh(np.arange(len(names)),mu,color='#628b70');axes[2].errorbar(mu,np.arange(len(names)),xerr=np.stack([mu-ci[:,0],ci[:,1]-mu]),fmt='none',color='black',capsize=2)
    axes[2].set_yticks(np.arange(len(names)),names);axes[2].invert_yaxis();axes[2].set_xlabel('Mean change in score contrast');axes[2].set_title('C. Dependence on later computation')
    fig.suptitle('Schubert witness selector: 249 confirmation pairs in 22 factor orbits',fontsize=13)
    fig.savefig(s.ART/'CIRCUIT_RESULTS.pdf');fig.savefig(s.ART/'CIRCUIT_RESULTS.png',dpi=170)
    plt.close(fig)
    print(json.dumps(dict(cohorts=counts,heads=head_result,matched=summary['matched_cases'])),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);budget=Budget('circuit_result_summary',120);status='failed_or_partial'
    try:main(budget);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
