"""Train a Schubert selector on compact exact factor-group populations."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
import common as r
import compact
import training_sampling
from budget import Budget


def training_tensors(with_counts=False,training_spec='TRAINING_DATA.json'):
    specification=json.loads((r.HERE/training_spec).read_text())
    spec=specification['sources']
    if 'base_specification_sha256' in specification:
        base_path=r.HERE/'TRAINING_DATA.json'
        assert r.c.a.sha(base_path)==specification['base_specification_sha256']
        base=json.loads(base_path.read_text())['sources']
        assert list(spec)[:len(base)]==list(base)
        assert all(spec[name]['rows']==entry['rows'] for name,entry in base.items())
    pieces={key:[] for key in ['permutations','n','lengths','coefficients']}
    eligibility=[]
    offsets=[0]
    for name,entry in spec.items():
        records=compact.load(r.HERE/f'cache/data/{name}.bin')
        assert len(records)==entry['rows']
        if 'data_sha256' in entry:assert r.c.a.sha(r.HERE/f'cache/data/{name}.bin')==entry['data_sha256']
        for key in pieces:
            values=np.array(records[key]>0 if key=='coefficients' and not with_counts else records[key],copy=True)
            if key=='coefficients' and with_counts:values=values.astype(np.int32)
            pieces[key].append(torch.as_tensor(values,device='cuda'))
        allowed=(np.load(r.HERE/'cache/legacy_reflection.npy') if name=='legacy_train'
                 else np.ones(len(records),bool))
        eligibility.append(torch.as_tensor(allowed,device='cuda'))
        offsets.append(offsets[-1]+len(records))
    return ({key:torch.cat(values) for key,values in pieces.items()},torch.cat(eligibility),
            offsets,[entry['weight'] for entry in spec.values()])


def validation_data(include_historical=False,include_exposed_assessment=False):
    data={'S7_validation':r.c.load('S7_validation'),'S8_validation':r.c.load('S8_validation'),
            **{name:dict(np.load(r.ART/f'{name}.npz')) for name in
               ['S9_validation','S10_validation','hard_validation']}}
    if include_historical:data['historical_pairs']=r.c.load('matched_pairs')
    if include_exposed_assessment:
        from prepare import encoded
        exposure=json.loads((r.HERE/'ASSESSMENT_EXPOSURE.json').read_text())
        assert exposure['role_after_failure']=='development'
        assert exposure['assessment_sha256']==r.c.a.sha(r.HERE/'qualified_assessment.json')
        for n in [9,10]:
            pairs=np.load(r.ART/f'qualified_assessment_S{n}_assessment.npz')['pairs']
            records=compact.load(r.HERE/f'cache/data/S{n}_assessment.bin')
            data[f'S{n}_exposed_pairs']=encoded(np.array(records[pairs[:,:2].reshape(-1)]),pairs[:,2].astype(bool).repeat(2))
    return data


def evaluate(model, datasets, robust=False):
    reports={}
    for name,data in datasets.items():
        for last in [False,True]:
            scores=r.c.predict(model,data,last)
            pred=np.where(data['mask'],scores,-1e20).argmax(1)
            good=data['y'][np.arange(len(pred)),pred]
            result=dict(n=len(good),top1=float(good.mean()),errors=int((~good).sum()))
            if name=='hard_validation' or name.endswith('_pairs'):
                result.update(pairs=len(good)//2,both_correct=int((good[::2]&good[1::2]).sum()),
                              selection_accuracy=float((good[::2]&good[1::2]).mean()))
            else:result['selection_accuracy']=result['top1']
            reports[f"{name}_{'last' if last else 'first'}"]=result
        if robust:
            from diagnose_all_certificates import measure
            metrics,_=measure(model,data)
            result={key:metrics[key] for key in ['inputs','errors','supplied_accuracy','all_supplied_correct']}
            if name=='hard_validation' or name.endswith('_pairs'):
                result.update(pairs=metrics['pairs'],both_all_supplied=metrics['both_all_supplied'],
                              selection_accuracy=metrics['both_all_supplied'])
            else:result['selection_accuracy']=metrics['all_supplied_correct']
            reports[f'{name}_all_supplied']=result
    values=[m['selection_accuracy'] for m in reports.values()]
    return reports,(min(values),float(np.mean(values)))


def constrain_input(model,mode):
    """Optional exact independence of the supplied-index features.

    The full output set depends on u,v,w. A selector may ignore its supplied
    certificate, but doing so removes potentially helpful information. The
    constraint is entirely in saved weights, so ordinary inference loading
    preserves it without an input transformation or additional oracle.
    """
    if mode=='supplied':return
    mask=torch.ones_like(model.input.weight)
    mask[:,[1,2,9,10,11,12,13,14,17]]=0
    supplied=list(getattr(model,'supplied_relation_channels',range(6,12)))
    order=torch.ones_like(model.order_bias);order[:,:,supplied]=0
    with torch.no_grad():
        model.input.weight.mul_(mask);model.order_bias.mul_(order)
    model.input.weight.register_hook(lambda grad:grad*mask)
    model.order_bias.register_hook(lambda grad:grad*order)


def check_independence(model, data):
    """Changing only the certified index must leave every score unchanged."""
    xx=torch.as_tensor(data['x'][:512],device='cuda')
    aa=torch.as_tensor(data['a'][:512],device='cuda')
    first=aa.long().argmax(1)
    last=15-aa.flip(1).long().argmax(1)
    model.eval()
    with torch.inference_mode():
        left=model(r.c.a.change_witness(xx,first))
        right=model(r.c.a.change_witness(xx,last))
    error=float((left-right).abs().max())
    assert error==0.,error
    return dict(examples=len(xx),different_indices=int((first!=last).sum()),maximum_score_error=error)


def sample_hard_groups(layout,pool,count,generator):
    if count==0:return torch.empty(0,dtype=torch.long,device='cuda')
    group=torch.multinomial(layout[pool+'_weights'],count,replacement=True,generator=generator)
    begin=layout[pool+'_offsets'][group]
    length=layout[pool+'_offsets'][group+1]-begin
    chosen=begin+(torch.rand(count,device='cuda',generator=generator)*length).long()
    return layout[pool+'_order'][chosen]


def main(args,budget):
    torch.manual_seed(r.SEED)
    path=r.HERE/f'{args.arm}.json'
    if path.exists():raise RuntimeError('Choose an unused arm name')
    initial=Path(args.initial)
    model=r.model_from(initial,full_ranks=args.full_ranks,extra_depth=args.extra_depth,
                       comparison_mode=args.comparison_mode,multiplier=args.multiplier,noise=args.initial_noise)
    if args.input_mode is None:
        args.input_mode=torch.load(initial,map_location='cpu',weights_only=True).get('configuration',{}).get('input_mode','supplied')
    constrain_input(model,args.input_mode)
    roots=None
    captured={}
    if args.auxiliary_weight:
        import auxiliary
        assert getattr(model,'comparison_mode','original')=='all_pairs'
        roots,root_spec=auxiliary.load_roots(args.training_spec)
        if not hasattr(model,'auxiliary_head'):
            model.auxiliary_head=torch.nn.Linear(model.head.in_features,3).cuda()
            torch.nn.init.zeros_(model.auxiliary_head.weight)
            torch.nn.init.zeros_(model.auxiliary_head.bias)
        def capture_hidden(module,inputs):
            if module.training:captured['hidden']=inputs[0]
        model.head.register_forward_pre_hook(capture_hidden)
    tr,eligible,offsets,weights=training_tensors(with_counts=bool(args.auxiliary_weight),training_spec=args.training_spec)
    source_layouts=training_sampling.layouts(args.training_spec,offsets)
    dual_eligible=None
    dual_count=torch.zeros((),dtype=torch.long,device='cuda')
    stabilization_counts=torch.zeros(3,dtype=torch.long,device='cuda')
    if args.duality_fraction:
        import duality
        assert args.auxiliary_weight>0
        dual_eligible,dual_spec=duality.load_eligibility(args.training_spec)
        assert len(dual_eligible)==offsets[-1]
    hard_pairs=None
    hard_layout=None
    hard_count=2*round(args.batch*args.hard_fraction/2)
    uniform_count=args.batch-hard_count
    error_count=round(args.batch*args.error_fraction)
    assert error_count<uniform_count
    error_pool=None
    if hard_count:
        hard_metadata,hard_file={
            'first':('HARD_TRAINING.json','training_pairs.npy'),
            'all_certificates':('HARD_ALL_CERTIFICATES.json','training_all_certificate_pairs.npy'),
            'complete':('HARD_COMPLETE_CERTIFICATES.json','complete_training_certificate_pairs.npy'),
        }[args.hard_source]
        if args.neighbor_hard:
            assert args.hard_source=='complete'
            hard_metadata,hard_file='NEIGHBOR_HARD_CERTIFICATES.json','neighbor_complete_training_pairs.npy'
        hard_spec=json.loads((r.HERE/hard_metadata).read_text())
        training_specification=json.loads((r.HERE/args.training_spec).read_text())
        expected_prefix=offsets[-1] if args.neighbor_hard else training_specification.get('base_rows',offsets[-1])
        assert hard_spec['training_rows']==expected_prefix and expected_prefix in offsets
        assert r.c.a.sha(r.ART/hard_file)==hard_spec['pair_sha256']
        hard_pairs=torch.as_tensor(np.load(r.ART/hard_file),device='cuda')
        assert len(hard_pairs)>0
        if args.hard_supplied=='recorded':assert hard_pairs.shape[1]==4
        if args.legacy_hard_weight is not None:
            assert args.hard_source=='complete'
            legacy_count=hard_spec['legacy_pairs']
            assert (hard_pairs[:legacy_count,:2]<offsets[1]).all()
            assert (hard_pairs[legacy_count:,:2]>=offsets[1]).all()
        if args.hard_group_balanced:
            assert args.hard_source=='complete' and args.legacy_hard_weight is not None
            group_metadata='NEIGHBOR_HARD_GROUPS.json' if args.neighbor_hard else 'HARD_GROUPS.json'
            group_file='neighbor_training_hard_groups.npz' if args.neighbor_hard else 'training_hard_groups.npz'
            group_spec=json.loads((r.HERE/group_metadata).read_text())
            assert group_spec['pair_sha256']==hard_spec['pair_sha256']
            assert group_spec['layout_sha256']==r.c.a.sha(r.ART/group_file)
            hard_layout={key:torch.as_tensor(value,device='cuda') for key,value in
                         dict(np.load(r.ART/group_file)).items()}
    validation=validation_data(args.historical_validation,args.exposed_assessment)
    generator=torch.Generator(device='cuda').manual_seed(r.SEED)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=args.precision=='float16',init_scale=1024.)
    best=(-1.,-1.)
    history=[]
    source=r.ART/f'{args.arm}_source.py'
    shutil.copyfile(Path(__file__),source)
    record=dict(configuration=vars(args),source_sha256=r.c.a.sha(source),
                initial_checkpoint_sha256=r.c.a.sha(initial),compact_source_sha256=r.c.a.sha(r.HERE/'compact.py'),
                training_specification_sha256=r.c.a.sha(r.HERE/args.training_spec),
                training_rows=offsets[-1],status='running',history=history)
    if args.robust_validation:
        evaluator=r.ART/f'{args.arm}_robust_evaluator.py'
        shutil.copyfile(r.HERE/'diagnose_all_certificates.py',evaluator)
        record['robust_evaluator_sha256']=r.c.a.sha(evaluator)
    if args.hard_group_balanced:record['hard_group_sampling']=group_spec
    model_source=r.ART/f'{args.arm}_models.py'
    shutil.copyfile(r.HERE/'models.py',model_source)
    record['model_loader_sha256']=r.c.a.sha(model_source)
    record['comparison_mode']=getattr(model,'comparison_mode','original')
    record['validation_precision']='float32'
    sampling_source=r.ART/f'{args.arm}_training_sampling.py'
    shutil.copyfile(r.HERE/'training_sampling.py',sampling_source)
    record['training_sampling_sha256']=r.c.a.sha(sampling_source)
    if args.stabilization_fraction:
        stabilization_source=r.ART/f'{args.arm}_stabilization.py'
        shutil.copyfile(r.HERE/'stabilization.py',stabilization_source)
        record['stabilization_source_sha256']=r.c.a.sha(stabilization_source)
    if args.exposed_assessment:
        record['assessment_exposure_sha256']=r.c.a.sha(r.HERE/'ASSESSMENT_EXPOSURE.json')
    if args.duality_fraction:
        dual_source=r.ART/f'{args.arm}_duality.py'
        shutil.copyfile(r.HERE/'duality.py',dual_source)
        record.update(duality_source_sha256=r.c.a.sha(dual_source),
                      duality_eligibility_sha256=r.c.a.sha(r.HERE/'POINCARE_ELIGIBILITY.json'))
    if args.auxiliary_weight:
        assert len(roots)==offsets[-1]
        auxiliary_source=r.ART/f'{args.arm}_auxiliary.py'
        shutil.copyfile(r.HERE/'auxiliary.py',auxiliary_source)
        record.update(auxiliary_source_sha256=r.c.a.sha(auxiliary_source),
                      root_target_metadata_sha256=r.c.a.sha(r.HERE/'ROOT_TARGETS.json'),
                      auxiliary_targets=['log(1+A_j)','log(1+B_j)','log(1+Q_j)'])
    if error_count:
        miner_source=r.ART/f'{args.arm}_error_miner.py'
        shutil.copyfile(r.HERE/'mine_prediction_errors.py',miner_source)
        record['error_miner_sha256']=r.c.a.sha(miner_source)
        record['error_refreshes']=[]
    start=time.monotonic()
    initial_metrics,initial_criterion=evaluate(model,validation,args.robust_validation)
    record['initial_validation']=initial_metrics
    record['initial_criterion']=initial_criterion
    if args.input_mode=='permutations':
        record['initial_independence_check']=check_independence(model,validation['S9_validation'])
    best=initial_criterion
    record.update(best_criterion=best,best_step=0)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},step=0,
                    criterion=best,configuration=vars(args)),r.ART/f'{args.arm}.pt')
    r.write(path.name,record)
    for step in range(1,args.steps+1):
        if step%100==1:budget.check()
        if args.max_run_seconds is not None and time.monotonic()-budget.start>=args.max_run_seconds:
            final=r.ART/f'{args.arm}_final.pt'
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            step=step-1,configuration=vars(args)),final)
            record.update(status='intentional_time_stop',completed_updates=step-1,
                          stop_reason='Predeclared per-run time limit reserves resources for evaluation and interventions',
                          final_checkpoint_sha256=r.c.a.sha(final))
            break
        if error_count and (step-1)%args.error_refresh_every==0:
            from mine_prediction_errors import mine
            refresh_step=step-1
            checkpoint=r.ART/f'{args.arm}_mining_{refresh_step:06d}.pt'
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            step=refresh_step,configuration=vars(args)),checkpoint)
            errors,margins,info=mine(model,tr,offsets,weights,r.SEED+9301+refresh_step,
                                     budget,args.error_mining_samples,args.error_cap,source_layouts=source_layouts)
            artifact=r.ART/f'{args.arm}_errors_{refresh_step:06d}.npz'
            np.savez_compressed(artifact,records=errors,margins=margins)
            info.update(step=refresh_step,checkpoint_sha256=r.c.a.sha(checkpoint),
                        artifact_sha256=r.c.a.sha(artifact))
            record['error_refreshes'].append(info)
            error_pool=torch.as_tensor(errors[:,:3],device='cuda') if len(errors) else None
            r.write(path.name,record)
            print(json.dumps(dict(arm=args.arm,error_refresh=info)),flush=True)
        model.train()
        ordinary_count=uniform_count-(error_count if error_pool is not None else 0)
        chunks=[]
        for source_id,weight in enumerate(weights):
            count=round(ordinary_count*weight) if source_id<len(weights)-1 else ordinary_count-sum(len(z) for z in chunks)
            chunks.append(training_sampling.sample(source_id,count,offsets,generator,source_layouts))
        idx=torch.cat(chunks)
        reverse=torch.rand(ordinary_count,device='cuda',generator=generator)<.5
        if error_pool is not None:
            selected_errors=error_pool[torch.randint(len(error_pool),(error_count,),device='cuda',generator=generator)]
            idx=torch.cat([idx,selected_errors[:,0]])
            reverse=torch.cat([reverse,selected_errors[:,1].bool()])
        if hard_count:
            if hard_layout is not None:
                old=round(hard_count//2*args.legacy_hard_weight)
                selected=hard_pairs[torch.cat([sample_hard_groups(hard_layout,'legacy',old,generator),
                                               sample_hard_groups(hard_layout,'new',hard_count//2-old,generator)])]
            elif args.legacy_hard_weight is None:
                selected=hard_pairs[torch.randint(len(hard_pairs),(hard_count//2,),device='cuda',generator=generator)]
            else:
                old=round(hard_count//2*args.legacy_hard_weight)
                selected=hard_pairs[torch.cat([
                    torch.randint(legacy_count,(old,),device='cuda',generator=generator),
                    torch.randint(legacy_count,len(hard_pairs),(hard_count//2-old,),device='cuda',generator=generator)])]
            idx=torch.cat([idx,selected[:,:2].reshape(-1)])
            reverse=torch.cat([reverse,selected[:,2].bool().repeat_interleave(2)])
        data={key:value[idx] for key,value in tr.items()}
        batch_roots=roots[idx] if roots is not None else None
        if dual_eligible is not None:
            anchor=torch.randint(2,(ordinary_count,),device='cuda',generator=generator)
            transform=(torch.rand(ordinary_count,device='cuda',generator=generator)<args.duality_fraction)
            transform&=dual_eligible[idx[:ordinary_count],anchor]
            selected_dual=transform.nonzero().flatten()
            if len(selected_dual):
                transformed,transformed_roots=duality.transform(
                    {key:value[selected_dual] for key,value in data.items()},
                    batch_roots[selected_dual],anchor[selected_dual])
                for key,value in transformed.items():data[key][selected_dual]=value
                batch_roots[selected_dual]=transformed_roots
                dual_count+=len(selected_dual)
        prefix=torch.zeros(args.batch,dtype=torch.long,device='cuda')
        if args.stabilization_fraction:
            import stabilization
            data,batch_roots,prefix,extra=stabilization.sample(data,batch_roots,args.stabilization_fraction,generator,uniform_count)
            stabilization_counts+=torch.stack([(extra>0).sum(),(prefix>0).sum(),extra.sum()])
        x,mask,y,aa=compact.encode(data,reverse)
        ii=torch.multinomial(aa.float(),1,generator=generator).squeeze(1)
        if error_pool is not None:
            ii[ordinary_count:uniform_count]=selected_errors[:,2]+prefix[ordinary_count:uniform_count]
            assert aa[torch.arange(ordinary_count,uniform_count,device='cuda'),ii[ordinary_count:uniform_count]].all()
        if hard_count and args.hard_supplied=='first':
            ii[uniform_count:]=aa[uniform_count:].long().argmax(1)
        if hard_count and args.hard_supplied=='recorded':
            ii[uniform_count:]=selected[:,3].repeat_interleave(2)+prefix[uniform_count:]
        x=r.c.a.change_witness(x,ii)
        reflection=(torch.rand(args.batch,device='cuda',generator=generator)<.5)&eligible[idx]
        if hard_count:
            both_eligible=eligible[idx[uniform_count:]].reshape(-1,2).all(1)
            reflection[uniform_count:]=(reflection[uniform_count::2]&both_eligible).repeat_interleave(2)
        if reflection.any():
            x[reflection],mask[reflection],y[reflection]=r.symmetry.reflect(x[reflection],mask[reflection],y[reflection])
        with torch.autocast('cuda',dtype=torch.float16,enabled=args.precision=='float16'):
            scores=model(x)
            auxiliary_prediction=(model.auxiliary_head(captured.pop('hidden'))
                                  if args.auxiliary_weight else None)
        scores=scores.float()
        masked=scores.masked_fill(~mask,-1e9)
        individual=masked.logsumexp(1)-masked.masked_fill(~y,-1e9).logsumexp(1)
        if hard_count:
            ranking=(individual[:uniform_count].sum()+
                     2*individual[uniform_count:].reshape(-1,2).max(1).values.sum())/args.batch
        else:ranking=individual.mean()
        bce=((F.binary_cross_entropy_with_logits(scores,y.float(),reduction='none')*mask).sum(1)/mask.sum(1)).mean()
        loss=ranking+args.bce_weight*bce
        auxiliary_loss=None
        if args.auxiliary_weight:
            auxiliary_target,available=auxiliary.targets(data,batch_roots,reverse,reflection)
            auxiliary_loss=auxiliary.loss(auxiliary_prediction.float(),auxiliary_target,available)
            loss=loss+args.auxiliary_weight*auxiliary_loss
        lr=args.min_lr+(args.lr-args.min_lr)*.5*(1+math.cos(math.pi*(step-1)/max(args.steps-1,1)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step%args.evaluate_every==0 or step==args.steps:
            if args.input_mode=='permutations':
                assert not model.input.weight[:,[1,2,9,10,11,12,13,14,17]].count_nonzero()
                supplied=list(getattr(model,'supplied_relation_channels',range(6,12)))
                assert not model.order_bias[:,:,supplied].count_nonzero()
                record['latest_independence_check']=check_independence(model,validation['S9_validation'])
            vm,criterion=evaluate(model,validation,args.robust_validation)
            item=dict(step=step,loss=float(loss.detach()),ranking_loss=float(ranking.detach()),bce=float(bce.detach()),
                      validation=vm,criterion=criterion,seconds=time.monotonic()-start)
            if auxiliary_loss is not None:item['auxiliary_loss']=float(auxiliary_loss.detach())
            item.update(gradient_scale=scaler.get_scale(),duality_examples=int(dual_count))
            item['stabilization_counts']=dict(zip(['examples','examples_with_initial_fixed_points','added_fixed_points'],stabilization_counts.tolist()))
            history.append(item)
            if criterion>best:
                best=criterion
                record.update(best_criterion=best,best_step=step)
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},step=step,
                                criterion=criterion,configuration=vars(args)),r.ART/f'{args.arm}.pt')
            # The current optimizer and RNG state permit an interrupted run to
            # be continued explicitly; previous retained scientific arms remain untouched.
            torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                            gradient_scaler=scaler.state_dict(),
                            generator_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),
                            cuda_rng_state=torch.cuda.get_rng_state_all(),configuration=vars(args)),
                       r.HERE/f'cache/{args.arm}_resume.pt')
            record['status']='completed' if step==args.steps else 'running'
            r.write(path.name,record)
            print(json.dumps(dict(arm=args.arm,**item)),flush=True)
            if args.stop_unimproved_at is not None and step>=args.stop_unimproved_at and record['best_step']==0:
                final=r.ART/f'{args.arm}_final.pt'
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                step=step,configuration=vars(args)),final)
                record.update(status='intentional_early_stop',completed_updates=step,
                    stop_reason='No scheduled evaluation improved the initial criterion by the predeclared limit',
                    final_checkpoint_sha256=r.c.a.sha(final))
                r.write(path.name,record)
                break
    record['checkpoint_sha256']=r.c.a.sha(r.ART/f'{args.arm}.pt')
    r.write(path.name,record)
    return record['status']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',default='expanded_base')
    parser.add_argument('--initial',default=str(r.ART/'uniform_views.pt'))
    parser.add_argument('--steps',type=int,default=50000)
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--lr',type=float,default=.0001)
    parser.add_argument('--min-lr',type=float,default=.00001)
    parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--bce-weight',type=float,default=.25)
    parser.add_argument('--evaluate-every',type=int,default=2500)
    parser.add_argument('--hard-fraction',type=float,default=0.)
    parser.add_argument('--hard-supplied',choices=['first','random','recorded'],default='first')
    parser.add_argument('--hard-source',choices=['first','all_certificates','complete'],default='first')
    parser.add_argument('--legacy-hard-weight',type=float,default=None)
    parser.add_argument('--hard-group-balanced',action='store_true')
    parser.add_argument('--neighbor-hard',action='store_true')
    parser.add_argument('--error-fraction',type=float,default=0.)
    parser.add_argument('--error-refresh-every',type=int,default=5000)
    parser.add_argument('--error-mining-samples',type=int,default=200000)
    parser.add_argument('--error-cap',type=int,default=16384)
    parser.add_argument('--full-ranks',action='store_const',const=True,default=None)
    parser.add_argument('--extra-depth',type=int,default=None)
    parser.add_argument('--historical-validation',action='store_true')
    parser.add_argument('--robust-validation',action='store_true')
    parser.add_argument('--input-mode',choices=['supplied','permutations'],default=None)
    parser.add_argument('--comparison-mode',choices=['original','all_pairs'],default=None)
    parser.add_argument('--auxiliary-weight',type=float,default=0.)
    parser.add_argument('--stop-unimproved-at',type=int,default=None)
    parser.add_argument('--duality-fraction',type=float,default=0.)
    parser.add_argument('--precision',choices=['float32','float16'],default='float32')
    parser.add_argument('--multiplier',type=int,default=None)
    parser.add_argument('--initial-noise',type=float,default=0.)
    parser.add_argument('--training-spec',default='TRAINING_DATA.json')
    parser.add_argument('--max-run-seconds',type=float,default=None)
    parser.add_argument('--stabilization-fraction',type=float,default=0.)
    parser.add_argument('--exposed-assessment',action='store_true')
    args=parser.parse_args()
    assert 0<=args.hard_fraction<1
    assert not args.neighbor_hard or args.hard_source=='complete' and args.hard_fraction>0
    assert 0<=args.error_fraction<1-args.hard_fraction
    assert args.error_refresh_every>0 and args.error_mining_samples>0 and args.error_cap>0
    assert args.legacy_hard_weight is None or 0<=args.legacy_hard_weight<=1
    assert args.auxiliary_weight>=0
    assert args.stop_unimproved_at is None or 1<=args.stop_unimproved_at<=args.steps
    assert 0<=args.duality_fraction<=1
    assert not args.duality_fraction or args.auxiliary_weight>0
    assert args.initial_noise>=0
    assert args.max_run_seconds is None or args.max_run_seconds>0
    assert 0<=args.stabilization_fraction<=1
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    budget=Budget(f'{args.arm}_train',gpu=True)
    status='failed_or_partial'
    try:
        status=main(args,budget)
    finally:
        print(json.dumps(budget.finish(status)),flush=True)
