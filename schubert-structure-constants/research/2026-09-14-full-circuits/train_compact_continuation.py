"""Continue exact supervision while retaining supported predictions of the earlier model."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
import session as s
from session import c
import richardson_model as rm
import compact
import auxiliary
import stabilization
import training_sampling
import compact_training_data as training_data
import stability_regularizer
import error_sampling
from diagnose_all_certificates import measure
from full_resources import Budget

SPEC='FINAL_NEIGHBOR_BALANCED_TRAINING_DATA.json'


def evaluate(model, datasets):
    model.eval();reports={}
    for name,data in datasets.items():
        report,_=measure(model,data)
        report['selection_accuracy']=report['both_all_supplied'] if name=='hard_validation' or name.endswith('_pairs') else report['all_supplied_correct']
        reports[name]=report
    values=[x['selection_accuracy'] for x in reports.values()]
    return reports,(min(values),float(np.mean(values)))


def main(args,budget):
    torch.manual_seed(c.SEED)
    initial=Path(args.initial)
    model=rm.load(initial)
    teacher=rm.load(Path(args.teacher)).eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    if args.rank_only:rm.use_rank_only(model)
    if args.relative_roots:rm.use_relative_roots(model)
    if args.supplied_independent:
        from supplied_independent_model import use
        use(model)
    if args.root_action:
        from root_action_model import use
        use(model)
    captured={}
    model.head.register_forward_pre_hook(lambda module,inputs:captured.update(hidden=inputs[0]) if module.training else None)
    if args.paired_views:
        model.encoder.layers[2].norm1.register_forward_pre_hook(lambda module,inputs:captured.update(middle=inputs[0]) if module.training else None)
    tr,eligible,offsets,weights,roots,layouts,hard,hard_layout,fresh_hard,fresh_layout=training_data.load(args.fresh,args.adaptive,args.screened)
    error_path=c.OLD/'artifacts/neighbor_hard_continuation_errors_007500.npz'
    errors=torch.as_tensor(np.load(error_path)['records'][:,:3],device='cuda')
    error_layout,error_balance=error_sampling.layout(errors,tr,offsets,weights)
    confirmed=None
    if args.screened:
        path=c.ART/'screened_exact_errors.npy'
        info=json.loads((c.HERE/'SCREENED_PREPARED.json').read_text());assert c.sha(path)==info['error_sha256']
        values=np.load(path).copy();values[:,0]+=training_data.screened_offset
        if len(values):confirmed=torch.as_tensor(values[:,:3],device='cuda')
    validation=training_data.validation(args.fresh)
    if args.exposed_assessment:
        from exposed_assessment import append_validation
        validation=append_validation(validation)
    generator=torch.Generator(device='cuda').manual_seed(c.SEED)
    extra=list(model.input.extra.parameters())
    if args.freeze_extra:
        for parameter in extra:parameter.requires_grad_(False)
    root_action=model.root_action_order_bias if getattr(model,'root_action_enabled',False) else None
    base=[p for p in model.parameters() if all(p is not q for q in extra) and p is not root_action]
    parameter_groups=[dict(params=base,lr=args.base_lr),dict(params=extra,lr=args.extra_lr)]
    if root_action is not None:parameter_groups.append(dict(params=[root_action],lr=args.root_action_lr))
    optimizer=torch.optim.AdamW(parameter_groups,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    parameters=list(model.parameters())
    averaged=[parameter.detach().clone() for parameter in parameters]
    source=s.ART/f'{args.arm}_source.py';shutil.copyfile(__file__,source)
    report=dict(configuration=vars(args),status='running',label_storage=training_data.label_storage,training_rows=offsets[-1],hard_pairs=len(hard)+(len(fresh_hard) if fresh_hard is not None else 0),
                initial_sha256=c.sha(initial),teacher_sha256=c.sha(Path(args.teacher)),
                teacher_regularization=dict(weight=args.teacher_weight,temperature=2.,scope='Full candidate distribution only on training examples where the teacher selects an exactly valid target; no development labels enter the loss'),source_sha256=c.sha(source),error_pool_sha256=c.sha(error_path),
                training_specification_sha256=c.sha(c.OLD/SPEC),history=[],error_refreshes=[],
                fresh_data_sha256=c.sha(c.HERE/'DATA.json') if args.fresh else None,
                adaptive_data_sha256=c.sha(c.HERE/'ADAPTIVE_DATA.json') if args.adaptive else None,
                screened_data_sha256=c.sha(c.HERE/'SCREENED_DATA.json') if args.screened else None,
                confirmed_screening_errors=0 if confirmed is None else len(confirmed),
                late_data_sha256=c.sha(c.HERE/'LATE_DATA.json'),late_prepared_sha256=c.sha(c.HERE/'LATE_PREPARED.json'),
                model_feature_kind=getattr(model,'extra_kind','richardson'),rank_only=getattr(model,'rank_only',False),
                supplied_index_independent=getattr(model,'supplied_index_independent',False),
                root_action_enabled=getattr(model,'root_action_enabled',False),
                exposed_assessment_pairs_sha256=c.sha(c.HERE/'EXPOSED_ASSESSMENT_PAIRS.json') if args.exposed_assessment else None,
                late_sampling=dict(ordinary_share=.08,share_of_fresh_hard_pool=.125),
                neighbor_sampling=dict(ordinary_share=.2,share_of_fresh_hard_pool=.5),
                neighbor_data_sha256=s.sha(s.HERE/'NEIGHBOR_DATA.json'),
                second_neighbor_data_sha256=s.sha(s.HERE/'SECOND_NEIGHBOR_DATA.json'),
                second_neighbor_prepared_sha256=s.sha(s.HERE/'SECOND_NEIGHBOR_PREPARED.json'),
                first_assessment_development_sha256=s.sha(s.HERE/'FIRST_ASSESSMENT_DEVELOPMENT.json'),
                neighbor_prepared_sha256=s.sha(s.HERE/'NEIGHBOR_PREPARED.json'),
                new_development_sha256=s.sha(s.HERE/'PREPARED.json'),
                initial_error_balance=error_balance,
                moving_average=dict(decay=.999,initialization='Initial model parameters',
                    update='After each attempted optimizer update; unchanged parameters on skipped updates',
                    selection='At each scheduled evaluation compare raw and averaged parameters by the same minimum-then-mean development criterion; save a single network'),
                validation_role='Development only, including the previously exposed assessment pairs',
                augmentation=dict(reflection=.5,stabilization=.5,poincare=0),
                sampling=dict(ordinary=.55,retained_training_errors=.20,hard_pairs=.25,hard_legacy_share=.25))
    for file in ['richardson_model.py','richardson_features.py','relative_root_features.py','supplied_independent_model.py','root_action_model.py','training_data.py','final_training_data.py','error_sampling.py','exposed_assessment.py','stability_regularizer.py','adaptive_training.py']:
        dst=s.ART/f'{args.arm}_{file}';shutil.copyfile(c.HERE/file,dst);report[file+'_sha256']=c.sha(dst)
    assert len(validation)==27
    for file in ['compact_training_data.py','second_training_data.py','neighbor_training_data.py','session.py']:
        dst=s.ART/f'{args.arm}_{file}';shutil.copyfile(s.HERE/file,dst);report[file+'_sha256']=s.sha(dst)
    metrics,best=evaluate(model,validation)
    report.update(initial_validation=metrics,best_criterion=best,best_step=0,best_variant='initial')
    rm.save(model,s.ART/f'{args.arm}.pt',step=0,criterion=best,configuration=vars(args))
    s.write(args.arm+'.json',report)
    print(json.dumps(dict(arm=args.arm,initial_criterion=best,elapsed=time.monotonic()-budget.start)),flush=True)
    batch=512;hard_count=128;error_count=102;ordinary=282;uniform=384
    status='completed';step=0
    for step in range(1,args.steps+1):
        if step%25==1:
            budget.check()
            if time.monotonic()-budget.start>args.wall_limit-150:
                status='intentional_time_stop';step-=1;break
        if args.error_refresh_every and (step-1)%args.error_refresh_every==0:
            from mine_prediction_errors import mine
            mined,margins,info=mine(model,tr,offsets,weights,c.SEED+9301+step-1,budget,args.mining_samples,16384,source_layouts=layouts)
            if len(mined):
                errors=torch.as_tensor(mined[:,:3],device='cuda')
                error_layout,error_balance=error_sampling.layout(errors,tr,offsets,weights)
            info['error_balance']=error_balance
            error_artifact=s.ART/f'{args.arm}_errors_{step-1:06d}.npz'
            np.savez_compressed(error_artifact,records=mined,margins=margins)
            info.update(step=step-1,sha256=c.sha(error_artifact));report['error_refreshes'].append(info)
            s.write(args.arm+'.json',report);print(json.dumps(dict(arm=args.arm,error_refresh=info)),flush=True)
        model.train();chunks=[]
        for source_id,weight in enumerate(weights):
            count=round(ordinary*weight) if source_id<len(weights)-1 else ordinary-sum(len(x) for x in chunks)
            chunks.append(training_sampling.sample(source_id,count,offsets,generator,layouts))
        idx=torch.cat(chunks);reverse=torch.rand(ordinary,device='cuda',generator=generator)<.5
        mandatory=error_count//2 if confirmed is not None else 0
        ee=errors[c.previous_training.sample_hard_groups(error_layout,'new',error_count,generator)]
        if mandatory:
            pinned=confirmed[torch.randint(len(confirmed),(mandatory,),device='cuda',generator=generator)]
            if args.pinned_errors:ee[:mandatory]=pinned
        selected=hard[torch.cat([c.previous_training.sample_hard_groups(hard_layout,'legacy',16,generator),
                                c.previous_training.sample_hard_groups(hard_layout,'new',24 if args.fresh else 48,generator)])]
        if args.fresh:
            selected=torch.cat([selected,fresh_hard[c.previous_training.sample_hard_groups(fresh_layout,'new',24,generator)]])
        idx=torch.cat([idx,ee[:,0],selected[:,:2].reshape(-1)])
        reverse=torch.cat([reverse,ee[:,1].bool(),selected[:,2].bool().repeat_interleave(2)])
        data={k:v[idx] for k,v in tr.items()};qq=roots[idx];original_data=data
        data,qq,prefix,extra_points=stabilization.sample(data,qq,.5,generator,uniform)
        x,mask,y,aa=compact.encode(data,reverse)
        ii=torch.multinomial(aa.float(),1,generator=generator).squeeze(1)
        ii[ordinary:uniform]=ee[:,2]+prefix[ordinary:uniform]
        if args.hard_supplied!='random':
            common=selected[:,3].repeat_interleave(2)+prefix[uniform:]
            choose=torch.ones(len(selected),device='cuda',dtype=torch.bool) if args.hard_supplied=='recorded' else torch.rand(len(selected),device='cuda',generator=generator)<.5
            ii[uniform:]=torch.where(choose.repeat_interleave(2),common,ii[uniform:])
            assert aa[torch.arange(uniform,batch,device='cuda'),ii[uniform:]].all()
        x=c.prior.c.a.change_witness(x,ii)
        reflection=(torch.rand(batch,device='cuda',generator=generator)<.5)&eligible[idx]
        reflection[uniform:]=(reflection[uniform::2]&eligible[idx[uniform:]].reshape(-1,2).all(1)).repeat_interleave(2)
        if reflection.any():x[reflection],mask[reflection],y[reflection]=c.prior.symmetry.reflect(x[reflection],mask[reflection],y[reflection])
        with torch.autocast('cuda',dtype=torch.float16):
            scores=model(x);aux=model.auxiliary_head(captured.pop('hidden'))
            consistency=paired_ranking=None
            if args.paired_views:
                augmented_hidden=captured.pop('middle')
                pool=(extra_points>0).nonzero().flatten()
                selected=pool[torch.randperm(len(pool),device='cuda',generator=generator)[:args.paired_views]]
                if len(selected):
                    bx,bm,by,alignment,real=stability_regularizer.original_view(original_data,reverse,ii,prefix,extra_points,reflection,selected)
                    bs=model(bx).float().masked_fill(~bm,-1e9);bh=captured.pop('middle');captured.pop('hidden')
                    consistency=stability_regularizer.aligned_loss(augmented_hidden[selected],bh,alignment,real)
                    paired_ranking=(bs.logsumexp(1)-bs.masked_fill(~by,-1e9).logsumexp(1)).mean()
        scores=scores.float();masked=scores.masked_fill(~mask,-1e9)
        individual=masked.logsumexp(1)-masked.masked_fill(~y,-1e9).logsumexp(1)
        ranking=(individual[:uniform].sum()+2*individual[uniform:].reshape(-1,2).max(1).values.sum())/batch
        bce=((F.binary_cross_entropy_with_logits(scores,y.float(),reduction='none')*mask).sum(1)/mask.sum(1)).mean()
        target,available=auxiliary.targets(data,qq,reverse,reflection)
        auxloss=auxiliary.loss(aux.float(),target,available)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            teacher_scores=torch.cat([teacher(x[start:start+256]) for start in range(0,len(x),256)]).float().masked_fill(~mask,-1e9)
        teacher_good=y[torch.arange(batch,device='cuda'),teacher_scores.argmax(1)]
        distillation=masked.new_zeros(())
        if teacher_good.any():
            distillation=4*F.kl_div((masked[teacher_good]/2).log_softmax(1),(teacher_scores[teacher_good]/2).softmax(1),reduction='batchmean')
        loss=ranking+.25*bce+.1*auxloss+args.teacher_weight*distillation
        if args.paired_views and consistency is not None:
            loss=loss+.25*paired_ranking+args.consistency_weight*consistency
        schedule=.2+.8*.5*(1+math.cos(math.pi*(step-1)/max(1,args.steps-1)))
        optimizer.param_groups[0]['lr']=args.base_lr*schedule
        optimizer.param_groups[1]['lr']=args.extra_lr*schedule
        if root_action is not None:optimizer.param_groups[2]['lr']=args.root_action_lr*schedule
        optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward()
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer);scaler.update()
        with torch.no_grad():
            torch._foreach_mul_(averaged,.999)
            torch._foreach_add_(averaged,parameters,alpha=.001)
        if step%100==0:
            print(json.dumps(dict(arm=args.arm,step=step,loss=float(loss.detach()),elapsed=time.monotonic()-budget.start)),flush=True)
        if step%args.evaluate_every==0 or step==args.steps:
            metrics,criterion=evaluate(model,validation)
            raw_metrics,raw_criterion=metrics,criterion
            raw_parameters=[parameter.detach().clone() for parameter in parameters]
            with torch.no_grad():
                for parameter,average in zip(parameters,averaged):parameter.copy_(average)
            average_metrics,average_criterion=evaluate(model,validation)
            variant='moving_average' if average_criterion>raw_criterion else 'raw'
            if variant=='moving_average':metrics,criterion=average_metrics,average_criterion
            else:
                with torch.no_grad():
                    for parameter,raw in zip(parameters,raw_parameters):parameter.copy_(raw)
            item=dict(step=step,loss=float(loss.detach()),ranking=float(ranking.detach()),auxiliary=float(auxloss.detach()),
                      teacher_distillation=float(distillation.detach()),teacher_training_correct_fraction=float(teacher_good.float().mean()),gradient_norm=float(norm),gradient_scale=scaler.get_scale(),validation=metrics,criterion=criterion,elapsed=time.monotonic()-budget.start)
            item.update(selected_variant=variant,raw_validation=raw_metrics,raw_criterion=raw_criterion,
                        moving_average_validation=average_metrics,moving_average_criterion=average_criterion)
            if args.paired_views and consistency is not None:item.update(consistency_loss=float(consistency.detach()),paired_ranking=float(paired_ranking.detach()))
            report['history'].append(item)
            if criterion>best:
                best=criterion;report.update(best_criterion=best,best_step=step,best_variant=variant)
                rm.save(model,s.ART/f'{args.arm}.pt',step=step,criterion=best,configuration=vars(args),parameter_variant=variant)
            with torch.no_grad():
                for parameter,raw in zip(parameters,raw_parameters):parameter.copy_(raw)
            s.write(args.arm+'.json',report)
            print(json.dumps(dict(arm=args.arm,step=step,criterion=criterion,populations={k:v['selection_accuracy'] for k,v in metrics.items()})),flush=True)
    rm.save(model,s.ART/f'{args.arm}_final.pt',step=step,configuration=vars(args))
    torch.save(dict(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),generator=generator.get_state()),s.HERE/f'cache/{args.arm}_optimizer.pt')
    torch.save(dict(parameters=averaged,decay=.999,completed_updates=step),s.HERE/f'cache/{args.arm}_moving_average.pt')
    report.update(status=status,completed_updates=step,checkpoint_sha256=c.sha(s.ART/f'{args.arm}.pt'),final_checkpoint_sha256=c.sha(s.ART/f'{args.arm}_final.pt'))
    s.write(args.arm+'.json',report);return status


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True)
    parser.add_argument('--teacher',required=True);parser.add_argument('--teacher-weight',type=float,default=.5)
    parser.add_argument('--initial',default=str(c.OLD/'artifacts/neighbor_hard_continuation.pt'))
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--evaluate-every',type=int,default=500)
    parser.add_argument('--base-lr',type=float,default=5e-6);parser.add_argument('--extra-lr',type=float,default=2e-5)
    parser.add_argument('--wall-limit',type=int,default=1500)
    parser.add_argument('--freeze-extra',action='store_true')
    parser.add_argument('--rank-only',action='store_true',help='Omit an identically zero, frozen additional projection')
    parser.add_argument('--relative-roots',action='store_true',help='Add the 24 input-derived relative-root coordinates')
    parser.add_argument('--supplied-independent',action='store_true',help='Discard all coordinates and comparisons depending on the supplied index')
    parser.add_argument('--exposed-assessment',action='store_true',help='Include failed assessment pairs only in development evaluation')
    parser.add_argument('--root-action',action='store_true',help='Add the six relative Weyl-group action matrices to attention')
    parser.add_argument('--root-action-lr',type=float,default=1e-3)
    parser.add_argument('--pinned-errors',action='store_true',help='Reserve half the error rows for the historical 71 screened errors')
    parser.add_argument('--hard-supplied',choices=['random','recorded','mixed'],default='random')
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--adaptive',action='store_true')
    parser.add_argument('--screened',action='store_true')
    parser.add_argument('--error-refresh-every',type=int,default=0)
    parser.add_argument('--mining-samples',type=int,default=100000)
    parser.add_argument('--paired-views',type=int,default=0)
    parser.add_argument('--consistency-weight',type=float,default=0.)
    args=parser.parse_args();assert not (s.HERE/(args.arm+'.json')).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    budget=Budget(args.arm+'_training',args.wall_limit,gpu=True);status='failed_or_partial'
    try:status=main(args,budget)
    finally:print(json.dumps(budget.finish(status)),flush=True)
