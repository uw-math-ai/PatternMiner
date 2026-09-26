"""Audit and quotient an exactly duplicated residual representation."""
import argparse
import json
from pathlib import Path
import torch
import session as s
from full_resources import Budget


def compress(state):
    width=state['input.base.weight'].shape[0];d=width//2;m=width
    assert width==384
    result={};differences={}
    def equal(name,a,b):differences[name]=float((a-b).abs().max())
    for name,value in state.items():
        if name in ['base_order_bias','root_action_order_bias','order_bias']:
            equal(name,value[:,:8],value[:,8:]);result[name]=value[:,:8].clone()
        elif name.startswith('input.'):
            equal(name,value[:d],value[d:]);result[name]=value[:d].clone()
        elif name in ['head.weight','auxiliary_head.weight']:
            equal(name,value[:,:d],value[:,d:]);result[name]=value[:,:d]+value[:,d:]
        elif name in ['head.bias','auxiliary_head.bias']:result[name]=value.clone()
        elif name.endswith('in_proj_weight'):
            v=value.reshape(3,width,width)
            equal(name+'_rows',v[:,:d],v[:,d:]);equal(name+'_columns',v[:,:,:d],v[:,:,d:])
            result[name]=(v[:,:d,:d]+v[:,:d,d:]).reshape(3*d,d)
        elif name.endswith('in_proj_bias'):
            v=value.reshape(3,width);equal(name,v[:,:d],v[:,d:]);result[name]=v[:,:d].reshape(3*d).clone()
        elif name.startswith('encoder.layers.') and value.ndim==2:
            rows=m if name.endswith('linear1.weight') else d
            columns=m if name.endswith('linear2.weight') else d
            equal(name+'_rows',value[:rows],value[rows:]);equal(name+'_columns',value[:,:columns],value[:,columns:])
            result[name]=value[:rows,:columns]+value[:rows,columns:]
        elif name.startswith('encoder.layers.') and value.ndim==1:
            rows=m if name.endswith('linear1.bias') else d
            equal(name,value[:rows],value[rows:]);result[name]=value[:rows].clone()
        else:raise ValueError(name)
    return result,differences


def main(args):
    path=Path(args.checkpoint);saved=torch.load(path,map_location='cpu',weights_only=True)
    result,differences=compress(saved['state_dict']);exact=max(differences.values())==0.
    report=dict(status='completed',exact_duplication=exact,checkpoint_sha256=s.sha(path),source_sha256=s.sha(Path(__file__)),
        maximum_parameter_discrepancy=max(differences.values()),differences=differences,
        original_parameters=sum(v.numel() for v in saved['state_dict'].values()),compressed_parameters=sum(v.numel() for v in result.values()),
        rule='Take one copy of each repeated output coordinate and sum the two repeated input-coordinate blocks; preserve one of each repeated attention head.',
        interpretation='A quotient by exact duplicated coordinates preserves the real-arithmetic function. Floating-point outputs and every declared accuracy criterion still require verification before using the compressed model.')
    if exact:
        output=s.ART/(args.name+'.pt');assert not output.exists()
        saved=dict(saved,state_dict=result,compression=dict(factor=2,source_sha256=s.sha(path)))
        torch.save(saved,output);report['compressed_checkpoint_sha256']=s.sha(output)
    s.write(args.name+'.json',report);print(json.dumps({k:v for k,v in report.items() if k!='differences'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--name',required=True);args=parser.parse_args()
    torch.set_num_threads(1);budget=Budget(args.name.lower(),60);status='failed_or_partial'
    try:main(args);status='completed'
    finally:print(json.dumps(budget.finish(status)),flush=True)
