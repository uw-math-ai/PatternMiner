"""Store bounded exact labels in signed 16-bit tensors without changing their values."""
import torch
import second_training_data as base
from second_training_data import validation

screened_offset=None
label_storage=None


def load(fresh=False,adaptive=False,screened=False):
    global screened_offset,label_storage
    values=list(base.load(fresh,adaptive,screened));screened_offset=base.screened_offset
    data=values[0];roots=values[4];metadata={}
    for name,source in [('coefficients',data['coefficients']),('roots',roots)]:
        minimum=int(source.min());maximum=int(source.max());assert -32768<=minimum<=maximum<=32767,(name,minimum,maximum)
        converted=source.to(torch.int16)
        # Check a fixed spread of complete rows, in addition to the full range check.
        take=torch.linspace(0,len(source)-1,1024,device=source.device).long()
        assert torch.equal(converted[take].int(),source[take])
        metadata[name]=dict(minimum=minimum,maximum=maximum,rows=len(source),
            before_bytes=source.numel()*source.element_size(),after_bytes=converted.numel()*converted.element_size(),
            dtype='torch.int16',sampled_roundtrip_rows=len(take))
        if name=='coefficients':data[name]=converted
        else:values[4]=converted
    label_storage=metadata
    del roots,source,converted
    torch.cuda.empty_cache()
    return tuple(values)
