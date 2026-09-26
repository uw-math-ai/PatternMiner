"""Sample training rows, with optional equal mass for each parent family."""
import json
import torch
import common as r


def layouts(training_spec, offsets):
    specification = json.loads((r.HERE / training_spec).read_text())
    result = []
    for source, entry in enumerate(specification['sources'].values()):
        sampling = entry.get('sampling')
        if sampling is None:
            result.append(None)
            continue
        assert sampling['kind'] == 'uniform_parent_then_factor_then_row'
        plan_path = r.HERE / sampling['plan']
        data_path = r.HERE / sampling['data']
        assert r.c.a.sha(plan_path) == sampling['plan_sha256']
        assert r.c.a.sha(data_path) == sampling['data_sha256']
        plan = json.loads(plan_path.read_text())
        data = json.loads(data_path.read_text())
        assert data['plan_sha256'] == sampling['plan_sha256']
        assert data['rows'] == entry['rows']
        assert len(data['offsets']) == len(plan['pairs']) + 1
        groups = [[] for _ in plan['anchors']]
        for group, pair in enumerate(plan['pairs']):
            if data['offsets'][group + 1] > data['offsets'][group]:
                groups[pair['parent']].append(group)
        assert all(groups), 'Every parent must have a nonempty neighboring product'
        parent_offsets = [0]
        for group in groups:
            parent_offsets.append(parent_offsets[-1] + len(group))
        result.append({
            'parents': torch.tensor(parent_offsets, device='cuda'),
            'groups': torch.tensor([g for group in groups for g in group], device='cuda'),
            'rows': torch.tensor(data['offsets'], device='cuda') + offsets[source],
        })
    return result


def sample(source, count, offsets, generator, source_layouts=None):
    layout = None if source_layouts is None else source_layouts[source]
    if layout is None:
        return torch.randint(offsets[source], offsets[source + 1], (count,),
                             device='cuda', generator=generator)
    parent = torch.randint(len(layout['parents']) - 1, (count,), device='cuda', generator=generator)
    begin = layout['parents'][parent]
    length = layout['parents'][parent + 1] - begin
    group_index = begin + (torch.rand(count, device='cuda', generator=generator) * length).long()
    group = layout['groups'][group_index]
    begin = layout['rows'][group]
    length = layout['rows'][group + 1] - begin
    return begin + (torch.rand(count, device='cuda', generator=generator) * length).long()
