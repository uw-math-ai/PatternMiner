"""Local causal forward; partial interventions recompute unselected components."""
import math
import torch
import torch.nn.functional as F


def explicit(model, x, patches=None, capture=False):
    """Replace tensors or apply a callable to the current tensor at a site.

    Shapes must agree. Patched patterns must remain probability distributions
    on nonpadding destination keys. Downstream operations are recomputed.
    Captures contain the tensors after any intervention. Evaluation only.
    """
    if model.training:
        raise ValueError('Circuit inference requires model.eval()')
    patches = {} if patches is None else patches
    cache = {}
    used = set()

    def site(name, value):
        if name in patches:
            replacement = patches[name]
            if callable(replacement):
                replacement = replacement(value)
            if replacement.shape != value.shape:
                raise ValueError(f'{name}: shape mismatch')
            value = replacement
            used.add(name)
        if capture:
            cache[name] = value
        return value

    padding = x.abs().sum(-1) == 0
    features = site('features', model.feature_map(x))
    relations = site('order_comparisons', model.comparisons(x)) if hasattr(model, 'order_bias') else None
    h = site('embedding', model.input(features))
    for number, layer in enumerate(model.encoder.layers):
        prefix = f'L{number}'
        h = site(f'{prefix}.resid_pre', h)
        batch, tokens, dim = h.shape
        heads = layer.self_attn.num_heads
        dh = dim//heads
        attn_input = site(f'{prefix}.attention_input', layer.norm1(h) if layer.norm_first else h)
        qkv = F.linear(attn_input, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = qkv.reshape(batch, tokens, 3, heads, dh).permute(2, 0, 3, 1, 4)
        q = site(f'{prefix}.q', q)
        k = site(f'{prefix}.k', k)
        v = site(f'{prefix}.v', v)
        logits = (q @ k.transpose(-2, -1))/math.sqrt(dh)
        if relations is not None:
            bias = site(f'{prefix}.order_bias', torch.einsum('bqkr,hr->bhqk', relations, model.order_bias[number]))
            logits = logits+bias
        logits = site(f'{prefix}.attention_logits', logits)
        pattern = logits.masked_fill(padding[:, None, None, :], -torch.inf).softmax(-1)
        pattern = site(f'{prefix}.pattern', pattern)
        if f'{prefix}.pattern' in patches:
            if not torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=2e-6, rtol=0):
                raise ValueError('A patched pattern must have unit row sums')
            if (pattern < 0).any() or (pattern.masked_select(padding[:, None, None, :].expand_as(pattern)).abs() > 1e-7).any():
                raise ValueError('A patched pattern violates the destination key mask')
        head_outputs = site(f'{prefix}.head_outputs', pattern @ v)
        transported = head_outputs.transpose(1, 2).reshape(batch, tokens, dim)
        output = site(f'{prefix}.attention_output', layer.self_attn.out_proj(transported))
        h = site(f'{prefix}.resid_after_attention', h+output if layer.norm_first else layer.norm1(h+output))
        mlp_input = site(f'{prefix}.mlp_input', layer.norm2(h) if layer.norm_first else h)
        # This is the fused activation used by native PyTorch 2.6 inference.
        if relations is not None:
            hidden = F.gelu(layer.linear1(mlp_input), approximate='tanh')
        else:
            hidden = torch._addmm_activation(layer.linear1.bias,
                        mlp_input.reshape(batch*tokens, dim), layer.linear1.weight.t(), use_gelu=True).reshape(batch, tokens, -1)
        hidden = site(f'{prefix}.mlp_hidden', hidden)
        mlp = site(f'{prefix}.mlp_output', layer.linear2(hidden))
        h = site(f'{prefix}.resid_post', h+mlp if layer.norm_first else layer.norm2(h+mlp))
    scores = site('scores', model.head(h).squeeze(-1))
    if used != set(patches):
        raise ValueError(f'Unknown intervention sites: {sorted(set(patches)-used)}')
    return (scores, cache) if capture else scores

