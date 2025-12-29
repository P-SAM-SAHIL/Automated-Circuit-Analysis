from transformer_lens.utils import get_act_name


def safe_get_z(cache, layer: int, head: int):
    """Safely extract attention head output z from TransformerLens cache"""
    z = cache[get_act_name("z", layer)]

    if z.ndim == 4:  # [batch, seq, head, d_head]
        return z[0, :, head, :]
    elif z.ndim == 3:  # [seq, head, d_head]
        return z[:, head, :]
    else:
        raise ValueError(f"Unexpected z shape: {z.shape}")