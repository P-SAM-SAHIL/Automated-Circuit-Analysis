from typing import Dict
from transformer_lens.utils import get_act_name
from transformer_lens import HookedTransformer

from utils.cache_utils import safe_get_z

def patch_head_delta(pair: Dict, layer: int, head: int, model: HookedTransformer) -> float:
    """Patch a single attention head's z activation from clean → corrupt run"""
    clean_cache = pair["clean_cache"]
    corrupt_tokens = pair["corrupt_tokens"]
    corrupt_loss = pair["corrupt_loss"]

    def hook_patch(z, hook):
        z = z.clone()
        clean_z = safe_get_z(clean_cache, layer, head)

        if z.ndim == 4:
            z[0, :, head, :] = clean_z
        elif z.ndim == 3:
            z[:, head, :] = clean_z
        else:
            raise ValueError(f"Unexpected z shape: {z.shape}")

        return z

    patched_loss = model.run_with_hooks(
        corrupt_tokens,
        return_type="loss",
        fwd_hooks=[(get_act_name("z", layer), hook_patch)]
    ).item()

    return patched_loss - corrupt_loss
