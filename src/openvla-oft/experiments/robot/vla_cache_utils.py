import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.util import view_as_blocks


@torch.no_grad()
def get_layer_mask_schedule(multihead_attention, apply_weighted_growth=True, growth_factor=0.55):
    """
    Computes per-layer reuse proportions based on normalized attention entropy.

    Args:
        multihead_attention (List[Tensor]): Attention maps per layer (shape: [1, heads, tokens, tokens]).
        apply_weighted_growth (bool): Whether to smooth upward deltas.
        growth_factor (float): Weight for smoothing.

    Returns:
        torch.Tensor: Layer-wise reuse proportions, shape (num_layers - 1,).
    """
    device = multihead_attention[0].device
    entropies = []
    
    for attn in multihead_attention[:-1]:
        attn = attn.mean(dim=1)[0]
        attn /= attn.sum(dim=-1, keepdim=True) + 1e-10
        attn = torch.nan_to_num(attn, nan=0.0)
        token_entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)
        entropies.append(token_entropy.mean())

    entropies = torch.stack(entropies)
    norm_entropy = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
    reuse = 1.0 - norm_entropy

    if apply_weighted_growth:
        reuse = reuse.tolist()
        for i in range(1, len(reuse)):
            delta = reuse[i] - reuse[i - 1]
            if delta > 0:
                reuse[i] = reuse[i - 1] + delta * growth_factor
        reuse = torch.tensor(reuse, dtype=torch.float32, device=device)

    return reuse

def patchify(image, patch_size=14):
    """
    Converts an image into non-overlapping patches.
    """
    image = np.array(image)
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0, "Image dimensions must be divisible by patch size."

    if image.ndim == 3:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size, image.shape[2]))
    else:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size))

    patches = blocks.reshape(-1, patch_size, patch_size, image.shape[2]) if image.ndim == 3 else blocks.reshape(-1, patch_size, patch_size)
    return patches

def calculate_patch_similarity(patches1, patches2):
    """
    Computes cosine similarity between two sets of patches.
    """
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)
    
    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)
    
    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)
    return cosine_sim

def find_static_patches(img_0, img_1, patch_size=14, top_k=150, sim_threshold=0.996):
    """
    Identifies significant patches with high similarity across two images.
    """
    patches1 = patchify(img_0, patch_size)
    patches2 = patchify(img_1, patch_size)

    similarity = calculate_patch_similarity(patches1, patches2)
    grid_size = 224 // patch_size
    similarity_2d = similarity.reshape(grid_size, grid_size)

    patch_scores = [(i * grid_size + j, similarity_2d[i, j])
                    for i in range(grid_size) for j in range(grid_size)
                    if similarity_2d[i, j] >= sim_threshold]

    patch_scores.sort(key=lambda x: x[1], reverse=True)
    top_patch_ids = [idx for idx, _ in patch_scores[:top_k]]
    return top_patch_ids

@torch.no_grad()
def token_attention_merge(multihead_attention, layer_id=15, primary=True):
    """
    Computes mean attention from text tokens to vision tokens.
    """
    attn_map = multihead_attention[layer_id].to(torch.float32).squeeze(0).mean(dim=0)

    v_token_start = 1 if primary else 257
    v_token_end = v_token_start + 256
    t_token_start = 513
    t_token_end = t_token_start + 34

    attention_pos = multihead_attention[-1]
    text_mask = (attention_pos >= t_token_start) & (attention_pos < t_token_end)

    relation = attn_map[text_mask, v_token_start:v_token_end]
    return relation.mean(dim=0).cpu()

def get_top_attention_patches(attn_scores, top_k=120):
    """
    Selects top-k patch indices based on attention scores.
    """
    attn_scores = attn_scores.cpu().numpy() if isinstance(attn_scores, torch.Tensor) else attn_scores
    attn = attn_scores.reshape(16, 16)
    attn_resized = cv2.resize(attn, (16, 16))

    flat = [(i * 16 + j, attn_resized[i, j]) for i in range(16) for j in range(16)]
    flat.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in flat[:top_k]]

def draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4):
    """
    Draws colored overlays on image for different patch groups.
    """
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width = image.size[0]
    num_patches = width // patch_size

    for patch_list, color in patch_groups:
        for pid in patch_list:
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            draw.rectangle([top_left, bottom_right], fill=color + (int(255 * alpha),))

    return Image.alpha_composite(image, overlay).convert("RGB")

def visualize_significant_patches_mask(image, patch_ids, patch_size=14, alpha=0.5, color=(255, 255, 255)):
    """
    Highlights specified patches with semi-transparent overlay.
    """
    overlay_group = [(patch_ids, color)]
    return draw_patches_overlay(image, overlay_group, patch_size, alpha)

def task_relevant_selection(multihead_attention, image, significant_patches, primary=True, top_k=100):
    """
    Highlights and compares significant patches with top attention patches.
    """
    attn_score = token_attention_merge(multihead_attention, primary=primary)
    top_patches = get_top_attention_patches(attn_score, top_k)

    only_significant = set(significant_patches) - set(top_patches)
    only_top = set(top_patches) - set(significant_patches)
    overlap = set(significant_patches) & set(top_patches)

    patch_groups = [
        (significant_patches, (15, 67, 223)),
        (top_patches, (254, 55, 13)),
        (only_significant, (40, 116, 166)),
        (only_top, (241, 196, 15)),
        (overlap, (231, 76, 60)),
    ]

    result_image = draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4)

    v_token_start = 1 if primary else 257
    remaining = sorted([pid + v_token_start for pid in only_significant])

    return np.array(result_image), remaining
