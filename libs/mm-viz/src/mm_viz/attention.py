"""Attention weight extraction and visualization for GPT models."""

from __future__ import annotations

import math
from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def extract_attention_weights(
    model: torch.nn.Module,
    input_ids: Tensor,  # (1, seq_len)
    device: torch.device,
) -> list[Tensor]:
    """Extract attention weights from all layers.

    Returns list of (n_heads, seq_len, seq_len) tensors, one per layer.

    Hooks into each CausalSelfAttention module, grabs Q and K after projection,
    and computes attention = softmax(Q @ K^T / sqrt(d_k)) with a causal mask.
    Does not modify the model's forward pass.
    """
    from mm_model.model import CausalSelfAttention

    model = model.to(device)
    input_ids = input_ids.to(device)
    model.eval()

    attention_maps: list[list[Tensor]] = []
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(layer_store: list[Tensor]) -> Callable[[Any, tuple[Any, ...], Any], None]:
        def hook_fn(module: Any, args: tuple[Any, ...], output: Any) -> None:
            # Re-compute Q, K from the input to this module
            x = args[0]
            b, t, c = x.size()
            qkv = module.qkv(x)
            q, k, _v = qkv.split(c, dim=2)

            n_heads = module.n_heads
            head_dim = module.head_dim

            q = q.view(b, t, n_heads, head_dim).transpose(1, 2)  # (b, n_heads, t, head_dim)
            k = k.view(b, t, n_heads, head_dim).transpose(1, 2)

            # Compute attention scores: (b, n_heads, t, t)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

            # Apply causal mask
            causal_mask = torch.triu(torch.ones(t, t, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)

            # Store (n_heads, seq_len, seq_len) — squeeze batch dim
            layer_store.append(attn_weights[0].detach().cpu())

        return hook_fn

    # Find all CausalSelfAttention modules and register hooks
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            store: list[Tensor] = []
            attention_maps.append(store)
            hook = module.register_forward_hook(_make_hook(store))
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Flatten: each store is a list with one tensor
    return [store[0] for store in attention_maps]  # type: ignore[index]


def _weight_to_color(weight: float, colormap: str = "blue") -> str:
    """Convert an attention weight [0,1] to a CSS color string."""
    w = max(0.0, min(1.0, weight))
    if colormap == "blue":
        # White (0) to deep blue (1)
        r = int(255 * (1 - w))
        g = int(255 * (1 - w))
        b = 255
    else:
        # White (0) to deep red (1)
        r = 255
        g = int(255 * (1 - w))
        b = int(255 * (1 - w))
    return f"rgb({r},{g},{b})"


def _text_color_for_weight(weight: float) -> str:
    """Return black or white text depending on background intensity."""
    return "#fff" if weight > 0.5 else "#000"


def _render_heatmap_table(
    weights: Tensor,  # (seq_len, seq_len)
    tokens: list[str],
    title: str,
) -> str:
    """Render a single attention heatmap as an HTML table."""
    seq_len = weights.shape[0]

    # Header row with token labels
    header_cells = '<th style="padding:2px 4px;font-size:11px;"></th>'
    header_cells += "".join(
        f'<th style="padding:2px 4px;font-size:11px;max-width:40px;overflow:hidden;'
        f"text-overflow:ellipsis;writing-mode:vertical-lr;text-orientation:mixed;"
        f'transform:rotate(180deg);height:60px;">{escape(tokens[j])}</th>'
        for j in range(seq_len)
    )

    rows: list[str] = []
    for i in range(seq_len):
        tok_label = escape(tokens[i])
        cells = f'<td style="padding:2px 4px;font-size:11px;font-weight:bold;white-space:nowrap;">{tok_label}</td>'
        for j in range(seq_len):
            w = weights[i, j].item()
            bg = _weight_to_color(w)
            tc = _text_color_for_weight(w)
            val_str = f"{w:.2f}" if w > 0.01 else ""
            cells += (
                f'<td style="padding:2px;width:28px;height:28px;text-align:center;'
                f"font-size:9px;background:{bg};color:{tc};"
                f'border:1px solid #eee;">{val_str}</td>'
            )
        rows.append(f"<tr>{cells}</tr>")

    return (
        f'<div style="margin:12px 0;">'
        f'<div style="font-weight:bold;margin-bottom:4px;">{escape(title)}</div>'
        f'<table style="border-collapse:collapse;">'
        f"<tr>{header_cells}</tr>"
        f"{''.join(rows)}"
        f"</table></div>"
    )


def render_attention_html(
    attention_weights: list[Tensor],  # list of (n_heads, seq_len, seq_len)
    tokens: list[str],
    title: str = "Attention Patterns",
    layer: int | None = None,
    head: int | None = None,
) -> str:
    """Render attention heatmaps as self-contained HTML.

    Each heatmap is a grid where cell (i,j) shows how much token i attends to token j.
    Color intensity = attention weight (0=white, 1=deep blue).

    Layout:
    - If layer and head specified: single large heatmap
    - If only layer: grid of heads for that layer
    - If neither: grid of layers, each showing mean attention across heads
    """
    sections: list[str] = []

    if layer is not None and head is not None:
        # Single heatmap for specific layer and head
        weights = attention_weights[layer][head]  # (seq_len, seq_len)
        sections.append(_render_heatmap_table(weights, tokens, f"Layer {layer}, Head {head}"))
    elif layer is not None:
        # All heads for a specific layer
        layer_weights = attention_weights[layer]  # (n_heads, seq_len, seq_len)
        n_heads = layer_weights.shape[0]
        for h in range(n_heads):
            sections.append(_render_heatmap_table(layer_weights[h], tokens, f"Layer {layer}, Head {h}"))
    else:
        # Mean attention per layer
        for l_idx, layer_weights in enumerate(attention_weights):
            mean_weights = layer_weights.mean(dim=0)  # (seq_len, seq_len)
            sections.append(_render_heatmap_table(mean_weights, tokens, f"Layer {l_idx} (mean)"))

    return (
        f'<div style="font-family:sans-serif;padding:16px;">'
        f"<h2>{escape(title)}</h2>"
        f'<div style="display:flex;flex-wrap:wrap;gap:16px;">'
        f"{''.join(sections)}"
        f"</div></div>"
    )


def render_wordle_attention_html(
    attention_weights: list[Tensor],
    game_state_tokens: list[str],
    title: str = "Wordle Attention Patterns",
) -> str:
    """Specialized attention viz for Wordle game states.

    Highlights the structure: which feedback tokens (green/yellow/gray)
    the model attends to when generating the next guess.

    Groups tokens by guess turn and uses visual separators.
    Highlights attention from the last position (where the next token is predicted)
    to all prior positions.
    """
    seq_len = len(game_state_tokens)
    sections: list[str] = []

    # Identify turn boundaries (look for common separators like newline, |, etc.)
    turn_boundaries: list[int] = [0]
    for i, tok in enumerate(game_state_tokens):
        if tok in ("\n", "|", ";", "[SEP]"):
            turn_boundaries.append(i + 1)

    # Color-code tokens in the header based on Wordle feedback
    _feedback_colors = {
        "green": "#6aaa64",
        "yellow": "#c9b458",
        "gray": "#787c7e",
        "grey": "#787c7e",
    }

    def _token_style(token: str) -> str:
        token_lower = token.strip().lower()
        for key, color in _feedback_colors.items():
            if key in token_lower:
                return f"background:{color};color:#fff;padding:1px 3px;border-radius:2px;"
        return ""

    # Section 1: Last-position attention (what does the model attend to for its next prediction?)
    sections.append(
        '<div style="margin:16px 0;">'
        "<h3>Last-Position Attention (Next Token Prediction)</h3>"
        "<p>Shows what the model attends to when predicting the next token.</p>"
    )

    for l_idx, layer_weights in enumerate(attention_weights):
        # Mean across heads for the last position
        last_pos_attn = layer_weights[:, -1, :].mean(dim=0)  # (seq_len,)

        # Render as a horizontal bar
        cells: list[str] = []
        for j in range(seq_len):
            w = last_pos_attn[j].item()
            bg = _weight_to_color(w, "blue")
            tc = _text_color_for_weight(w)
            style = (
                f"display:inline-block;padding:4px 6px;margin:1px;"
                f"font-size:12px;background:{bg};color:{tc};"
                f"border:1px solid #ddd;border-radius:3px;"
                f"min-width:24px;text-align:center;"
            )
            tok_display = escape(game_state_tokens[j])
            cells.append(f'<span style="{style}" title="weight: {w:.3f}">{tok_display}</span>')

            # Add visual separator at turn boundaries
            if (j + 1) in turn_boundaries and j < seq_len - 1:
                cells.append(
                    '<span style="display:inline-block;width:2px;height:24px;'
                    'background:#333;margin:0 4px;vertical-align:middle;"></span>'
                )

        sections.append(
            f'<div style="margin:8px 0;">'
            f"<strong>Layer {l_idx}:</strong><br>"
            f'<div style="margin:4px 0;">{"".join(cells)}</div>'
            f"</div>"
        )
    sections.append("</div>")

    # Section 2: Full heatmaps (mean across heads, per layer)
    sections.append('<div style="margin:16px 0;"><h3>Full Attention Heatmaps</h3>')
    for l_idx, layer_weights in enumerate(attention_weights):
        mean_weights = layer_weights.mean(dim=0)
        sections.append(_render_heatmap_table(mean_weights, game_state_tokens, f"Layer {l_idx} (mean)"))
    sections.append("</div>")

    return f'<div style="font-family:sans-serif;padding:16px;"><h2>{escape(title)}</h2>{"".join(sections)}</div>'
