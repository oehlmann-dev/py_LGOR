"""mirrorlines (LGOR 3.7 name): alias of the gtlines package (LGOR 3.8)."""
from .gtlines.tft import chiodo4, fjrw_corr, LIT_COUNT, _RSpinAdapter
from .gtlines.blocks import group_closure
from .gtlines.pipeline import analyse_model, compare_states


def compare_pair(pathX, pathY, nmax=5):
    text, rows, _ = analyse_model(pathX, mirror=pathY, nmax=nmax, chats=(1,))
    return text, rows


def compare_lgor_states(lgoX, csvX, lgoY, csvY, block_fields, nmax=4):
    return compare_states(lgoX, csvX, lgoY, csvY, block_fields, nmax=nmax)
