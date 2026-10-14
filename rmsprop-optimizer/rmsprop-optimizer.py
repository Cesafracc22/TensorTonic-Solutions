import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    W = np.asarray (w)
    G = np.asarray(g)
    S = np.asarray(s)
    new_s = beta * S + (1-beta) * np.square(G)
    new_w = W - (lr/(np.sqrt(new_s + eps)))*G
    return (new_w, new_s)