import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    p = np.asarray(param)
    g = np.asarray(grad)
    M = np.asarray(m)
    V = np.asarray(v)
    
    M = beta1 * M + (1.0-beta1)*g
    V = beta2 * V + (1.0-beta2)*np.square(g)
    m_hat= M/(1.0-beta1**t)
    v_hat = V/(1.0-beta2**t)
    param_new = p -lr * m_hat/(np.sqrt(v_hat)+ eps)
    return (param_new,M,V)