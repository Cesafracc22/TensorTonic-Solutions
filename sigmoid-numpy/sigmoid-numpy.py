import numpy as np
import math
def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    z = np.asarray(x)
    res = 1.0/(1.0+np.exp(-z))
        
    return res