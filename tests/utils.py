import numpy as np
def compute_error(x, x_true, rtol, atol):
    e = (x - x_true) / (atol + rtol * np.abs(x_true))
    return np.linalg.norm(e, axis=0) / np.sqrt(e.shape[0])
