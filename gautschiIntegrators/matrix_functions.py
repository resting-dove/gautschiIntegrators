# Default matrix functions
import numpy as np
import scipy.sparse


def cosm_sqrt(h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
    w, v = scipy.linalg.eig(omega2.todense())
    fw = np.diag(np.cos(h * np.sqrt(w)))
    return v @ (fw @ (v.T @ b))


def sincm_sqrt(h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
    w, v = scipy.linalg.eig(omega2.todense())
    fw = np.diag(np.sinc(h * np.sqrt(w) / np.pi))
    return v @ (fw @ (v.T @ b))


def sym_cosm_sqrt(h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
    w, v = scipy.linalg.eigh(omega2.todense())
    fw = np.diag(np.cos(h * np.sqrt(w)))
    return v @ (fw @ (v.T @ b))


def sym_sincm_sqrt(h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
    w, v = scipy.linalg.eigh(omega2.todense())
    fw = np.diag(np.sinc(h * np.sqrt(w) / np.pi))
    return v @ (fw @ (v.T @ b))


def sym_msinm_sqrt(h: float, omega2: scipy.sparse.sparray, b: np.array) -> np.array:
    w, v = scipy.linalg.eigh(omega2.todense())
    fw = np.diag(np.sqrt(w) * np.sin(h * np.sqrt(w)))
    return v @ (fw @ (v.T @ b))
