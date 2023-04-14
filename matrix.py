import numpy as np
from PIL import Image
from typing import Optional


class Matrix:
    def __init__(self, rows: int, cols: int, min_val: int, max_val: int):
        self.rows = rows
        self.cols = cols
        self.min_val = min_val
        self.max_val = max_val
        self.matrix = np.random.randint(min_val, max_val + 1, size=(rows, cols))

    def get_matrix(self):
        return self.matrix.tolist()
    
    def calculate_trace(self):
        min_dim =min(self.rows, self.cols)
        return np.trace(self.matrix[:min_dim, :min_dim])


def calculate_rank(matrix: np.ndarray) -> int:
    return np.linalg.matrix_rank(matrix)

def calculate_pseudo_determinant(matrix: np.ndarray) -> float:
    _, s, _ = np.linalg.svd(matrix)
    pseudo_determinant = np.prod(s)
    return pseudo_determinant

def calculate_inverse(matrix: np.ndarray):
    return np.linalg.pinv(matrix)

def svd_approximation(image, k):
    img_matrix = np.array(image)
    U, S, V = np.linalg.svd(img_matrix, full_matrices=False)
    approx_matrix = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    approx_image = Image.fromarray(np.uint8(approx_matrix))
    return approx_image


