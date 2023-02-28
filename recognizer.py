import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data_path = r"./data/apple_vector.csv"
base = pd.read_csv(data_path, index_col = None).values

def recognizer(mat, base = base):
    """
    170개의 사과 이미지 벡터를 사전에 저장된 9개의 사과 이미지 벡터와 비교하여, 170개의 사과가 각각 어느 사과인지 반환한다.

    input
    - mat : 170개의 사과 이미지의 벡터를 포함한 매트릭스, shape : (170, 300)
    - base : 9개의 사과 이미지의 대표 벡터를 포함한 매트릭스, shape (9, 300)

    output
    - result : 각 사과 이미지의 라벨을 포함한 리스트
    """
    sim = cosine_similarity(mat, base)
    result = np.argmax(sim, axis = 1)
    result = np.flip(result).reshape([10, -1]) +1

    return result



    