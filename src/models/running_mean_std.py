"""
이동 평균과 표준편차 계산 모듈

Welford의 알고리즘을 사용하여 상태 및 보상의 정규화를 위한 이동 평균과 표준편차를 계산하는 클래스를 구현합니다.
주로 강화학습 환경에서 입력 상태나 보상의 스케일을 정규화하기 위해 사용됩니다.
"""

import numpy as np
from src.constants import RMS_EPSILON

class RunningMeanStd:
    """
    Welford's online algorithm을 사용하여 이동 평균과 표준편차를 계산합니다.
    상태 및 보상 정규화에 사용됩니다.
    Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, epsilon=RMS_EPSILON, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """배치 데이터로 평균과 분산을 업데이트합니다."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """계산된 평균과 분산으로 내부 상태를 업데이트합니다."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count 