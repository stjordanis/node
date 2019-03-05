from .layer import Layer 
import numpy as np
import math

class MultivariateNormal(Layer):
    """
    多変量正規分布

    注意
        簡単のため、分散共分散行列は(対角行列の)対角成分のベクトルで渡されることを仮定する。
    """

    def __call__(self, x, mu, sigma):

        return sigma * x + mu

    def log_prob(self, x, mu, sigma):
        """
        対数尤度を計算する。
        """

        double_mahalanobis = ((x - mu) / sigma * (x - mu)).sum(1) 
        half_log_det = 0.5 * sigma.sum(1).log()
        
        return -0.5 * (x.value.shape[1] * np.log(2 * math.pi) + double_mahalanobis) - half_log_det