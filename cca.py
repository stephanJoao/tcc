import numpy as np
from sklearn.cross_decomposition import CCA

def build_target(target_freq, sfreq, total_time, num_harmonics=3, plot=False):
    y = np.zeros((num_harmonics * 2, total_time))
    for i in range(1, num_harmonics + 1):
        y_sin = np.sin(
            2 * np.pi * target_freq * i * np.arange(total_time) / sfreq
        )
        y_cos = np.cos(
            2 * np.pi * target_freq * i * np.arange(total_time) / sfreq
        )
        y[(i - 1) * 2] = y_sin
        y[(i - 1) * 2 + 1] = y_cos            )
    return y

class cca():
	def __init__(self, n_harmonics: int = 3):
		if type(n_harmonics) != int:
			raise ValueError ("Has to be an integer type value")
		else:
			self.n_harmonics = n_harmonics
		self.cca = CCA(n_components=1)
		self.harmonics = []

	def fit(self, eegdata):
		if type(eegdata) != dict:
			raise ValueError ("Has to be a dict type")         
		
		return self