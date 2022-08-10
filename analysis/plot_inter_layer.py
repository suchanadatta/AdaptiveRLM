import numpy as np
import matplotlib.pyplot as plt

histogram = np.genfromtxt('/store/adaptive_feedback/sample_run_trec/submodel.train', delimiter=" ")
matrix = histogram[:, 2:]
print(matrix)
print(matrix.shape)
# plt.figure(figsize=(50,50))
plt.matshow(matrix, fignum=1, aspect='auto')
# plt.matshow(foo)
plt.colorbar()
plt.show()