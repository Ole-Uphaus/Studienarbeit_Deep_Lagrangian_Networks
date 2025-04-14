import numpy as np
import torch

# Arrays definieren (Numpy)
arr_a = np.array([1, 2, 3, 4])
arr_b = np.array([[1, 2], [3, 4]])
arr_c = np.array([[1, 2], [3, 4,]])
arr_d = np.array([[1, 2, 3, 4]])

# print('Array vorher:', arr_a)
# print('Shape von Array:', arr_a.shape)
# print('Array gereshaped (-1, 1): \n', arr_a.reshape((-1, 1)))

# print('Array vorher: \n', arr_b)
# print('Shape von Array:', arr_b.shape)
# print('Array gereshaped (-1, 2): \n', arr_b.reshape((-1, 2)))
# print('Array gereshaped (-1, 1): \n', arr_b.reshape((-1, 1)))

# Arrays definieren Torch
arr_e = torch.asarray([1, 2, 3, 4])


print('Tensor vorher:', arr_e)
print('Shape von Array:', arr_e.size())
print('Tensor view(2, 2): \n', arr_e.view((2, 2)))

print('Tensor vorher:', arr_e)
print('Shape von Array:', arr_e.size())
print('Tensor view(-1, 1): \n', arr_e.view((-1, 1)))
