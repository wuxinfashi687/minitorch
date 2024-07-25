import sys

import numpy as np

sys.path.append("./")

from minitorch import Tensor, Device, DType, DTypeEnum, Shape, zeros, from_numpy


def gen_tensor():
    tensor = zeros((3, 2, 4), DType(DTypeEnum.KFloat32), Device.KCpu)
    return tensor


if __name__ == "__main__":
    tensor = Tensor(np.ones(shape=(3, 2, 4)),  Shape([3, 2, 4]), DType(DTypeEnum.KFloat32), Device.KCpu)
    print(tensor.shape)
    tensor = gen_tensor()
    print(tensor[1:2])
    array = np.random.randn(3, 2, 4).astype(np.float32)
    tensor = from_numpy(array)
    print(tensor)
    tensor_b = tensor[1:2]
    print(tensor_b)
