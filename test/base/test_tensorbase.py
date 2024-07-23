import sys

import numpy as np

sys.path.append("./")

from minitorch import Tensor, Device, DType, DTypeEnum, Shape, zeros


def gen_tensor():
    tensor = zeros((3, 2, 4), DType(DTypeEnum.KFloat32), Device.KCpu)
    return tensor


if __name__ == "__main__":
    tensor = Tensor(np.ones(shape=(3, 2, 4)),  Shape([3, 2, 4]), DType(DTypeEnum.KFloat32), Device.KCpu)
    print(tensor.shape)
    tensor = gen_tensor()
    print(tensor)
    print(tensor.flatten_get(24))
