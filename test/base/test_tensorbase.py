import sys
sys.path.append("./")

from minitorch import Tensor, Shape, Device, DType, DTypeEnum


if __name__ == "__main__":
    tensor = Tensor(bytearray(10 * 4), Shape(3, 2, 4), DType(DTypeEnum.KFloat32), Device.KCpu)
    print(memoryview(tensor))
    print(1)
