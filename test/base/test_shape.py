import sys
sys.path.append("./")
from minitorch.base.shape import Shape


if __name__ == "__main__":
    shape = Shape(sizes=[1, 2, 3])
    print(shape)
    print(shape.ndim)
    shape[1] = 5
    print(shape)
    print(shape[-1])
