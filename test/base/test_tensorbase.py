import sys
sys.path.append("./")

import minitorch._C.binding as binding


if __name__ == "__main__":
    tensor = binding._Tensor()
    print(1)
