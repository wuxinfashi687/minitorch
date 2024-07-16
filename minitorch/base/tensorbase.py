from typing import Sequence

from minitorch._C.binding import __Tensor, DTypeEnum
from minitorch.base.types import WithMemoryView
from minitorch.base.shape import Shape
from minitorch.base.dtype import DType
from minitorch.base.device import Device


class Tensor(__Tensor):
    def __init__(
        self,
        memory: WithMemoryView,
        shape: Shape | Sequence[int],
        dtype: DType | DTypeEnum,
        device: Device
    ):
        """


        Args:
            memory:
            shape:
            dtype:
            device:
        """

        if isinstance(shape, list):
            shape = Shape(shape)
        elif isinstance(shape, tuple):
            shape = Shape(*shape)
        if isinstance(dtype, DTypeEnum):
            dtype = DType(dtype)
        super(Tensor, self).__init__(memory, shape, dtype, device)
