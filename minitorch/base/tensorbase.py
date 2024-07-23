from typing import Sequence, Optional

from minitorch._C.binding import Tensor_, DTypeEnum, zeros_
from minitorch.base.device import Device
from minitorch.base.dtype import DType
from minitorch.base.shape import Shape, Shape_
from minitorch.base.types import WithMemoryView


class Tensor(object):
    def __init__(
        self,
        memory: WithMemoryView,
        shape: Shape | Sequence[int] | None = (1, ),
        dtype: DType | DTypeEnum | None = DType(DTypeEnum.KInt32),
        device: Device | None = Device.KCpu,
        obj: Optional[Tensor_] = None
    ):
        """


        Args:
            memory:
            shape:
            dtype:
            device:
        """

        if obj is not None:
            self._obj = obj
            return
        if isinstance(shape, list):
            shape: Shape = Shape(shape)
        elif isinstance(shape, tuple):
            shape: Shape = Shape(*shape)
        else:
            if not isinstance(shape, Shape):
                raise TypeError(f"预计输入Sequence[int]或者Shape类型的参数，实际为{type(shape)}!")
        if isinstance(dtype, DTypeEnum):
            dtype = DType(dtype)
        self._obj = Tensor_(memory, Shape_(shape.tolist()), dtype, device)

    @property
    def shape(self) -> Shape:
        return Shape(self._obj.get_shape().shape)

    @property
    def dtype(self) -> DType:
        return DType(self._obj.get_dtype().type)

    def __repr__(self) -> str:
        return self._obj.to_string()

    def flatten_get(self, index: int) -> int | float | bool:
        if index < 0 or index >= self.shape.size:
            raise IndexError
        return self._obj.flatten_get(index)


def zeros(shape: Shape | Sequence[int], dtype: DType, device: Device) -> Tensor:
    if isinstance(shape, list):
        shape = Shape_(shape)
    elif isinstance(shape, tuple):
        shape = Shape_(list(shape))
    elif isinstance(shape, Shape):
        shape = Shape_(shape.tolist())
    else:
        raise TypeError(f"shape参数预计接受Shape类型或者Sequence[int]类型, 实际为{type(shape)}!")
    _C_tensor = zeros_(shape, dtype, device)
    return Tensor(None, None, None, None, _C_tensor)
