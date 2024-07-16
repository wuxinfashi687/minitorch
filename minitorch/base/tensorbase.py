from minitorch._C.binding import __Tensor
from minitorch.base.shape import Shape


class Tensor(__Tensor):
    @property
    def shape(self) -> Shape:
        pass
