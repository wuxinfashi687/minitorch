from typing import overload, List

from minitorch._C.binding import __Shape


class Shape(__Shape):
    @overload
    def __init__(self, sizes: List[int]) -> None:
        pass

    @overload
    def __init__(self, *sizes: int) -> None:
        pass

    def __init__(self, *args, **kwargs) -> None:
        if "sizes" in kwargs:
            super(Shape, self).__init__(kwargs.pop("sizes"))

        if len(args) == 1 and isinstance(args[0], list):
            super(Shape, self).__init__(args[0])
        else:
            super(Shape, self).__init__(list(args))
        
    @property
    def ndim(self) -> int:
        return self._ndim()
    
    def __repr__(self) -> str:
        return self._shape.__repr__()  

    def __setitem__(self, index: int, item: int) -> None:
        if index >= 0:
            return super().__setitem__(index, item)
        else:
            return super().__setitem__(self.ndim + index, item)
        
    def __getitem__(self, index: int) -> int:
        if index >= 0:
            return super().__getitem__(index)
        else:
            return super().__getitem__(self.ndim + index)
