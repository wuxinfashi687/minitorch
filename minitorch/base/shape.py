from typing import overload, List, Tuple

from minitorch._C.binding import Shape_


class Shape(object):
    @overload
    def __init__(self, sizes: List[int]) -> None:
        pass

    @overload
    def __init__(self, sizes: Tuple[int]) -> None:
        pass

    @overload
    def __init__(self, *sizes: int) -> None:
        pass

    def __init__(self, *args, **kwargs) -> None:
        if "sizes" in kwargs:
            self._obj = kwargs.pop("sizes")
            return

        if len(args) == 1:
            if isinstance(args[0], list):
                self._obj = args[0]
            elif isinstance(args[0], tuple):
                self._obj = list(args[0])
            else:
                raise TypeError(f"预计接受一个List[int]类型或者Tuple[int]类型的参数，实际为{type(args[0])}!")
        else:
            self._obj = list(args)

    @property
    def ndim(self) -> int:
        return Shape_(self._obj).ndim()

    @property
    def stride(self) -> List[int]:
        return Shape_(self._obj).stride

    def tolist(self) -> list:
        return self._obj.copy()

    @property
    def size(self) -> int:
        return Shape_(self._obj).elem_size()
    
    def __repr__(self) -> str:
        return f"shape: {Shape_(self._obj).shape}, stride: {Shape_(self._obj).stride}"

    def __setitem__(self, index: int, item: int) -> None:
        if index >= 0:
            return self._obj.__setitem__(index, item)
        else:
            return self._obj.__setitem__(self.ndim + index, item)
        
    def __getitem__(self, index: int) -> int:
        if index >= 0:
            return self._obj.__getitem__(index)
        else:
            return self._obj.__getitem__(self.ndim + index)
