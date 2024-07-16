from minitorch._C.binding import __DType


class DType(__DType):
    def __repr__(self):
        return f"{self.type}"
