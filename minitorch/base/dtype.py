from minitorch._C.binding import DType_


class DType(DType_):
    def __repr__(self):
        return f"{self.type}"
