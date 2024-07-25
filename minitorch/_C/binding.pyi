#
# Automatically generated file, do not edit!
#

from __future__ import annotations
import minitorch._C.binding
import typing
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "CopyKindEnum",
    "DTypeEnum",
    "DType_",
    "DeviceEnum",
    "FloatEnum",
    "GenericRawDType_",
    "GenericRawFloatType_",
    "GenericRawIntType_",
    "IndexType",
    "IntEnum",
    "NewDim",
    "Shape_",
    "Slice_",
    "Tensor_",
    "UIntEnum",
    "from_numpy_",
    "get_version",
    "zeros_"
]


class CopyKindEnum():
    """
    Members:

      Cpu2Cpu

      Cpu2Cuda

      Cuda2Cpu

      Cuda2Cuda
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    Cpu2Cpu: minitorch._C.binding.CopyKindEnum # value = <CopyKindEnum.Cpu2Cpu: 0>
    Cpu2Cuda: minitorch._C.binding.CopyKindEnum # value = <CopyKindEnum.Cpu2Cuda: 1>
    Cuda2Cpu: minitorch._C.binding.CopyKindEnum # value = <CopyKindEnum.Cuda2Cpu: 2>
    Cuda2Cuda: minitorch._C.binding.CopyKindEnum # value = <CopyKindEnum.Cuda2Cuda: 3>
    __members__: dict # value = {'Cpu2Cpu': <CopyKindEnum.Cpu2Cpu: 0>, 'Cpu2Cuda': <CopyKindEnum.Cpu2Cuda: 1>, 'Cuda2Cpu': <CopyKindEnum.Cuda2Cpu: 2>, 'Cuda2Cuda': <CopyKindEnum.Cuda2Cuda: 3>}
    pass
class DTypeEnum():
    """
    Members:

      KUnKnown

      KInt32

      KInt64

      KInt16

      KInt8

      KUInt32

      KUInt64

      KUInt16

      KUInt8

      KFloat32

      KFloat64

      KBool
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    KBool: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KBool: 11>
    KFloat32: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KFloat32: 9>
    KFloat64: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KFloat64: 10>
    KInt16: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KInt16: 3>
    KInt32: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KInt32: 1>
    KInt64: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KInt64: 2>
    KInt8: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KInt8: 4>
    KUInt16: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KUInt16: 7>
    KUInt32: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KUInt32: 5>
    KUInt64: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KUInt64: 6>
    KUInt8: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KUInt8: 8>
    KUnKnown: minitorch._C.binding.DTypeEnum # value = <DTypeEnum.KUnKnown: 0>
    __members__: dict # value = {'KUnKnown': <DTypeEnum.KUnKnown: 0>, 'KInt32': <DTypeEnum.KInt32: 1>, 'KInt64': <DTypeEnum.KInt64: 2>, 'KInt16': <DTypeEnum.KInt16: 3>, 'KInt8': <DTypeEnum.KInt8: 4>, 'KUInt32': <DTypeEnum.KUInt32: 5>, 'KUInt64': <DTypeEnum.KUInt64: 6>, 'KUInt16': <DTypeEnum.KUInt16: 7>, 'KUInt8': <DTypeEnum.KUInt8: 8>, 'KFloat32': <DTypeEnum.KFloat32: 9>, 'KFloat64': <DTypeEnum.KFloat64: 10>, 'KBool': <DTypeEnum.KBool: 11>}
    pass
class DType_():
    def __init__(self, dtype_enum: DTypeEnum) -> None: ...
    @property
    def type(self) -> DTypeEnum:
        """
        :type: DTypeEnum
        """
    pass
class DeviceEnum():
    """
    Members:

      KUnknown

      KCpu

      kCuda
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    KCpu: minitorch._C.binding.DeviceEnum # value = <DeviceEnum.KCpu: 1>
    KUnknown: minitorch._C.binding.DeviceEnum # value = <DeviceEnum.KUnknown: 0>
    __members__: dict # value = {'KUnknown': <DeviceEnum.KUnknown: 0>, 'KCpu': <DeviceEnum.KCpu: 1>, 'kCuda': <DeviceEnum.kCuda: 2>}
    kCuda: minitorch._C.binding.DeviceEnum # value = <DeviceEnum.kCuda: 2>
    pass
class FloatEnum():
    """
    Members:

      KFloat32

      KFloat64
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    KFloat32: minitorch._C.binding.FloatEnum # value = <FloatEnum.KFloat32: 0>
    KFloat64: minitorch._C.binding.FloatEnum # value = <FloatEnum.KFloat64: 1>
    __members__: dict # value = {'KFloat32': <FloatEnum.KFloat32: 0>, 'KFloat64': <FloatEnum.KFloat64: 1>}
    pass
class GenericRawDType_():
    def __init__(self) -> None: ...
    pass
class GenericRawFloatType_():
    def __init__(self) -> None: ...
    pass
class GenericRawIntType_():
    def __init__(self) -> None: ...
    pass
class IndexType():
    pass
class IntEnum():
    """
    Members:

      KInt32

      KInt64

      KInt16

      KInt8
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    KInt16: minitorch._C.binding.IntEnum # value = <IntEnum.KInt16: 2>
    KInt32: minitorch._C.binding.IntEnum # value = <IntEnum.KInt32: 0>
    KInt64: minitorch._C.binding.IntEnum # value = <IntEnum.KInt64: 1>
    KInt8: minitorch._C.binding.IntEnum # value = <IntEnum.KInt8: 3>
    __members__: dict # value = {'KInt32': <IntEnum.KInt32: 0>, 'KInt64': <IntEnum.KInt64: 1>, 'KInt16': <IntEnum.KInt16: 2>, 'KInt8': <IntEnum.KInt8: 3>}
    pass
class NewDim():
    pass
class Shape_():
    def __getitem__(self, index: int) -> int: ...
    @typing.overload
    def __init__(self, sizes: list[int]) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    def __setitem__(self, index: int, item: int) -> None: ...
    def elem_size(self) -> int: ...
    def ndim(self) -> int: ...
    @property
    def shape(self) -> list[int]:
        """
        :type: list[int]
        """
    @property
    def stride(self) -> list[int]:
        """
        :type: list[int]
        """
    pass
class Slice_():
    @typing.overload
    def __init__(self, start: typing.Union[int, None], end: typing.Union[int, None], step: int) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    pass
class Tensor_():
    def __init__(self, buffer: Buffer, shape: Shape_, dtype: DType_, device_enum: DeviceEnum) -> None: ...
    def flatten_get(self, index: int) -> typing.Union[int, int, int, int, int, int, int, int, float, float, bool, None]: ...
    def get_dtype(self) -> DType_: ...
    def get_item(self, slices: list[Slice_]) -> Tensor_: ...
    def get_shape(self) -> Shape_: ...
    def to_string(self) -> str: ...
    pass
class UIntEnum():
    """
    Members:

      UKInt32

      UKInt64

      UKInt16

      UKInt8
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    UKInt16: minitorch._C.binding.UIntEnum # value = <UIntEnum.UKInt16: 2>
    UKInt32: minitorch._C.binding.UIntEnum # value = <UIntEnum.UKInt32: 0>
    UKInt64: minitorch._C.binding.UIntEnum # value = <UIntEnum.UKInt64: 1>
    UKInt8: minitorch._C.binding.UIntEnum # value = <UIntEnum.UKInt8: 3>
    __members__: dict # value = {'UKInt32': <UIntEnum.UKInt32: 0>, 'UKInt64': <UIntEnum.UKInt64: 1>, 'UKInt16': <UIntEnum.UKInt16: 2>, 'UKInt8': <UIntEnum.UKInt8: 3>}
    pass
def __init_print_stream() -> None:
    pass
def from_numpy_(ndarray: numpy.ndarray) -> Tensor_:
    pass
def get_version() -> str:
    pass
def zeros_(shape: Shape_, dtype: DType_, device_enum: DeviceEnum) -> Tensor_:
    pass
