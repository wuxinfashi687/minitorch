import sys
import os
from shutil import copyfile
import subprocess
from typing import List

from pybind11 import get_include
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from pybind11_stubgen import ModuleStubsGenerator


def get_python_path() -> str:
    poetry_root_path = sys.exec_prefix
    cfg_path = os.path.join(poetry_root_path, "pyvenv.cfg")
    with open(cfg_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "base-exec-prefix" in line:
                python_root_path = line.split(" = ")[-1].replace('\\', '\\\\')[:-1]
                return python_root_path
            

def get_python_include() -> str:
    python_root_path = get_python_path()
    return python_root_path + "\\\\include"


def get_python_lib_dir() -> str:
    python_root_path = get_python_path()
    return python_root_path + "\\\\libs"


def get_cuda_path() -> str:
    cur_cmd = "where.exe nvcc"
    exit_code = subprocess.run(cur_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    path = None
    if not exit_code.returncode:
        path = exit_code.stdout.decode("gbk")
    else:
        print(exit_code.stderr.decode("gbk"))
        sys.exit(-1)
    path = path.strip()
    path = path[:-13].replace('\\', '\\\\')
    return path


def gen_cmake_file() -> None:
    with open("./CMakeLists.txt", "w") as f:
        exe_path = sys.executable.replace('\\', '\\\\')
        pybind11_include_path = get_include().replace('\\', '\\\\')
        f.write(
            f"""cmake_minimum_required(VERSION 3.28)
project(minitorch CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(Python_EXECUTABLE {exe_path})
            """ +
            """
find_package(Python3 COMPONENTS Interpreter Development)
link_libraries(${PYTHON_LIBRARIES})
            """ + 
            f"""
include_directories({get_python_include()})
include_directories({pybind11_include_path})
include_directories({get_cuda_path()}\\\\include)
link_directories({get_python_lib_dir()})

add_library(minitorch SHARED minitorch/_C/include/binding.cu)
            """ + 
            """
target_link_libraries(minitorch python310)
            
if(WIN32)
    file(GLOB DLL_FILES "${CMAKE_BINARY_DIR}/Release/*.dll")
    foreach(DLL_FILE ${DLL_FILES})
        file(COPY ${DLL_FILE} DESTINATION ../minitorch/_C)
    endforeach()
endif()
if(LINUX)
    file(GLOB DLL_FILES "${CMAKE_BINARY_DIR}/Release/*.so")
    foreach(DLL_FILE ${DLL_FILES})
        file(COPY ${DLL_FILE} DESTINATION ../minitorch/_C)
    endforeach()
endif()

set_target_properties(
        minitorch PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON 
        OUTPUT_NAME "binding" 
        VERSION "0.0.1"
)
            """
        )


def get_version() -> str:
    poetry_file_path = "./pyproject.toml"
    with open(poetry_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "version" in line:
                version = line.split(" = ")[-1].replace('"', '')
                return version
            

def get_python_require() -> str:
    poetry_file_path = "./pyproject.toml"
    with open(poetry_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "python" in line:
                require = line.split(" = ")[-1].replace('"', '')
                return require


class MiniTorchExtension(object):
    def __init__(self):
        pass

    @staticmethod
    def gen_stubs() -> None:
        print("正在生成Python接口文件......")
        try:
            from minitorch._C import binding
            module = ModuleStubsGenerator(binding)
            module.parse()
            module.write_setup_py = False

            with open("./minitorch/_C/binding.pyi", "w") as fp:
                fp.write("#\n# Automatically generated file, do not edit!\n#\n\n")
                fp.write("\n".join(module.to_lines()))
        except ImportError:
            for pyd in os.listdir("./minitorch/_C"):
                if os.path.isfile(os.path.join("./minitorch/_C", pyd)) and pyd.startswith("binding") and pyd.endswith(".pyd"):
                    raise RuntimeError(f"minitorch._C模块下存在扩展{pyd}, 但是无法正常打开，请检查该文件是否被正确的编译!!!")
            raise FileNotFoundError("minitorch._C模块下不存在可以使用的binding.pyd!")

    @staticmethod
    def gen_cmake() -> None:
        gen_cmake_file()

    @staticmethod
    def find_extension_path(find_path: str, ext_name: str) -> list:
        valid_path_list = []
        for (root, folders, files) in os.walk(find_path):
            for file in files:
                if file.endswith(".dll") and file.startswith(ext_name):
                    valid_path_list.append(os.path.join(root, file))

        return valid_path_list

    def build_ext(self):
        print("正在编译CUDA C 扩展......")
        if not os.path.exists("./build"):
            cur_cmd = r"mkdir build && cd build && cmake build .. && cd .. && cmake --build .\build --config Release"
        else:
            cur_cmd = r"cd build && cmake build .. && cd .. && cmake --build .\build --config Release"
        exit_code = subprocess.run(cur_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not exit_code.returncode:
            print(exit_code.stdout.decode("gbk"))
        else:
            print(exit_code.stderr.decode("gbk"))
            raise RuntimeError("构建CUDA C 扩展失败!!!")
        print("正在将生成的CUDA C扩展复制到指定位置......")
        dll_path = self.find_extension_path("./build/Release", "binding")
        if len(dll_path) > 1:
            raise RuntimeError("发现Cmake构建文件下有多个minitorch.dll文件，无法区分哪个是目标扩展!")
        if len(dll_path) == 0:
            raise RuntimeError("发现Cmake构建文件下没有minitorch.dll文件!!!")
        dll_path = dll_path[-1]
        dll_name = os.path.basename(dll_path)
        pyd_name = dll_name.replace(".dll", ".pyd")
        copyfile(dll_path, f"./minitorch/_C/{dll_name}")
        os.rename(f"./minitorch/_C/{dll_name}", f"./minitorch/_C/{pyd_name}")
            

if __name__ == "__main__":
    sys.path.append("./")
    extension = MiniTorchExtension()
    extension.gen_cmake()
    extension.build_ext()
    extension.gen_stubs()
    # ext_modules = [
    #     Pybind11Extension(
    #         "binding",
    #         ["minitorch/_C/include/binding.cpp"],
    #         # Example: passing in the version to the compiled code
    #         define_macros=[("VERSION_INFO", get_version())],
    #         include_dirs=["minitorch/_C/include"]
    #     ),
    # ]
    #
    # setup(
    #     name="binding",
    #     version=get_version(),
    #     author="wuxin",
    #     author_email="",
    #     url="",
    #     description="minitorch cpp binding",
    #     long_description="",
    #     ext_modules=ext_modules,
    #     extras_require={"test": "pytest"},
    #     # Currently, build_ext only provides an optional "highest supported C++
    #     # level" feature, but in the future it may provide more features.
    #     cmdclass={"build_ext": build_ext},
    #     zip_safe=False,
    #     python_requires=get_python_require(),
    # )
    #
    # copyfile("./build/lib.win-amd64-cpython-310/binding.cp310-win_amd64.pyd", "./minitorch/_C/binding.cp310-win_amd64.pyd")

