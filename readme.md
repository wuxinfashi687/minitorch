'''
pip install pipx  # cpp环境
pipx install poetry
poetry install
poetry run gen_cmake
mkdir build
cd build
cmake build ..
cmake --build .\build\ --config Release
'''