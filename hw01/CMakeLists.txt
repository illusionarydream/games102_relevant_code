cmake_minimum_required(VERSION 3.12)
project(MyProject)

set(CMAKE_CXX_STANDARD 11)

# 设置 Anaconda 的 Python 路径
set(Python3_ROOT_DIR "/opt/anaconda")
set(Python3_EXECUTABLE "/opt/anaconda/bin/python3")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# message(STATUS "Python interpreter: ${Python3_EXECUTABLE}")
# message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
# message(STATUS "NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")

# 包含Python3和NumPy的头文件
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
include_directories(./include)

# 添加可执行文件
add_executable(MyExecutable main.cpp)

# 链接Python3库
target_link_libraries(MyExecutable Python3::Python Python3::NumPy)