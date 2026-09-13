cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# 1. 寻找生成消息所需的依赖
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 2. 声明你的 .msg 文件
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VlmTask.msg"
  DEPENDENCIES std_msgs sensor_msgs
)

# 3. 导出依赖
ament_export_dependencies(rosidl_default_runtime)

ament_package()
