cmake_minimum_required(VERSION 3.10)
project(DatcomWindForce)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-math7 REQUIRED)
find_package(gz-common5 REQUIRED)
add_library(DatcomWindForce SHARED DatcomWindForce.cc)
set_target_properties(DatcomWindForce PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(DatcomWindForce PRIVATE
  gz-sim8::gz-sim8
  gz-plugin2::register
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
  gz-math7::gz-math7
  gz-common5::gz-common5)
