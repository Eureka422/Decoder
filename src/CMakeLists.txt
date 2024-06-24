cmake_minimum_required(VERSION 3.10)

project(srp CXX)

add_executable(srp
    main.cpp
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenCV 4 REQUIRED)

set(ZBar_INCLUDE_DIRS /usr/local/include)
set(ZBAR_LIBRARIES /usr/local/lib/libzbar.so)

target_include_directories(srp
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${ZBar_INCLUDE_DIRS}
        "include"
)

target_link_libraries(srp
    PUBLIC
	libmali.so
	libdmtx.so
    	libpthread.so
        ${OpenCV_LIBS} 
        ${ZBAR_LIBRARIES}
)



