cmake_minimum_required(VERSION 3.10)
project(sakura)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finput-charset=UTF-8 -fexec-charset=UTF-8")

find_package(PkgConfig REQUIRED)
pkg_check_modules(PC_SIXEL QUIET sixel)

if (PC_SIXEL_FOUND)
  message(STATUS "Found sixel via pkg-config.")
  add_library(SIXEL::sixel INTERFACE IMPORTED)
  set_property(TARGET SIXEL::sixel PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PC_SIXEL_INCLUDE_DIRS})
  set_property(TARGET SIXEL::sixel PROPERTY INTERFACE_LINK_LIBRARIES ${PC_SIXEL_LIBRARIES})
else()
  message(STATUS "Could not find sixel via pkg-config. Searching manually.")
  find_path(SIXEL_INCLUDE_DIR sixel.h
    HINTS /usr/include /usr/local/include /opt/local/include)
  find_library(SIXEL_LIBRARY NAMES sixel
    HINTS /usr/lib /usr/local/lib /opt/local/lib)

  if (SIXEL_INCLUDE_DIR AND SIXEL_LIBRARY)
    message(STATUS "Found sixel manually.")
    add_library(SIXEL::sixel UNKNOWN IMPORTED)
    set_target_properties(SIXEL::sixel PROPERTIES
      IMPORTED_LOCATION "${SIXEL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${SIXEL_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Could not find sixel library. Please ensure it is installed and CMAKE_PREFIX_PATH is set correctly.")
  endif()
endif()

find_package(OpenSSL REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
find_package(cpr REQUIRED)
find_package(Threads REQUIRED)

add_library(SakuraLib STATIC
  sakura.cpp
)

target_include_directories(SakuraLib PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(SakuraLib PRIVATE
  cpr::cpr
  ${OpenCV_LIBS}
  SIXEL::sixel
  Threads::Threads
)

add_executable(sakura
  example.cpp
)

target_include_directories(sakura PRIVATE
  .
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sakura PRIVATE
  SakuraLib
  ${OpenCV_LIBS}
  SIXEL::sixel
  Threads::Threads
)

install(TARGETS sakura
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
