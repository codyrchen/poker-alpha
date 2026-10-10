cmake_minimum_required(VERSION 3.26)
project(poker_alpha_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(poker_alpha_native
  src/evaluator.cpp
  src/features.cpp
  src/game.cpp
  src/key.cpp
  src/solver.cpp
  src/tabular.cpp
  src/bindings.cpp
)

target_compile_options(poker_alpha_native PRIVATE
  -Wall -Wextra -Wpedantic
  # Bit-parity with the Python reference requires plain IEEE arithmetic:
  # no fast-math, and no FP contraction (clang's default FMA fusion changed
  # betting arithmetic in the last ulp and broke cross-backend equality).
  -ffp-contract=off
)
# Release semantics: -O3 -DNDEBUG via CMAKE_BUILD_TYPE=Release.

# Developer sanitizer build: cmake -DPOKERALPHA_SANITIZE=address;undefined
set(POKERALPHA_SANITIZE "" CACHE STRING "Semicolon list: address;undefined")
if(POKERALPHA_SANITIZE)
  foreach(san ${POKERALPHA_SANITIZE})
    target_compile_options(poker_alpha_native PRIVATE -fsanitize=${san} -fno-omit-frame-pointer -g)
    target_link_options(poker_alpha_native PRIVATE -fsanitize=${san})
  endforeach()
endif()

install(TARGETS poker_alpha_native LIBRARY DESTINATION .)

# Standalone self-test (no Python): the sanitizer harness. Not built by
# default pip installs; see cpp/tests/native_selftest.cpp for usage.
option(POKERALPHA_BUILD_SELFTEST "Build the native self-test executable" OFF)
if(POKERALPHA_BUILD_SELFTEST)
  add_executable(native_selftest
    tests/native_selftest.cpp
    src/evaluator.cpp
    src/features.cpp
    src/game.cpp
    src/key.cpp
    src/solver.cpp
  )
  target_compile_options(native_selftest PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
  if(POKERALPHA_SANITIZE)
    foreach(san ${POKERALPHA_SANITIZE})
      target_compile_options(native_selftest PRIVATE -fsanitize=${san} -fno-omit-frame-pointer -g)
      target_link_options(native_selftest PRIVATE -fsanitize=${san})
    endforeach()
  endif()
endif()
