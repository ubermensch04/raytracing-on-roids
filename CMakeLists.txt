cmake_minimum_required(VERSION 3.18)
project(ParallelRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Raylib via FetchContent (downloads and builds automatically) ──
include(FetchContent)
FetchContent_Declare(
  raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG        5.5
)
FetchContent_MakeAvailable(raylib)

# ── OpenMP ──
find_package(OpenMP REQUIRED)

# ── CUDA architecture (change sm_86 to match YOUR GPU) ──
# RTX 20-series: sm_75  |  RTX 30-series: sm_86  |  RTX 40-series: sm_89
set(CMAKE_CUDA_ARCHITECTURES 86)

# ── Source files ──
set(SOURCES
  src/main.cpp
  src/render_seq.cpp
  src/render_omp.cpp
  src/render_cuda.cu
)

add_executable(raytracer ${SOURCES})

# ── Link everything ──
target_link_libraries(raytracer PRIVATE
  raylib
  OpenMP::OpenMP_CXX
)

# ── Pass OpenMP flags through nvcc to host compiler ──
target_compile_options(raytracer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
)

# ── CUDA: allow __host__ __device__ in headers included by .cpp files ──
set_target_properties(raytracer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)