cmake_minimum_required(VERSION 3.16)

project(fast_rag LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(BLAS REQUIRED)

add_executable(fast_rag
    src/main.cpp
    src/doc_utilities.cpp
    src/embedding_utilities.cpp
    src/graph_utilities.cpp
)

target_link_libraries(fast_rag PRIVATE ${BLAS_LIBRARIES})

target_include_directories(fast_rag PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

find_package(Threads REQUIRED)
target_link_libraries(fast_rag PRIVATE Threads::Threads)

target_compile_features(fast_rag PRIVATE cxx_std_20)
set_target_properties(fast_rag PROPERTIES
    CXX_EXTENSIONS OFF
)
target_link_libraries(fast_rag PRIVATE MPI::MPI_CXX)
