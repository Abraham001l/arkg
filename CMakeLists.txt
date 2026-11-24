cmake_minimum_required(VERSION 3.16)

project(fast_rag LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(BLAS REQUIRED)

add_executable(arkg_builder
    src/arkg_builder/main.cpp
    src/arkg_builder/doc_utilities.cpp
    src/arkg_builder/embedding_utilities.cpp
    src/arkg_builder/graph_utilities.cpp
)

target_link_libraries(arkg_builder PRIVATE ${BLAS_LIBRARIES})

target_include_directories(arkg_builder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

find_package(Threads REQUIRED)
target_link_libraries(arkg_builder PRIVATE Threads::Threads)

target_compile_features(arkg_builder PRIVATE cxx_std_20)
set_target_properties(arkg_builder PROPERTIES
    CXX_EXTENSIONS OFF
)
target_link_libraries(arkg_builder PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(file_host
    src/file_host/main.cpp
    src/file_host/endpoints.cpp
)

target_include_directories(file_host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(file_host PRIVATE Threads::Threads)
target_compile_features(file_host PRIVATE cxx_std_20)
set_target_properties(file_host PROPERTIES CXX_EXTENSIONS OFF)

add_executable(arkg_host
    src/arkg_host/main.cpp
    src/arkg_host/query_utilities.cpp
)

target_include_directories(arkg_host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(arkg_host PRIVATE Threads::Threads MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_features(arkg_host PRIVATE cxx_std_20)
set_target_properties(arkg_host PROPERTIES CXX_EXTENSIONS OFF)
