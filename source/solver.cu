#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/vec2.hpp>
#include <iostream>
#include "fem/solver.h"

/* The data in the simulator must have been initialized, 
   i.e. createMesh() before setUp() */
void Solver::setUp(Simulator *sim) {
  this->sim = sim;
  int device_count = 0;
  std::string name;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  printf("---------------------------------------------------------\n");
  printf("Initializing CUDA for CudaRenderer\n");
  printf("Found %d CUDA devices\n", device_count);

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    name = deviceProps.name;
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");

  cudaMalloc(&cudaDevicePosition,     N_POINTS * sizeof(glm::vec2));
  cudaMalloc(&cudaDeviceVelocity,     N_POINTS * sizeof(glm::vec2));
  cudaMalloc(&cudaDeviceAcceleration, N_POINTS * sizeof(glm::vec2));

}

void Solver::cp2GPU() {
  cudaMemcpy(cudaDevicePosition,      sim->x, N_POINTS, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceVelocity,      sim->v, N_POINTS, cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceAcceleration,  sim->f, N_POINTS, cudaMemcpyHostToDevice);
}

void Solver::cp2CPU() {
  cudaMemcpy(sim->x, cudaDevicePosition, N_POINTS, cudaMemcpyDeviceToHost);
}

void Solver::solve() {
  std::cout << "Compiled successfully" << std::endl;
}

