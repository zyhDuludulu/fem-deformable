#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/vec2.hpp>
#include <iostream>
#include "fem/parameters.h"
#include "fem/simulator.h"
#include "fem/solver.h"
#include "glm/ext/matrix_float2x2.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/matrix.hpp"

struct Constants {
  float dt;
  float g;
  float mu;
  float lambda;

  glm::vec2 *position;
  glm::vec2 *velocity;
  glm::vec2 *acceleration;
  glm::ivec3 *triangles;
  glm::mat2 *A;
  glm::mat2 *DmInv;
};

__constant__ Constants cuConstants;

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
  cudaMalloc(&cudaDeviceTriangles,    N_TRIANGLES * sizeof(glm::ivec3));
  cp2GPU();

  cudaMalloc(&cudaDeviceA,            N_TRIANGLES * sizeof(glm::mat2));
  cudaMalloc(&cudaDeviceDmInv,        N_TRIANGLES * sizeof(glm::mat2));

  Constants constants;
  constants.dt = params::dt;
  constants.g = params::g;
  constants.mu = params::mu;
  constants.lambda = params::lambda;
  constants.position = sim->x;
  constants.velocity = sim->v;
  constants.acceleration = sim->f;
  constants.triangles = sim->triangles;
  constants.A = sim->A;

  cudaMemcpyToSymbol(cuConstants, &constants, sizeof(Constants));

}

void Solver::cp2GPU() {
  cudaMemcpy(cudaDevicePosition,      sim->x,     N_POINTS * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceVelocity,      sim->v,     N_POINTS * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceAcceleration,  sim->f,     N_POINTS * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceTriangles, sim->triangles, N_TRIANGLES * sizeof(glm::ivec3), cudaMemcpyHostToDevice);
}

void Solver::cp2CPU() {
  cudaMemcpy(sim->x, cudaDevicePosition, N_POINTS, cudaMemcpyDeviceToHost);
}

__global__
void kernelComputeDmInv() {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int a = cuConstants.triangles[i].x;
  int b = cuConstants.triangles[i].y;
  int c = cuConstants.triangles[i].z;
  glm::mat2 Dm = glm::mat2(cuConstants.position[b] - cuConstants.position[a], cuConstants.position[c] - cuConstants.position[a]);
  cuConstants.DmInv[i] = glm::inverse(Dm);
  cuConstants.A[i] = 0.5f * glm::determinant(Dm);
}

void Solver::computeDmInv() {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N_TRIANGLES + threadsPerBlock - 1) / threadsPerBlock;
  kernelComputeDmInv<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();
}

void Solver::solve() {
  std::cout << "Compiled successfully" << std::endl;
}

