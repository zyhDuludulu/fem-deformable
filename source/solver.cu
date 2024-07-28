#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/vec2.hpp>
#include <cstdio>
#include <string>
#include "fem/parameters.h"
#include "fem/simulator.h"
#include "fem/solver.h"
#include "glm/exponential.hpp"
#include "glm/ext/matrix_float2x2.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/matrix.hpp"

struct Constants {
  glm::vec3 *position;
  glm::vec3 *velocity;
  glm::vec3 *force;
  glm::ivec4 *quads;
  float *A;
  glm::mat3 *DmInv;
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
  cudaMalloc(&cudaDeviceForce, N_POINTS * sizeof(glm::vec2));
  cudaMalloc(&cudaDeviceQuads,    N_QUADS * sizeof(glm::ivec3));
  cp2GPU();

  cudaMalloc(&cudaDeviceA,            N_QUADS * sizeof(float));
  cudaMalloc(&cudaDeviceDmInv,        N_QUADS * sizeof(glm::mat2));

  Constants constants;
  constants.position = cudaDevicePosition;
  constants.velocity = cudaDeviceVelocity;
  constants.force = cudaDeviceForce;
  constants.quads = cudaDeviceQuads;
  constants.DmInv = cudaDeviceDmInv;
  constants.A = cudaDeviceA;

  cudaMemcpyToSymbol(cuConstants, &constants, sizeof(Constants));

}

void Solver::cp2GPU() {
  cudaMemcpy(cudaDevicePosition,      sim->x,     N_POINTS * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceVelocity,      sim->v,     N_POINTS * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceForce,         sim->f,     N_POINTS * sizeof(glm::vec2), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceQuads, sim->quads, N_QUADS * sizeof(glm::ivec3), cudaMemcpyHostToDevice);
}

void Solver::cp2CPU() {
  cudaMemcpy(sim->x, cudaDevicePosition, N_POINTS * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
}

__global__
void kernelComputeDmInv() {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int a = cuConstants.quads[i].x;
  int b = cuConstants.quads[i].y;
  int c = cuConstants.quads[i].z;
  int d = cuConstants.quads[i].w;
  glm::mat3 Dm = glm::mat3(cuConstants.position[b] - cuConstants.position[a], cuConstants.position[c] - cuConstants.position[a], cuConstants.position[d] - cuConstants.position[a]);
  cuConstants.DmInv[i] = glm::inverse(Dm);
  cuConstants.A[i] = 0.5f * glm::abs(glm::determinant(Dm));
}

void Solver::computeDmInv() {
  kernelComputeDmInv<<<N_QUADS, 1>>>();
  cudaDeviceSynchronize();
}

__global__
void kernelComputeForces() {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int a = cuConstants.quads[i].x;
  int b = cuConstants.quads[i].y;
  int c = cuConstants.quads[i].z;
  glm::mat3 Ds = glm::mat2(cuConstants.position[b] - cuConstants.position[a], cuConstants.position[c] - cuConstants.position[a]);
  glm::mat3 F = Ds * cuConstants.DmInv[i];
  glm::mat3 E = 0.5f * (glm::transpose(F) * F - glm::identity<glm::mat3>());
  glm::mat3 P = F * (2.0f * params::mu * E);
  glm::mat3 grad = glm::transpose(cuConstants.A[i] * P * glm::transpose(cuConstants.DmInv[i]));
  atomicAdd(&cuConstants.force[b].x, grad[0][0]);
  atomicAdd(&cuConstants.force[b].y, grad[1][0]);
  atomicAdd(&cuConstants.force[c].x, grad[0][1]);
  atomicAdd(&cuConstants.force[c].y, grad[1][1]);
  atomicAdd(&cuConstants.force[a].x, -grad[0][0] - grad[0][1]);
  atomicAdd(&cuConstants.force[a].y, -grad[1][0] - grad[1][1]);
}

__global__
void kernelUpdatePosition() {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  glm::vec3 acc = -cuConstants.force[i] - glm::vec3(0.0f, params::g, 0.0f);
  cuConstants.velocity[i] += acc * params::dt;
  cuConstants.position[i] += cuConstants.velocity[i] * params::dt;
  cuConstants.velocity[i] *= glm::exp(-params::dt * params::damping);
  cuConstants.force[i] = glm::vec3(0.0f, 0.0f, 0.0f);

  if (cuConstants.position[i].y < -1.0f) {
    cuConstants.position[i].y = -1.0f;
    cuConstants.velocity[i].y = 0.0f;
    cuConstants.velocity[i].x *= 0.9f;
  }
}

void Solver::solve() {
  for (int i = 0; i < params::sub_steps; i++) {
    kernelComputeForces<<<N_QUADS, 1>>>();
    cudaDeviceSynchronize();
    kernelUpdatePosition<<<N_POINTS, 1>>>();
    cudaDeviceSynchronize();
  }
  cp2CPU();
}

