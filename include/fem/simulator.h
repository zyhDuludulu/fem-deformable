#ifndef FEM_SIMULATOR_H
#define FEM_SIMULATOR_H

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat2x2.hpp>
#include "glm/fwd.hpp"
#include "parameters.h"

#define N_POINTS (params::n_x * params::n_y * params::n_z)
#define N_QUADS (params::quads_per_cube * (params::n_x - 1) * (params::n_y - 1) * (params::n_z - 1))

class Simulator {
public:
  Simulator() = default;
  ~Simulator() = default;
  void createMesh();
  void step();

public:
  glm::vec3 x[N_POINTS];
  glm::vec3 v[N_POINTS];
  glm::vec3 f[N_POINTS];

  float A         [N_QUADS];
  glm::ivec4 quads [N_QUADS];
  glm::mat3 Dm_inv    [N_QUADS];
};

#endif // FEM_SIMULATOR_H