#ifndef FEM_SIMULATOR_H
#define FEM_SIMULATOR_H

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat2x2.hpp>
#include "glm/fwd.hpp"
#include "parameters.h"

#define N_POINTS (params::n_x) * (params::n_y)
#define N_TRIANGLES 2 * (params::n_x - 1) * (params::n_y - 1)

class Simulator {
public:
  Simulator() = default;
  ~Simulator() = default;
  void init();
  void createMesh();
  void step();

public:
  int n_points;
  int n_triangles;
  float mu;
  float lambda;
  glm::vec2 x[N_POINTS];
  glm::vec2 v[N_POINTS];
  glm::vec2 f[N_POINTS];

  float A         [N_TRIANGLES];
  glm::ivec3 triangles [N_TRIANGLES];
  glm::mat2 Dm_inv    [N_TRIANGLES];
};

#endif // FEM_SIMULATOR_H