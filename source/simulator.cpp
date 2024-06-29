#include "fem/simulator.h"

#define INDEX(i, j) ((i) * params::n_y + (j))

void Simulator::createMesh() {
  // init the points
  for (int i = 0; i < params::n_x; i++) {
    for (int j = 0; j < params::n_y; j++) {
      int idx = INDEX(i, j);
      x[idx] = glm::vec2((i - int(params::n_x / 2)) * params::dx, (j - int(params::n_y / 2)) * params::dx);
      v[idx] = glm::vec2(0.0f, 0.0f);
      f[idx] = glm::vec2(0.0f, 0.0f);
    }
  }

  // init the triangles
  for (int i = 0; i < params::n_x - 1; i++) {
    for (int j = 0; j < params::n_y - 1; j++) {
      int idx = 2 * (i * (params::n_y - 1) + j);
      triangles[idx] = glm::vec3(INDEX(i, j), INDEX(i + 1, j), INDEX(i, j + 1));
      triangles[idx + 1] = glm::vec3(INDEX(i, j + 1), INDEX(i + 1, j + 1), INDEX(i + 1, j));
    }
  }
}