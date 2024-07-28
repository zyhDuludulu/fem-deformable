#include "fem/simulator.h"
#include "fem/parameters.h"

#define INDEX(i, j, k) ((i) * params::n_y * params::n_z + (j) * params::n_z + (k))

void Simulator::createMesh() {
  // init the points
  for (int i = 0; i < params::n_x; i++) {
    for (int j = 0; j < params::n_y; j++) {
      for (int k = 0; k < params::n_z; k++) {
        int idx = INDEX(i, j, k);
        x[idx] = glm::vec3((i - params::n_x / 2.0) * params::dx, (j - params::n_y / 2.0) * params::dx, (k - params::n_z / 2.0) * params::dx);
        v[idx] = glm::vec3(0.0f, 0.0f, 0.0f);
        f[idx] = glm::vec3(0.0f, 0.0f, 0.0f);
      }
    }
  }

  // init the quads
  for (int i = 0; i < params::n_x - 1; i++) {
    for (int j = 0; j < params::n_y - 1; j++) {
      for (int k = 0; k < params::n_z - 1; k++) {
        int idx = (i * (params::n_y - 1) * (params::n_z - 1) + j * (params::n_z - 1) + k) * params::quads_per_cube;
        quads[idx] = glm::ivec4(INDEX(i, j, k), INDEX(i + 1, j, k), INDEX(i, j + 1, k), INDEX(i, j, k + 1));
        quads[idx + 1] = glm::ivec4(INDEX(i + 1, j + 1, k), INDEX(i, j + 1, k), INDEX(i + 1, j, k), INDEX(i + 1, j + 1, k + 1));
        quads[idx + 2] = glm::ivec4(INDEX(i + 1, j, k + 1), INDEX(i + 1, j + 1, k + 1), INDEX(i, j, k + 1), INDEX(i + 1, j, k));
        quads[idx + 3] = glm::ivec4(INDEX(i, j + 1, k + 1), INDEX(i, j, k + 1), INDEX(i + 1, j + 1, k + 1), INDEX(i, j + 1, k));
      }
    }
  }
}