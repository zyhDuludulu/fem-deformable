#include "fem/renderer.h"

#define INDEX(i, j) ((i) * params::n_y + (j))

void Renderer::init() {
  int base = 0;
  for (int i = 0; i < params::n_x-1; i++) {
    for (int j = 0; j < params::n_y; j++) {
      int idx = base + i * params::n_y + j;
      edges[idx] = glm::vec2(INDEX(i, j), INDEX(i+1, j));
    }
  }

  base += (params::n_x - 1) * params::n_y;
  for (int i = 0; i < params::n_x; i++) {
    for (int j = 0; j < params::n_y-1; j++) {
      int idx = base + i * (params::n_y - 1) + j;
      edges[idx] = glm::vec2(INDEX(i, j), INDEX(i, j+1));
    }
  }

  base += (params::n_y - 1) * params::n_x;
  for (int i = 0; i < params::n_x-1; i++) {
    for (int j = 0; j < params::n_y-1; j++) {
      int idx = base + i * (params::n_y - 1) + j;
      edges[idx] = glm::vec2(INDEX(i, j), INDEX(i+1, j+1));
    }
  }
}