#include <glm/vec2.hpp>
#include "parameters.h"

#define N_EDGES (params::n_x - 1) * params::n_y + (params::n_y - 1) * params::n_x + (params::n_x - 1) * (params::n_y - 1)

class Renderer {
public:
  Renderer() = default;
  ~Renderer() = default;
  void init();
  void render();
private:
  glm::vec2 edges[N_EDGES];
};