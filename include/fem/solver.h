#include <glm/vec2.hpp>
#include "simulator.h"

class Solver {
public:
  Solver() = default;
  ~Solver() = default;
  void setUp(Simulator *sim);
  void solve();

private:
  void cp2CPU();
  void cp2GPU();
  int *cudaDeviceBlockId;
  Simulator *sim;

  glm::vec2 *cudaDevicePosition;
  glm::vec2 *cudaDeviceVelocity;
  glm::vec2 *cudaDeviceAcceleration;
};