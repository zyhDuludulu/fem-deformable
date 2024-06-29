#include "simulator.h"

class Solver {
public:
  Solver() = default;
  ~Solver() = default;
  void setUp(Simulator *sim);
  void computeDmInv();
  void solve();

private:
  void cp2CPU();
  void cp2GPU();
  int *cudaDeviceBlockId;
  Simulator *sim;

  glm::vec2 *cudaDevicePosition;
  glm::vec2 *cudaDeviceVelocity;
  glm::vec2 *cudaDeviceForce;

  /* the following data can be made constant */
  glm::ivec3 *cudaDeviceTriangles;
  glm::mat2 *cudaDeviceDmInv;
  float *cudaDeviceA;
};