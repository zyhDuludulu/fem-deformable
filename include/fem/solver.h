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

  glm::vec3 *cudaDevicePosition;
  glm::vec3 *cudaDeviceVelocity;
  glm::vec3 *cudaDeviceForce;

  /* the following data can be made constant */
  glm::ivec4 *cudaDeviceQuads;
  glm::mat3 *cudaDeviceDmInv;
  float *cudaDeviceA;
};