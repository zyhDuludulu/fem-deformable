class Solver {
public:
  Solver() = default;
  ~Solver() = default;
  void solve();

private:
  glm::vec2* cudaDevicePosition;
  glm::vec2* cudaDeviceVelocity;
  glm::vec2* cudaDeviceAcceleration;
  int* cudaDeviceBlockId;
}