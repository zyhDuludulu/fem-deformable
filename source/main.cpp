#include <glm/vec2.hpp>
#include <iostream>
#include "LLGL/Surface.h"
#include "fem/simulator.h"
#include "fem/solver.h"
#include "fem/renderer.h"

int main() {
  Simulator   simulator;
  Solver      solver;
  Renderer    renderer;

  simulator.createMesh();
  renderer.init(&simulator);
  solver.setUp(&simulator);
  solver.computeDmInv();
  while (LLGL::Surface::ProcessEvents() && renderer.shouldClose()) {
    // solver.solve();
    renderer.render();
  }
  std::cout << "program finished" << std::endl;
}