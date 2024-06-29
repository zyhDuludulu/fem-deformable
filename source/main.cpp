#include <glm/vec2.hpp>
#include <iostream>
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
  while (!renderer.shouldClose()) {
    solver.solve();
    renderer.render();
    glfwPollEvents();
  }
  std::cout << "program finished" << std::endl;
}