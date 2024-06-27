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
  renderer.init();
  solver.setUp(&simulator);
  while (true) {
    for (int i = 0; i < 100; i++) {
      // sim.step();
    }
    break;
  }
  std::cout << "program finished" << std::endl;
}