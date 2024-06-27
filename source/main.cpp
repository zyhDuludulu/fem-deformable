#include <iostream>
#include <glm/vec2.hpp>
#include "fem/simulator.h"

int main() {
  Simulator sim;
  sim.init();
  while (true) {
    for (int i = 0; i < 100; i++) {
      // sim.step();
    }
  }
}