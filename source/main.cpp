#include <iostream>
#include <glm/vec2.hpp>
#include "fem/simulator.h"

int main() {
  std::cout << "Hello, World!" << std::endl;
  glm::vec2 v;
  v.x = 1.0f;
  v.y = 2.0f;
  std::cout << v.x << " " << v.y << std::endl;
  Simulator sim;
}