#include "fem/renderer.h" 
#include "fem/simulator.h"
#include "glm/ext/vector_float2.hpp"

#define INDEX(i, j) ((i) * params::n_y + (j))

glm::vec2 v[] = {
    {0.0f, 0.5f},
    {-0.5f, -0.5f},
    {0.5f, -0.5f}
};

// 边数据
glm::ivec2 e[] = {
    {0, 1},
    {1, 2},
    {2, 0}
};

Renderer::~Renderer() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

void Renderer::init(Simulator* simulator) {
  this->simulator = simulator;
  int base = 0;
  for (int i = 0; i < params::n_x-1; i++) {
    for (int j = 0; j < params::n_y; j++) {
      int idx = base + i * params::n_y + j;
      edges[idx] = glm::ivec2(INDEX(i, j), INDEX(i+1, j));
    }
  }

  base += (params::n_x - 1) * params::n_y;
  for (int i = 0; i < params::n_x; i++) {
    for (int j = 0; j < params::n_y-1; j++) {
      int idx = base + i * (params::n_y - 1) + j;
      edges[idx] = glm::ivec2(INDEX(i, j), INDEX(i, j+1));
    }
  }

  base += (params::n_y - 1) * params::n_x;
  for (int i = 0; i < params::n_x-1; i++) {
    for (int j = 0; j < params::n_y-1; j++) {
      int idx = base + i * (params::n_y - 1) + j;
      edges[idx] = glm::ivec2(INDEX(i+1, j), INDEX(i, j+1));
    }
  }

  initWindow();
}

void Renderer::initWindow() {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    exit(EXIT_FAILURE);
  }

  window = glfwCreateWindow(params::window_width, params::window_height, "FEM", NULL, NULL);
  if (!window) {
    std::cerr << "Failed to create window" << std::endl;
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    exit(EXIT_FAILURE);
  }

  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);

  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);

  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(simulator->x), simulator->x, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(edges), edges, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::ivec2), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

bool Renderer::shouldClose() const {
  return glfwWindowShouldClose(window) || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
}

void Renderer::render() {
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(simulator->x), simulator->x, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(shaderProgram);
  glBindVertexArray(vao);
  glDrawElements(GL_LINES, N_EDGES * 2, GL_UNSIGNED_INT, 0);

  glfwSwapBuffers(window);
  glfwPollEvents();
}