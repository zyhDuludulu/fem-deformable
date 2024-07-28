#include <glm/vec2.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "LLGL/RenderSystem.h"
#include "LLGL/ResourceHeap.h"
#include "fem/simulator.h"
#include "parameters.h"
#include "LLGL/LLGL.h"

#define N_TRIANGLES (4 * ((params::n_x - 1) * (params::n_y-1) + (params::n_y - 1) * (params::n_x-1) + (params::n_x - 1) * (params::n_y-1)))

class Renderer {
public:
  Renderer() = default;
  ~Renderer();
  void init(Simulator* simulator);
  void render();
  bool shouldClose() const;
private:
  void initWindow();

  const char* vertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec2 aPos;
    void main()
    {
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    )";

  const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    void main()
    {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    )";
  unsigned int vao, vbo, ebo, shaderProgram;
  LLGL::Window* window;
  LLGL::RenderSystemPtr myRdr;
  LLGL::SwapChain* mySwapChain;
  LLGL::Buffer* myVertexBuffer;
  LLGL::Buffer* myIndexBuffer;
  LLGL::Buffer* myConstantBuffer;
  LLGL::Shader* vertexShader;
  LLGL::Shader* fragmentShader;
  LLGL::PipelineState* myPSO;
  LLGL::CommandBuffer* myCmdBuffer;
  LLGL::ResourceHeap* myResourceHeap;
  Simulator* simulator;
  int traingles[N_TRIANGLES * 3];
};