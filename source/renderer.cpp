#include "fem/renderer.h" 
#include "LLGL/Buffer.h"
#include "LLGL/BufferFlags.h"
#include "LLGL/CommandBufferFlags.h"
#include "LLGL/Format.h"
#include "LLGL/PipelineLayout.h"
#include "LLGL/PipelineLayoutFlags.h"
#include "LLGL/ResourceHeapFlags.h"
#include "LLGL/Shader.h"
#include "LLGL/ShaderFlags.h"
#include "LLGL/TypeInfo.h"
#include "LLGL/Utils/TypeNames.h"
#include "LLGL/Utils/VertexFormat.h"
#include "LLGL/Window.h"
#include "fem/parameters.h"
#include "fem/simulator.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_float3.hpp"

#define INDEX(i, j, k) ((i) * params::n_y * params::n_z + (j) * params::n_z + (k))

struct SceneState {
  glm::mat4 mvp;
}
sceneState;

Renderer::~Renderer() {
  // glfwDestroyWindow(window);
  // glfwTerminate();
}

void Renderer::init(Simulator* simulator) {
  this->simulator = simulator;
  // init mesh
  int base = 0;
  for (int i = 0; i < params::n_x - 1; i++) {
    for (int j = 0; j < params::n_y - 1; j++) {
      int idx = base + (i * (params::n_y - 1) + j) * 4 * 3;
      traingles[idx] = INDEX(i, j, 0);
      traingles[idx + 1] = INDEX(i + 1, j, 0);
      traingles[idx + 2] = INDEX(i, j + 1, 0);

      idx += 3;
      traingles[idx] = INDEX(i, j + 1, 0);
      traingles[idx + 1] = INDEX(i + 1, j + 1, 0);
      traingles[idx + 2] = INDEX(i + 1, j, 0);

      idx += 3;
      traingles[idx] = INDEX(i, j, params::n_z - 1);
      traingles[idx + 1] = INDEX(i, j + 1, params::n_z - 1);
      traingles[idx + 2] = INDEX(i + 1, j, params::n_z - 1);

      idx += 3;
      traingles[idx] = INDEX(i + 1, j, params::n_z - 1);
      traingles[idx + 1] = INDEX(i, j + 1, params::n_z - 1);
      traingles[idx + 2] = INDEX(i + 1, j + 1, params::n_z - 1);
    }
  }

  base += (params::n_x - 1) * (params::n_y - 1) * 4 * 3;
  for (int j = 0; j < params::n_y - 1; j++) {
    for (int k = 0; k < params::n_z - 1; k++) {
      int idx = base + (j * (params::n_z - 1) + k) * 4 * 3;
      traingles[idx] = INDEX(0, j, k);
      traingles[idx + 1] = INDEX(0, j + 1, k);
      traingles[idx + 2] = INDEX(0, j, k + 1);

      idx += 3;
      traingles[idx] = INDEX(0, j, k + 1);
      traingles[idx + 1] = INDEX(0, j + 1, k);
      traingles[idx + 2] = INDEX(0, j + 1, k + 1);

      idx += 3;
      traingles[idx] = INDEX(params::n_x - 1, j, k);
      traingles[idx + 1] = INDEX(params::n_x - 1, j, k + 1);
      traingles[idx + 2] = INDEX(params::n_x - 1, j + 1, k);

      idx += 3;
      traingles[idx] = INDEX(params::n_x - 1, j + 1, k);
      traingles[idx + 1] = INDEX(params::n_x - 1, j, k + 1);
      traingles[idx + 2] = INDEX(params::n_x - 1, j + 1, k + 1);
    }
  }

  base += (params::n_y - 1) * (params::n_z - 1) * 4 * 3;
  for (int i = 0; i < params::n_x - 1; i++) {
    for (int k = 0; k < params::n_z - 1; k++) {
      int idx = base + (i * (params::n_z - 1) + k) * 4 * 3;
      traingles[idx] = INDEX(i, 0, k);
      traingles[idx + 1] = INDEX(i + 1, 0, k);
      traingles[idx + 2] = INDEX(i, 0, k + 1);

      idx += 3;
      traingles[idx] = INDEX(i, 0, k + 1);
      traingles[idx + 1] = INDEX(i + 1, 0, k);
      traingles[idx + 2] = INDEX(i + 1, 0, k + 1);

      idx += 3;
      traingles[idx] = INDEX(i, params::n_y - 1, k);
      traingles[idx + 1] = INDEX(i, params::n_y - 1, k + 1);
      traingles[idx + 2] = INDEX(i + 1, params::n_y - 1, k);

      idx += 3;
      traingles[idx] = INDEX(i + 1, params::n_y - 1, k);
      traingles[idx + 1] = INDEX(i, params::n_y - 1, k + 1);
      traingles[idx + 2] = INDEX(i + 1, params::n_y - 1, k + 1);
    }
  }
  initWindow();
}

void Renderer::initWindow() {
  LLGL::Report report;
  this->myRdr = LLGL::RenderSystem::Load("OpenGL", &report);
  std::cout << "Render System Info: " << report.GetText() << std::endl;
  const auto& info = myRdr->GetRendererInfo();

  LLGL::SwapChainDescriptor mySwapChainDesc;
  mySwapChainDesc.resolution = {800, 600};
  this->mySwapChain = myRdr->CreateSwapChain(mySwapChainDesc);
  
  printf("---------------------------------------------------------\n"
      "Renderer:             %s\n"
      "Device:               %s\n"
      "Vendor:               %s\n"
      "Shading Language:     %s\n"
      "Swap Chain Format:    %s\n"
      "Depth/Stencil Format: %s\n",
      info.rendererName.c_str(),
      info.deviceName.c_str(),
      info.vendorName.c_str(),
      info.shadingLanguageName.c_str(),
      LLGL::ToString(mySwapChain->GetColorFormat()),
      LLGL::ToString(mySwapChain->GetDepthStencilFormat())
  );

  this->window = &LLGL::CastTo<LLGL::Window>(mySwapChain->GetSurface());
  window->SetTitle("jelly");
  window->Show();

  // vertex buffer
  LLGL::VertexFormat vertexFormat;
  vertexFormat.AppendAttribute({ "position", LLGL::Format::RGB32Float });
  vertexFormat.SetStride(sizeof(glm::vec3));
  LLGL::BufferDescriptor myVertexBufferDesc;
  myVertexBufferDesc.size = sizeof(glm::vec3) * N_POINTS;
  myVertexBufferDesc.bindFlags = LLGL::BindFlags::VertexBuffer;
  myVertexBufferDesc.vertexAttribs = vertexFormat.attributes;
  this->myVertexBuffer = myRdr->CreateBuffer(myVertexBufferDesc, simulator->x);

  LLGL::BufferDescriptor myIndexBufferDesc;
  myIndexBufferDesc.size = sizeof(this->traingles);
  myIndexBufferDesc.bindFlags = LLGL::BindFlags::IndexBuffer;
  myIndexBufferDesc.format = LLGL::Format::R32UInt;
  this->myIndexBuffer = myRdr->CreateBuffer(myIndexBufferDesc, this->traingles);

  // shader
  LLGL::ShaderDescriptor vertexShaderDesc;
  vertexShaderDesc.type = LLGL::ShaderType::Vertex;
  vertexShaderDesc.source = "../source/vertex.glsl";
  vertexShaderDesc.sourceType = LLGL::ShaderSourceType::CodeFile;
  vertexShaderDesc.vertex.inputAttribs = vertexFormat.attributes;

  this->vertexShader = myRdr->CreateShader(vertexShaderDesc);
  this->fragmentShader = myRdr->CreateShader({LLGL::ShaderType::Fragment, "../source/fragment.glsl"});
  for (LLGL::Shader *shader : {vertexShader, fragmentShader}) {
    if (const LLGL::Report* report = shader->GetReport())
      std::cout << report->GetText() << std::endl;
  }

  LLGL::GraphicsPipelineDescriptor myPSODesc;
  myPSODesc.vertexShader = this->vertexShader;
  myPSODesc.fragmentShader = this->fragmentShader;
  myPSODesc.renderPass = this->mySwapChain->GetRenderPass();
  this->myPSO = myRdr->CreatePipelineState(myPSODesc);

  this->myCmdBuffer = myRdr->CreateCommandBuffer(LLGL::CommandBufferFlags::ImmediateSubmit);

  LLGL::BufferDescriptor constantBufferDesc;
  constantBufferDesc.size = sizeof(SceneState);
  constantBufferDesc.bindFlags = LLGL::BindFlags::ConstantBuffer;

  this->myConstantBuffer = myRdr->CreateBuffer(constantBufferDesc, &sceneState);
  
  LLGL::UniformDescriptor uniformDesc;
  uniformDesc.type = LLGL::UniformType::Float4x4;
  uniformDesc.arraySize = 1;
  LLGL::PipelineLayoutDescriptor pipelineLayoutDesc;
  pipelineLayoutDesc.uniforms = {uniformDesc};
  LLGL::PipelineLayout* layout = myRdr->CreatePipelineLayout(pipelineLayoutDesc);
  LLGL::ResourceViewDescriptor resourceViewDesc[] = {myConstantBuffer,};
  LLGL::ResourceHeapDescriptor resourceHeapDesc;
  resourceHeapDesc.pipelineLayout = layout;
  resourceHeapDesc.numResourceViews = sizeof(resourceViewDesc) / sizeof(LLGL::ResourceViewDescriptor);
  this->myResourceHeap = myRdr->CreateResourceHeap(resourceHeapDesc, resourceViewDesc);
}

bool Renderer::shouldClose() const {
  return !this->window->HasQuit();
}

void Renderer::render() {
  glm::mat4 model = glm::mat4(1.0f);
  glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
  glm::mat4 mvp = projection * view * model;
  sceneState.mvp = mvp;
  // std::cout << mvp[0][0] << std::endl;
  // vertexShader->SetConstantBuffer(0, &mvp, sizeof(mvp));
  myCmdBuffer->Begin();

  myCmdBuffer->UpdateBuffer(*myConstantBuffer, 0, &sceneState, sizeof(sceneState));
  myCmdBuffer->SetViewport(mySwapChain->GetResolution());
  myCmdBuffer->SetVertexBuffer(*myVertexBuffer);
  myCmdBuffer->SetIndexBuffer(*myIndexBuffer);
  myCmdBuffer->BeginRenderPass(*mySwapChain);

  myCmdBuffer->Clear(LLGL::ClearFlags::Color);
  myCmdBuffer->SetPipelineState(*myPSO);
  myCmdBuffer->SetResourceHeap(*myResourceHeap);
  myCmdBuffer->DrawIndexed(3 * N_TRIANGLES, 0);

  myCmdBuffer->EndRenderPass();
  myCmdBuffer->End();
  mySwapChain->Present();
}
