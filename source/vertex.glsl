#version 450 core

in vec3 position;
// in vec4 color;
layout(std140, row_major, binding = 3) uniform SceneState {
    mat4 mvp;
};

out vec4 vertexColor;

void main() {
    gl_Position = vec4(position, 1);// * mvp;
    vertexColor = vec4(1.0, 1.0, 1.0, 1.0);
    float color = mvp[0][0]+0.5;
    vertexColor = vec4(color, color, color, 1.0);
}
