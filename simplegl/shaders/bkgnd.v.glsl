
#version 410 core

layout (location = 0) in vec3 vert;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(2 * vert, 1.0);

    vTexCoord = vert.xy;
}
