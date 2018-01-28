
#version 410 core

layout(location = 0) in vec3 aVert;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform float uTheta;

out vec2 vTexCoord;

void main() {
  // transform vertex
  gl_Position = uPMatrix * uMVMatrix * vec4(aVert, 1.0); 
  // set texture coord
  vTexCoord = aVert.xy + vec2(0.5, 0.5);
}

