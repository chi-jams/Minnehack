
#version 410 core

in vec2 vTexCoord;

out vec4 fragColor;

uniform sampler2D tex2D;

void main() {
    fragColor = texture(tex2D, 0.5 * (vTexCoord + vec2(0.5, 0.5)));
}
