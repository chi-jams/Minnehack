#version 410 core

in vec2 vTexCoord;

uniform sampler2D tex2D;
uniform bool showCircle;

out vec4 fragColor;

void main() {
  if (showCircle) {
#if 1
    // discard fragment outside circle
    if (distance(vTexCoord, vec2(0.5, 0.5)) > 0.5) {
      discard;
    }
    else {
      fragColor = texture(tex2D, vTexCoord);
    }
#else

   // Answer to home work!
    
   #define M_PI 3.1415926535897932384626433832795
   
   float r = distance(vTexCoord, vec2(0.5, 0.5));
   if (sin(16*M_PI*r) < 0.0) {
      discard;
   }
   else {
      fragColor = texture(tex2D, vTexCoord);
   }
#endif
  }
  else {
     //fragColor = texture(tex2D, vTexCoord);
     fragColor = vec4(1, 0, 0, 1);
  }
}

