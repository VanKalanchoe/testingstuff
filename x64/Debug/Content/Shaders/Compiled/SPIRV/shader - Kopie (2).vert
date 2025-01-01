#version 450

// Input structure from vertex data
layout(location = 0) in vec3 inPosition;  // Input vertex position
layout(location = 1) in vec2 inTexCoord;  // Input texture coordinates 

// Output to fragment shader
layout(location = 0) out vec2 TexCoord;   // Pass texture coordinates to the fragment shader

void main()
{
    // Set the output texture coordinate
    TexCoord = inTexCoord;

    // Set the vertex position (homogeneous coordinates)
    gl_Position = vec4(inPosition, 1.0);
}