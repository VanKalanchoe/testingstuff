#version 450 core

// Uniform block (equivalent to the constant buffer in HLSL)
layout(set = 1, binding = 0) uniform UniformBlock
{
    mat4 MatrixTransform;
};

// Input variables (equivalent to the HLSL Input structure)
layout(location = 0) in vec4 Position;    // TEXCOORD0
layout(location = 1) in vec2 TexCoord;    // TEXCOORD1

// Output variables (equivalent to the HLSL Output structure)
layout(location = 0) out vec2 FragTexCoord;   // TEXCOORD0
layout(location = 1) out vec4 FragPosition;    // SV_Position

// Main function
void main()
{
    // Pass TexCoord to the output
    FragTexCoord = TexCoord;
    
    // Apply the transformation matrix to the position
    FragPosition = MatrixTransform * Position;
    
    // Set the output position for the fragment shader
    gl_Position = FragPosition;
}