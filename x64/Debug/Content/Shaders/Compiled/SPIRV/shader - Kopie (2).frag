#version 450

layout(set = 2, binding = 0) uniform texture2D Texture; // Texture at set 2, binding 0
layout(set = 2, binding = 0) uniform sampler Sampler;   // Sampler at set 2, binding 1

layout(location = 0) in vec2 TexCoord; // Input texture coordinates
layout(location = 0) out vec4 FragColor; // Output color

void main()
{
    FragColor = texture(sampler2D(Texture, Sampler), TexCoord); // Sample the texture
}