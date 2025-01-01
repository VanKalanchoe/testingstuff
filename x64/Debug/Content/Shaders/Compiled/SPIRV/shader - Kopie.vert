#version 450

layout(location = 0) out vec3 fragColor;

vec2 positions[4] = vec2[](
    vec2(-0.5, -0.5),  // bottom-left 
    vec2( 0.5, -0.5),  // bottom-right
    vec2( 0.5,  0.5),  // top-right
    vec2(-0.5,  0.5)   // top-left
);

vec3 colors[4] = vec3[](
    vec3(0.0, 0.0, 0.0),  // red
    vec3(0.0, 0.0, 0.0),  // green
    vec3(0.0, 0.0, 0.0),  // blue
    vec3(0.0, 0.0, 0.0)   // yellow
);

void main()
{
    // Define two triangles for the quad using the 4 vertices
    // The first triangle uses vertex indices 0, 1, and 2 (bottom-left, bottom-right, top-right)
    // The second triangle uses vertex indices 2, 3, and 0 (top-right, top-left, bottom-left)
    vec2 indexed_positions[6] = vec2[](
        positions[0], positions[1], positions[2], // First triangle
        positions[2], positions[3], positions[0]  // Second triangle
    );

    vec3 indexed_colors[6] = vec3[](
        colors[0], colors[1], colors[2], // First triangle colors
        colors[2], colors[3], colors[0]  // Second triangle colors
    );

    gl_Position = vec4(indexed_positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = indexed_colors[gl_VertexIndex];
}
