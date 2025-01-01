/*#include <algorithm>
#include <format>
#include <iostream>
#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>
#include <SDL3_ttf/SDL_ttf.h>
static const char* SamplerNames[] =
{
    "PointClamp",
    "PointWrap",
    "LinearClamp",
    "LinearWrap",
    "AnisotropicClamp",
    "AnisotropicWrap",
};
static float t = 0;
// Matrix Math
typedef struct Matrix4x4
{
    float m11, m12, m13, m14;
    float m21, m22, m23, m24;
    float m31, m32, m33, m34;
    float m41, m42, m43, m44;
} Matrix4x4;
// Matrix Math

Matrix4x4 Matrix4x4_Multiply(Matrix4x4 matrix1, Matrix4x4 matrix2)
{
	Matrix4x4 result;

	result.m11 = (
		(matrix1.m11 * matrix2.m11) +
		(matrix1.m12 * matrix2.m21) +
		(matrix1.m13 * matrix2.m31) +
		(matrix1.m14 * matrix2.m41)
	);
	result.m12 = (
		(matrix1.m11 * matrix2.m12) +
		(matrix1.m12 * matrix2.m22) +
		(matrix1.m13 * matrix2.m32) +
		(matrix1.m14 * matrix2.m42)
	);
	result.m13 = (
		(matrix1.m11 * matrix2.m13) +
		(matrix1.m12 * matrix2.m23) +
		(matrix1.m13 * matrix2.m33) +
		(matrix1.m14 * matrix2.m43)
	);
	result.m14 = (
		(matrix1.m11 * matrix2.m14) +
		(matrix1.m12 * matrix2.m24) +
		(matrix1.m13 * matrix2.m34) +
		(matrix1.m14 * matrix2.m44)
	);
	result.m21 = (
		(matrix1.m21 * matrix2.m11) +
		(matrix1.m22 * matrix2.m21) +
		(matrix1.m23 * matrix2.m31) +
		(matrix1.m24 * matrix2.m41)
	);
	result.m22 = (
		(matrix1.m21 * matrix2.m12) +
		(matrix1.m22 * matrix2.m22) +
		(matrix1.m23 * matrix2.m32) +
		(matrix1.m24 * matrix2.m42)
	);
	result.m23 = (
		(matrix1.m21 * matrix2.m13) +
		(matrix1.m22 * matrix2.m23) +
		(matrix1.m23 * matrix2.m33) +
		(matrix1.m24 * matrix2.m43)
	);
	result.m24 = (
		(matrix1.m21 * matrix2.m14) +
		(matrix1.m22 * matrix2.m24) +
		(matrix1.m23 * matrix2.m34) +
		(matrix1.m24 * matrix2.m44)
	);
	result.m31 = (
		(matrix1.m31 * matrix2.m11) +
		(matrix1.m32 * matrix2.m21) +
		(matrix1.m33 * matrix2.m31) +
		(matrix1.m34 * matrix2.m41)
	);
	result.m32 = (
		(matrix1.m31 * matrix2.m12) +
		(matrix1.m32 * matrix2.m22) +
		(matrix1.m33 * matrix2.m32) +
		(matrix1.m34 * matrix2.m42)
	);
	result.m33 = (
		(matrix1.m31 * matrix2.m13) +
		(matrix1.m32 * matrix2.m23) +
		(matrix1.m33 * matrix2.m33) +
		(matrix1.m34 * matrix2.m43)
	);
	result.m34 = (
		(matrix1.m31 * matrix2.m14) +
		(matrix1.m32 * matrix2.m24) +
		(matrix1.m33 * matrix2.m34) +
		(matrix1.m34 * matrix2.m44)
	);
	result.m41 = (
		(matrix1.m41 * matrix2.m11) +
		(matrix1.m42 * matrix2.m21) +
		(matrix1.m43 * matrix2.m31) +
		(matrix1.m44 * matrix2.m41)
	);
	result.m42 = (
		(matrix1.m41 * matrix2.m12) +
		(matrix1.m42 * matrix2.m22) +
		(matrix1.m43 * matrix2.m32) +
		(matrix1.m44 * matrix2.m42)
	);
	result.m43 = (
		(matrix1.m41 * matrix2.m13) +
		(matrix1.m42 * matrix2.m23) +
		(matrix1.m43 * matrix2.m33) +
		(matrix1.m44 * matrix2.m43)
	);
	result.m44 = (
		(matrix1.m41 * matrix2.m14) +
		(matrix1.m42 * matrix2.m24) +
		(matrix1.m43 * matrix2.m34) +
		(matrix1.m44 * matrix2.m44)
	);

	return result;
}
Matrix4x4 Matrix4x4_CreateRotationZ(float radians)
{
    return Matrix4x4 {
        SDL_cosf(radians), SDL_sinf(radians), 0, 0,
       -SDL_sinf(radians), SDL_cosf(radians), 0, 0,
                        0, 				0, 1, 0,
                        0,					0, 0, 1
   };
}
Matrix4x4 Matrix4x4_CreateTranslation(float x, float y, float z)
{
    return Matrix4x4 {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        x, y, z, 1
    };
}
typedef struct FragMultiplyUniform
{
    float r, g, b, a;
} FragMultiplyUniform;

typedef struct PositionTextureVertex
{
    float x, y, z;
    float u, v;
} PositionTextureVertex;
static SDL_GPUSampler* Samplers[SDL_arraysize(SamplerNames)];
SDL_GPUShader* LoadShader(
    SDL_GPUDevice* device,
    const char* shaderFilename,
    Uint32 samplerCount,
    Uint32 uniformBufferCount,
    Uint32 storageBufferCount,
    Uint32 storageTextureCount
)
{
    // Auto-detect the shader stage from the file name for convenience
    SDL_GPUShaderStage stage;
    if (SDL_strstr(shaderFilename, ".vert"))
    {
        stage = SDL_GPU_SHADERSTAGE_VERTEX;
    }
    else if (SDL_strstr(shaderFilename, ".frag"))
    {
        stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    }
    else
    {
        SDL_Log("Invalid shader stage!");
        return NULL;
    }
    
    char fullPath[256];
    SDL_GPUShaderFormat backendFormats = SDL_GetGPUShaderFormats(device);
    SDL_GPUShaderFormat format;
    const char* entrypoint;

    if (backendFormats & SDL_GPU_SHADERFORMAT_SPIRV)
    {
        SDL_snprintf(fullPath, sizeof(fullPath), "%sContent/Shaders/Compiled/SPIRV/%s.spv", SDL_GetBasePath(),
                     shaderFilename);
        format = SDL_GPU_SHADERFORMAT_SPIRV;
        entrypoint = "main";
    }
    else if (backendFormats & SDL_GPU_SHADERFORMAT_MSL)
    {
        SDL_snprintf(fullPath, sizeof(fullPath), "%sContent/Shaders/Compiled/MSL/%s.msl", SDL_GetBasePath(),
                     shaderFilename);
        format = SDL_GPU_SHADERFORMAT_MSL;
        entrypoint = "main0";
    }
    else if (backendFormats & SDL_GPU_SHADERFORMAT_DXIL)
    {
        SDL_snprintf(fullPath, sizeof(fullPath), "%sContent/Shaders/Compiled/DXIL/%s.dxil", SDL_GetBasePath(),
                     shaderFilename);
        format = SDL_GPU_SHADERFORMAT_DXIL;
        entrypoint = "main";
    }
    else
    {
        SDL_Log("%s", "Unrecognized backend shader format!");
        return NULL;
    }

    size_t codeSize;
    void* code = SDL_LoadFile(fullPath, &codeSize);
    if (code == NULL)
    {
        SDL_Log("Failed to load shader from disk! %s", fullPath);
        return NULL;
    }

    SDL_GPUShaderCreateInfo shaderInfo{
        .code_size = codeSize,
        .code = (const Uint8*)code,
        .entrypoint = entrypoint,
        .format = format,
        .stage = stage,
        .num_samplers = samplerCount,
        .num_storage_textures = storageTextureCount,
        .num_storage_buffers = storageBufferCount,
        .num_uniform_buffers = uniformBufferCount
    };
    
    SDL_GPUShader* shader = SDL_CreateGPUShader(device, &shaderInfo);
    if (shader == NULL)
    {
        SDL_Log("Failed to create shader!");
        SDL_free(code);
        return NULL;
    }

    SDL_free(code);
    return shader;
}
SDL_Surface* LoadImage(const char* imageFilename, int desiredChannels)
{
    char fullPath[256];
    SDL_Surface *result;
    SDL_PixelFormat format;

    SDL_snprintf(fullPath, sizeof(fullPath), "%sContent/Images/%s", SDL_GetBasePath(), imageFilename);

    result = IMG_Load(fullPath);
    if (result == NULL)
    {
        SDL_Log("Failed to load BMP: %s", SDL_GetError());
        return NULL;
    }

    if (desiredChannels == 4)
    {
        format = SDL_PIXELFORMAT_ABGR8888;
    }
    else
    {
        SDL_assert(!"Unexpected desiredChannels");
        SDL_DestroySurface(result);
        return NULL;
    }
    if (result->format != format)
    {
        SDL_Surface *next = SDL_ConvertSurface(result, format);
        SDL_DestroySurface(result);
        result = next;
    }

    return result;
}
SDL_Window* window;
SDL_GPUDevice* device;

bool running = true;

SDL_GPUBuffer* VertexBuffer;

SDL_GPUBuffer* IndexBuffer;

SDL_GPUTexture* Texture;

int CurrentSamplerIndex;

void createWindow()
{
    std::cout << "create Window" << std::endl;

    // Initialize SDL and check for errors
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        // SDL_Init returns non-zero on error
        SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
        return; // Return early if initialization failed
    }

    // Create an application window with the following settings:
    window = SDL_CreateWindow(
        "An SDL3 window", // window title
        640, // width, in pixels
        480, // height, in pixels
        0 // flags - using OpenGL here
    );

    // Check if the window was successfully created
    if (window == nullptr)
    {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s", SDL_GetError());
        SDL_Quit(); // Clean up SDL if window creation failed
        return;
    }

    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, nullptr);
    if (!device)
    {
        SDL_Log("SDL_CreateGPUDevice failed: %s", SDL_GetError());
    }

    std::cout << "Device is rendered with: " << SDL_GetGPUDeviceDriver(device) << std::endl;
    if (!SDL_ClaimWindowForGPUDevice(device, window))
    {
        SDL_Log("SDL_ClaimWindowForGPUDevice failed: %s", SDL_GetError());
    }

    // Create the shaders
    SDL_GPUShader* vertexShader = LoadShader(device, "shader.vert", 0, 1, 0, 0);
    if (vertexShader == NULL)
    {
        SDL_Log("Failed to create vertex shader!");
        return;
    }
std::cout << "Vertex shader loaded successfully!" << std::endl;
    SDL_GPUShader* fragmentShader = LoadShader(device, "shader.frag", 1, 1, 0, 0);
    if (fragmentShader == NULL)
    {
        SDL_Log("Failed to create fragment shader!");
        return;
    }
    std::cout << "Vertex shader loaded successfully!" << std::endl;
    /*SDL_Surface* imageData = IMG_Load("sample.bmp");
    if (imageData == NULL)
    {
        SDL_Log("Could not load image data!");
        return ;
    }#1#
    // Load the image
    SDL_Surface *imageData = LoadImage("ravioli3.bmp", 4);
    if (imageData == NULL)
    {
        SDL_Log("Could not load image data!");
        return ;
    }
    
    SDL_GPUColorTargetDescription color_target_description;
    color_target_description.format = SDL_GetGPUSwapchainTextureFormat(device, window);
    color_target_description.blend_state = {
        .src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .color_blend_op = SDL_GPU_BLENDOP_ADD,
        .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
        .enable_blend = true
    };

    SDL_GPUVertexBufferDescription vertex_buffer_description;
    vertex_buffer_description.slot = 0;
    vertex_buffer_description.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
    vertex_buffer_description.instance_step_rate = 0;
    vertex_buffer_description.pitch = sizeof(PositionTextureVertex);

    SDL_GPUVertexAttribute vertex_attribute;
    vertex_attribute.buffer_slot = 0;
    vertex_attribute.format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
    vertex_attribute.location = 0;
    vertex_attribute.offset = 0;

    SDL_GPUVertexAttribute vertex_attribute2;
    vertex_attribute2.buffer_slot = 0;
    vertex_attribute2.format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
    vertex_attribute2.location = 1;
    vertex_attribute2.offset = sizeof(float) * 3;

    SDL_GPUVertexAttribute vertex_attribute_array[2];
    vertex_attribute_array[0] = vertex_attribute;
    vertex_attribute_array[1] = vertex_attribute2;
    
    SDL_GPUVertexInputState vertex_input_state;
    vertex_input_state.num_vertex_buffers = 1;
    vertex_input_state.vertex_buffer_descriptions = &vertex_buffer_description;
    vertex_input_state.num_vertex_attributes = 2;
    vertex_input_state.vertex_attributes = vertex_attribute_array;
    
    // Create the pipelines
    SDL_GPUGraphicsPipelineCreateInfo pipelineCreateInfo = {
        .vertex_shader = vertexShader,
        .fragment_shader = fragmentShader,
        .vertex_input_state = vertex_input_state,
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .multisample_state = SDL_GPU_SAMPLECOUNT_1,
        .depth_stencil_state = {},
        .target_info = {
            //.color_target_descriptions = &color_target_description,
            .color_target_descriptions = &color_target_description,
            .num_color_targets = 1
        }
    };

    //pipelineCreateInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    SDL_GPUGraphicsPipeline* Pipeline = SDL_CreateGPUGraphicsPipeline(device, &pipelineCreateInfo);
    if (Pipeline == NULL)
    {
        SDL_Log("Failed to create fill pipeline!");
        return;
    }

    // Clean up shader resources
    SDL_ReleaseGPUShader(device, vertexShader);
    SDL_ReleaseGPUShader(device, fragmentShader);
    std::cout << "Vertex shader loaded successfully!sampler" << std::endl;
    SDL_GPUSamplerCreateInfo sampler_create_info = {};
    sampler_create_info.min_filter = SDL_GPU_FILTER_NEAREST;
        sampler_create_info.mag_filter = SDL_GPU_FILTER_NEAREST;
        sampler_create_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
        sampler_create_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
        sampler_create_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
        sampler_create_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    std::cout << "Vertex shader loaded successfully!sampler5" << std::endl;
    // PointClamp
    if (!device) {
        SDL_Log("Error: Device is NULL");
        return;
    }
    if (!Samplers) {
        SDL_Log("Error: Samplers array is NULL");
        return;
    }
    if (device == NULL) {
        SDL_Log("Device is NULL");
        return;
    }
    if (Samplers == NULL) {
        SDL_Log("Samplers array is NULL");
        return;
    }
    
	Samplers[0] = SDL_CreateGPUSampler(device, &sampler_create_info);
    
    std::cout << "Vertex shader loaded successfully!sampler6" << std::endl;
    SDL_GPUSamplerCreateInfo sampler_create_info2= {};
    sampler_create_info2.min_filter = SDL_GPU_FILTER_NEAREST;
        sampler_create_info2.mag_filter = SDL_GPU_FILTER_NEAREST;
        sampler_create_info2.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
        sampler_create_info2.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        sampler_create_info2.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        sampler_create_info2.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    
	// PointWrap
	Samplers[1] = SDL_CreateGPUSampler(device, &sampler_create_info2);

    SDL_GPUSamplerCreateInfo sampler_create_info3= {};
    sampler_create_info3.min_filter = SDL_GPU_FILTER_LINEAR;
            sampler_create_info3.mag_filter = SDL_GPU_FILTER_LINEAR;
            sampler_create_info3.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
            sampler_create_info3.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
            sampler_create_info3.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
            sampler_create_info3.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    
	// LinearClamp
	Samplers[2] = SDL_CreateGPUSampler(device, &sampler_create_info3);

    SDL_GPUSamplerCreateInfo sampler_create_info4= {};
    sampler_create_info4.min_filter = SDL_GPU_FILTER_LINEAR;
            sampler_create_info4.mag_filter = SDL_GPU_FILTER_LINEAR;
            sampler_create_info4.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
            sampler_create_info4.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
            sampler_create_info4.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
            sampler_create_info4.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    
	// LinearWrap
	Samplers[3] = SDL_CreateGPUSampler(device, &sampler_create_info4);
    
    SDL_GPUSamplerCreateInfo sampler_create_info5= {};
    sampler_create_info5.min_filter = SDL_GPU_FILTER_LINEAR;
            sampler_create_info5.mag_filter = SDL_GPU_FILTER_LINEAR;
            sampler_create_info5.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
            sampler_create_info5.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
            sampler_create_info5.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
            sampler_create_info5.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
            sampler_create_info5.enable_anisotropy = true;
            sampler_create_info5.max_anisotropy = 4;
    
	// AnisotropicClamp
	Samplers[4] = SDL_CreateGPUSampler(device, &sampler_create_info5);
    std::cout << "Vertex shader loaded successfully!" << std::endl;
    SDL_GPUSamplerCreateInfo sampler_create_info6= {};
    sampler_create_info6.min_filter = SDL_GPU_FILTER_LINEAR;
            sampler_create_info6.mag_filter = SDL_GPU_FILTER_LINEAR;
            sampler_create_info6.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
            sampler_create_info6.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
            sampler_create_info6.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
            sampler_create_info6.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
            sampler_create_info6.enable_anisotropy = true;
            sampler_create_info6.max_anisotropy = 4;
    
	// AnisotropicWrap
	Samplers[5] = SDL_CreateGPUSampler(device, &sampler_create_info6);
    
    SDL_GPUBufferCreateInfo buffer_create_info;
    buffer_create_info.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
    buffer_create_info.size = sizeof(PositionTextureVertex) * 4;
    std::cout << "Vertex shader loaded successfully!" << std::endl;
    // Create the GPU resources
    VertexBuffer = SDL_CreateGPUBuffer(device, &buffer_create_info);
    SDL_SetGPUBufferName(
        device,
        VertexBuffer,
        "Ravioli Vertex Buffer 🥣"
    );
    std::cout << "Vertex shader loaded successfully!4" << std::endl;
    SDL_GPUBufferCreateInfo buffer_create_info2;
    buffer_create_info2.usage = SDL_GPU_BUFFERUSAGE_INDEX;
    buffer_create_info2.size = sizeof(Uint16) * 6;
    
    IndexBuffer = SDL_CreateGPUBuffer(device,&buffer_create_info2);
    std::cout << "Vertex shader loaded successfully!8" << std::endl;
    SDL_GPUTextureCreateInfo texture_create_info = {};
    texture_create_info.type = SDL_GPU_TEXTURETYPE_2D,
            texture_create_info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
            texture_create_info.width = (unsigned)imageData->w;
            texture_create_info.height = (unsigned)imageData->h;
            texture_create_info.layer_count_or_depth = 1;
            texture_create_info.num_levels = 1;
            texture_create_info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    std::cout << "Vertex shader loaded successfully!9" << std::endl;
    Texture = SDL_CreateGPUTexture(device, &texture_create_info);
    SDL_SetGPUTextureName(
        device,
        Texture,
        "Ravioli Texture 🖼️"
    );
    std::cout << "Vertex shader loaded successfully!10" << std::endl;
    SDL_GPUTransferBufferCreateInfo transfer_buffer_create_info = {};
    transfer_buffer_create_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
            transfer_buffer_create_info.size = (sizeof(PositionTextureVertex) * 4) + (sizeof(Uint16) * 6);
    
    // Set up buffer data
    SDL_GPUTransferBuffer* bufferTransferBuffer = SDL_CreateGPUTransferBuffer(device, &transfer_buffer_create_info);
    std::cout << "Vertex shader loaded successfully!10" << std::endl;
    PositionTextureVertex* transferData = (PositionTextureVertex*)SDL_MapGPUTransferBuffer(device,bufferTransferBuffer,false);
    std::cout << "Vertex shader loaded successfully!11" << std::endl;
    transferData[0] = PositionTextureVertex { -1,  1, 0, 0, 0 };
    transferData[1] = PositionTextureVertex {  1,  1, 0, 1, 0 };
    transferData[2] = PositionTextureVertex {  1, -1, 0, 1, 1 };
    transferData[3] = PositionTextureVertex { -1, -1, 0, 0, 1 };

    Uint16* indexData = (Uint16*) &transferData[4];
    indexData[0] = 0;
    indexData[1] = 1;
    indexData[2] = 2;
    indexData[3] = 0;
    indexData[4] = 2;
    indexData[5] = 3;
    
    SDL_UnmapGPUTransferBuffer(device, bufferTransferBuffer);
    std::cout << "Vertex shader loaded successfully!12" << std::endl;
    SDL_GPUTransferBufferCreateInfo transfer_buffer_create_info2 = {};
    transfer_buffer_create_info2.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
            transfer_buffer_create_info2.size = (unsigned)imageData->w * imageData->h * 4;
    
    // Set up texture data
    SDL_GPUTransferBuffer* textureTransferBuffer = SDL_CreateGPUTransferBuffer( device, &transfer_buffer_create_info2);
    if (imageData->w <= 0 || imageData->h <= 0 || imageData->pixels == NULL) {
        SDL_Log("Invalid image width/height or imageData is NULL");
        return;
    }
    Uint8* textureTransferPtr = (Uint8*)SDL_MapGPUTransferBuffer(
        device,
        textureTransferBuffer,
        false
    );
    if (textureTransferPtr == NULL) {
        SDL_Log("textureTransferPtr is NULL");
        return;
    }

    if (imageData == NULL || imageData->pixels == NULL) {
        SDL_Log("imageData or imageData->pixels is NULL");
        return;
    }
    std::cout << "Vertex shader loaded successfully!13l" << std::endl;
    if (imageData->w <= 0 || imageData->h <= 0) {
        SDL_Log("Invalid image width or height: %d x %d", imageData->w, imageData->h);
        return;
    }
    
        SDL_memcpy(textureTransferPtr, imageData->pixels, imageData->w * imageData->h *4);
   
    std::cout << "Vertex shader loaded successfully!13lol" << std::endl;
    SDL_UnmapGPUTransferBuffer(device, textureTransferBuffer);
    // Upload the transfer data to the GPU resources
    SDL_GPUCommandBuffer* uploadCmdBuf = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(uploadCmdBuf);
    SDL_GPUTransferBufferLocation transfer_buffer_location;
    transfer_buffer_location.transfer_buffer = bufferTransferBuffer;
            transfer_buffer_location.offset = 0;

    SDL_GPUBufferRegion buffer_region;
    buffer_region.buffer = VertexBuffer;
            buffer_region.offset = 0;
            buffer_region.size = sizeof(PositionTextureVertex) * 4;
    
    SDL_UploadToGPUBuffer(copyPass, &transfer_buffer_location, &buffer_region, false);
    std::cout << "uploaddd" << std::endl;
    SDL_GPUTransferBufferLocation transfer_buffer_location2;
    transfer_buffer_location2.transfer_buffer = bufferTransferBuffer;
    transfer_buffer_location2.offset = sizeof(PositionTextureVertex) * 4;

    SDL_GPUBufferRegion buffer_region2;
    buffer_region2.buffer = IndexBuffer;
    buffer_region2.offset = 0;
    buffer_region2.size = sizeof(Uint16) * 6;
    
    SDL_UploadToGPUBuffer(copyPass, &transfer_buffer_location2, &buffer_region2, false);
    std::cout << "uploaddd2" << std::endl;
    // Declare the structs beforehand
    SDL_GPUTextureTransferInfo transferInfo = {
        .transfer_buffer = textureTransferBuffer,
        .offset = 0,  // Zeros out the rest
    };

    SDL_GPUTextureRegion textureRegion = {
        .texture = Texture,
        .w = (Uint32)imageData->w,
        .h = (Uint32)imageData->h,
        .d = 1
    };

    // Now pass their addresses
    SDL_UploadToGPUTexture(
        copyPass,
        &transferInfo,
        &textureRegion,
        false
    );
    SDL_DestroySurface(imageData);
    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(uploadCmdBuf);
    SDL_ReleaseGPUTransferBuffer(device, bufferTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, textureTransferBuffer);

    CurrentSamplerIndex = 0;
    
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_EVENT_QUIT:
                {
                    running = false;
                    break;
                }
            case SDL_EVENT_KEY_DOWN:
                {
                    std::cout << "Key down" << std::endl;
                    if (event.key.scancode == SDL_SCANCODE_ESCAPE)
                    {
                        std::cout << "Quitting" << std::endl;
                        running = false;
                        break;
                    }
                }
            default: continue;;
            }
        }

        SDL_GPUCommandBuffer* cmdbuf = SDL_AcquireGPUCommandBuffer(device);
        if (cmdbuf == NULL)
        {
            SDL_Log("SDL_AcquireGPUCommandBuffer failed: %s", SDL_GetError());
        }
        
        SDL_GPUTexture* swapchainTexture;
        if (!SDL_AcquireGPUSwapchainTexture(cmdbuf, window, &swapchainTexture, NULL, NULL))
        {
            SDL_Log("SDL_AcquireGPUSwapchainTexture failed: %s", SDL_GetError());
        }
        
        if (swapchainTexture != NULL)
        {
            SDL_GPURenderPass* renderPass;
            SDL_GPUColorTargetInfo color_target_info { 0 };
            color_target_info.texture = swapchainTexture;
            color_target_info.clear_color.r = 0.0f;
            color_target_info.clear_color.g = 0.0f;
            color_target_info.clear_color.b = 0.0f;
            color_target_info.clear_color.a = 1.0f;
            color_target_info.load_op = SDL_GPU_LOADOP_CLEAR;
            color_target_info.store_op = SDL_GPU_STOREOP_STORE;

            renderPass = SDL_BeginGPURenderPass(cmdbuf, &color_target_info, 1, NULL);
            SDL_BindGPUGraphicsPipeline(renderPass, Pipeline);
            SDL_GPUBufferBinding binding;
            binding.buffer = VertexBuffer;
            binding.offset = 0;
            SDL_BindGPUVertexBuffers(renderPass, 0, &binding, 1);
            SDL_GPUBufferBinding binding2;
            binding2.buffer = IndexBuffer;
            binding2.offset = 0;
            SDL_BindGPUIndexBuffer(renderPass, &binding2, SDL_GPU_INDEXELEMENTSIZE_16BIT);
            SDL_GPUTextureSamplerBinding texture_sampler_binding;
            texture_sampler_binding.texture = Texture;
            texture_sampler_binding.sampler = Samplers[CurrentSamplerIndex];
            SDL_BindGPUFragmentSamplers(renderPass, 0, &texture_sampler_binding, 1);
            SDL_DrawGPUIndexedPrimitives(renderPass, 6, 1, 0, 0, 0);

            SDL_EndGPURenderPass(renderPass);
           
            SDL_SubmitGPUCommandBuffer(cmdbuf);
        }
        else
        {
            SDL_CancelGPUCommandBuffer(cmdbuf);
        }
    }
    SDL_ReleaseGPUGraphicsPipeline(device, Pipeline);
    SDL_ReleaseGPUBuffer(device, VertexBuffer);
    SDL_ReleaseGPUBuffer(device, IndexBuffer);
    SDL_ReleaseGPUTexture(device, Texture);
    for (int i = 0; i < SDL_arraysize(Samplers); i += 1)
    {
        SDL_ReleaseGPUSampler(device, Samplers[i]);
    }
    CurrentSamplerIndex = 0;
}

int main(int argc, char** argv)
{
    createWindow();
    std::cout << "delete Window" << std::endl;

    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyGPUDevice(device);
    // Close and destroy the window
    if (window)
    {
        SDL_DestroyWindow(window);
    }

    // Clean up SDL
    SDL_Quit();
    return 0;
}*/

/*#include <format>
#include <iostream>
#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>
#include <SDL3_ttf/SDL_ttf.h>

SDL_GPUShader* LoadShader(
    SDL_GPUDevice* device,
    const char* shaderFilename,
    Uint32 samplerCount,
    Uint32 uniformBufferCount,
    Uint32 storageBufferCount,
    Uint32 storageTextureCount
)
{
    // Auto-detect the shader stage from the file name for convenience
    SDL_GPUShaderStage stage;
    if (SDL_strstr(shaderFilename, ".vert"))
    {
        stage = SDL_GPU_SHADERSTAGE_VERTEX;
    }
    else if (SDL_strstr(shaderFilename, ".frag"))
    {
        stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    }
    else
    {
        SDL_Log("Invalid shader stage!");
        return NULL;
    }
    
    char fullPath[256];
    SDL_GPUShaderFormat backendFormats = SDL_GetGPUShaderFormats(device);
    SDL_GPUShaderFormat format;
    const char* entrypoint;

    if (backendFormats & SDL_GPU_SHADERFORMAT_SPIRV)
    {
        SDL_snprintf(fullPath, sizeof(fullPath), "%sContent/Shaders/Compiled/SPIRV/%s.spv", SDL_GetBasePath(),
                     shaderFilename);
        format = SDL_GPU_SHADERFORMAT_SPIRV;
        entrypoint = "main";
    }
    else if (backendFormats & SDL_GPU_SHADERFORMAT_MSL)
    {
        SDL_snprintf(fullPath, sizeof(fullPath), "%sContent/Shaders/Compiled/MSL/%s.msl", SDL_GetBasePath(),
                     shaderFilename);
        format = SDL_GPU_SHADERFORMAT_MSL;
        entrypoint = "main0";
    }
    else if (backendFormats & SDL_GPU_SHADERFORMAT_DXIL)
    {
        SDL_snprintf(fullPath, sizeof(fullPath), "%sContent/Shaders/Compiled/DXIL/%s.dxil", SDL_GetBasePath(),
                     shaderFilename);
        format = SDL_GPU_SHADERFORMAT_DXIL;
        entrypoint = "main";
    }
    else
    {
        SDL_Log("%s", "Unrecognized backend shader format!");
        return NULL;
    }

    size_t codeSize;
    void* code = SDL_LoadFile(fullPath, &codeSize);
    if (code == NULL)
    {
        SDL_Log("Failed to load shader from disk! %s", fullPath);
        return NULL;
    }

    SDL_GPUShaderCreateInfo shaderInfo{
        .code_size = codeSize,
        .code = (const Uint8*)code,
        .entrypoint = entrypoint,
        .format = format,
        .stage = stage,
        .num_samplers = samplerCount,
        .num_storage_textures = storageTextureCount,
        .num_storage_buffers = storageBufferCount,
        .num_uniform_buffers = uniformBufferCount
    };
    
    SDL_GPUShader* shader = SDL_CreateGPUShader(device, &shaderInfo);
    if (shader == NULL)
    {
        SDL_Log("Failed to create shader!");
        SDL_free(code);
        return NULL;
    }

    SDL_free(code);
    return shader;
}

SDL_Window* window;
SDL_GPUDevice* device;

bool running = true;

SDL_GPUBuffer* VertexBuffer;

SDL_GPUBuffer* IndexBuffer;

void createWindow()
{
    std::cout << "create Window" << std::endl;

    // Initialize SDL and check for errors
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        // SDL_Init returns non-zero on error
        SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
        return; // Return early if initialization failed
    }

    // Create an application window with the following settings:
    window = SDL_CreateWindow(
        "An SDL3 window", // window title
        640, // width, in pixels
        480, // height, in pixels
        0 // flags - using OpenGL here
    );

    // Check if the window was successfully created
    if (window == nullptr)
    {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s", SDL_GetError());
        SDL_Quit(); // Clean up SDL if window creation failed
        return;
    }

    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, nullptr);
    if (!device)
    {
        SDL_Log("SDL_CreateGPUDevice failed: %s", SDL_GetError());
    }

    std::cout << "Device is rendered with: " << SDL_GetGPUDeviceDriver(device) << std::endl;
    if (!SDL_ClaimWindowForGPUDevice(device, window))
    {
        SDL_Log("SDL_ClaimWindowForGPUDevice failed: %s", SDL_GetError());
    }

    // Create the shaders
    SDL_GPUShader* vertexShader = LoadShader(device, "shader.vert", 0, 0, 0, 0);
    if (vertexShader == NULL)
    {
        SDL_Log("Failed to create vertex shader!");
        return;
    }

    SDL_GPUShader* fragmentShader = LoadShader(device, "shader.frag", 0, 0, 0, 0);
    if (fragmentShader == NULL)
    {
        SDL_Log("Failed to create fragment shader!");
        return;
    }
    
   SDL_PixelFormat format = SDL_PIXELFORMAT_ABGR8888;
    
    SDL_GPUColorTargetDescription color_target_description;
    color_target_description.format = SDL_GetGPUSwapchainTextureFormat(device, window);
    color_target_description.blend_state = {
        .src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .color_blend_op = SDL_GPU_BLENDOP_ADD,
        .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
        .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
        .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
        .enable_blend = true
    };
    
    // Create the pipelines
    SDL_GPUGraphicsPipelineCreateInfo pipelineCreateInfo = {
        .vertex_shader = vertexShader,
        .fragment_shader = fragmentShader,
        .vertex_input_state = nullptr,
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .multisample_state = SDL_GPU_SAMPLECOUNT_1,
        .depth_stencil_state = {},
        .target_info = {
            //.color_target_descriptions = &color_target_description,
            .color_target_descriptions = &color_target_description,
            .num_color_targets = 1
        }
    };
    //(SDL_GPUColorTargetDescription[]){{
    //    .format = SDL_GetGPUSwapchainTextureFormat(device, window)
    // }},
    pipelineCreateInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    SDL_GPUGraphicsPipeline* FillPipeline = SDL_CreateGPUGraphicsPipeline(device, &pipelineCreateInfo);
    if (FillPipeline == NULL)
    {
        SDL_Log("Failed to create fill pipeline!");
        return;
    }

    pipelineCreateInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_LINE;
    SDL_GPUGraphicsPipeline* LinePipeline = SDL_CreateGPUGraphicsPipeline(device, &pipelineCreateInfo);
    if (LinePipeline == NULL)
    {
        SDL_Log("Failed to create line pipeline!");
        return;
    }

    // Clean up shader resources
    SDL_ReleaseGPUShader(device, vertexShader);
    SDL_ReleaseGPUShader(device, fragmentShader);

    int width, height;
    SDL_GetWindowSizeInPixels(window, &width, &height);
    
    static SDL_GPUViewport SmallViewport = {0.0f, 0.0f, (float)width / 4, (float)height / 4, 0.1f, 1.0f};
    //static SDL_GPUViewport SmallViewport = {0, 0, (float)width / 4, (float)height / 4, 0.1f, 1.0f};
    static SDL_Rect ScissorRect = {0, 0, 640, 480};

    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_EVENT_QUIT:
                {
                    running = false;
                    break;
                }
            case SDL_EVENT_KEY_DOWN:
                {
                    std::cout << "Key down" << std::endl;
                    if (event.key.scancode == SDL_SCANCODE_ESCAPE)
                    {
                        std::cout << "Quitting" << std::endl;
                        running = false;
                        break;
                    }
                }
            default: continue;;
            }
        }

        SDL_GPUCommandBuffer* cmdbuf = SDL_AcquireGPUCommandBuffer(device);
        if (cmdbuf == NULL)
        {
            SDL_Log("SDL_AcquireGPUCommandBuffer failed: %s", SDL_GetError());
        }
        
        SDL_GPUTexture* swapchainTexture;
        if (!SDL_AcquireGPUSwapchainTexture(cmdbuf, window, &swapchainTexture, NULL, NULL))
        {
            SDL_Log("SDL_AcquireGPUSwapchainTexture failed: %s", SDL_GetError());
        }
        
        if (swapchainTexture != NULL)
        {
            SDL_GPURenderPass* renderPass;
            SDL_GPUColorTargetInfo color_target_info;
            SDL_zero(color_target_info);
            color_target_info.texture = swapchainTexture;
            color_target_info.clear_color.r = 0.212f;
            color_target_info.clear_color.g = 0.075f;
            color_target_info.clear_color.b = 0.541f;
            color_target_info.clear_color.a = 1.0f;
            color_target_info.load_op = SDL_GPU_LOADOP_CLEAR;
            color_target_info.store_op = SDL_GPU_STOREOP_STORE;

            renderPass = SDL_BeginGPURenderPass(cmdbuf, &color_target_info, 1, NULL);
            SDL_BindGPUGraphicsPipeline(renderPass, true ? LinePipeline : FillPipeline);
            if (false)
            {
                SDL_SetGPUViewport(renderPass, &SmallViewport);
            }
            if (false)
            {
                SDL_SetGPUScissor(renderPass, &ScissorRect);
            }

            SDL_DrawGPUPrimitives(renderPass, 6, 1, 0, 0);
            SDL_EndGPURenderPass(renderPass);

            SDL_SubmitGPUCommandBuffer(cmdbuf);
        }
        else
        {
            SDL_CancelGPUCommandBuffer(cmdbuf);
        }
    }
    SDL_ReleaseGPUGraphicsPipeline(device, FillPipeline);
    SDL_ReleaseGPUGraphicsPipeline(device, LinePipeline);
}

int main(int argc, char** argv)
{
    createWindow();
    std::cout << "delete Window" << std::endl;

    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyGPUDevice(device);
    // Close and destroy the window
    if (window)
    {
        SDL_DestroyWindow(window);
    }

    // Clean up SDL
    SDL_Quit();
    return 0;
}*/


/*#include <iostream>
#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>

SDL_Window* window;

SDL_GLContext glContext;

SDL_Renderer* renderer;

bool running = true;

void createWindow()
{
        std::cout << "create Window" << std::endl;

        // Initialize SDL and check for errors
        if (!SDL_Init(SDL_INIT_VIDEO))
        {
            // SDL_Init returns non-zero on error
            SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
            return; // Return early if initialization failed
        }
    if (!TTF_Init()) {
        std::cout << "Error initializing SDL_ttf: " << SDL_GetError() << std::endl;
    }

    TTF_Font *font = NULL;
    font = TTF_OpenFont("Roboto-Thin.ttf", 36);
    if ( !font ) {
        std::cout << "Failed to load font: " << SDL_GetError() << std::endl;
    }
        /*
        // Set OpenGL attributes (optional, but may fail if system doesn't support it)
        if (!SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1))
        {
            SDL_Log("Error setting SDL_GL_DOUBLEBUFFER: %s", SDL_GetError());
        }
        if (!SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4))
        {
            SDL_Log("Error setting SDL_GL_CONTEXT_MAJOR_VERSION: %s", SDL_GetError());
        }
        if (!SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6))
        {
            SDL_Log("Error setting SDL_GL_CONTEXT_MINOR_VERSION: %s", SDL_GetError());
        }
        if (!SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE))
        {
            SDL_Log("Error setting SDL_GL_CONTEXT_PROFILE_MASK: %s", SDL_GetError());
        }
        #1#
        
        // Create an application window with the following settings:
        window = SDL_CreateWindow(
            "An SDL3 window", // window title
            640, // width, in pixels
            480, // height, in pixels
            SDL_WINDOW_OPENGL // flags - using OpenGL here
        );
        
        // Check if the window was successfully created
        if (window == nullptr)
        {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not create window: %s", SDL_GetError());
            SDL_Quit(); // Clean up SDL if window creation failed
            return;
        }
        
        // Create an OpenGL context associated with the window.
        glContext = SDL_GL_CreateContext(window);
        if (glContext == nullptr)
        {
            std::cout << "OpenGL context could not be created" << std::endl;
        }

        renderer = SDL_CreateRenderer(window, NULL);
        if (renderer == nullptr)
        {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Could not create renderer: %s", SDL_GetError());
        }
    // Clear the window to white
    //SDL_SetRenderDrawColor( renderer, 255, 255, 255, 255 );
    //SDL_RenderClear( renderer );
    SDL_Surface* text;
    // Set color to black
    SDL_Color color = { 255, 0, 255 };
    std::string texts = "hurenosh ndeines kindes mannes!äü";
    text = TTF_RenderText_Solid( font, texts.c_str(), texts.length(),color );

    SDL_Texture* text_texture;

    text_texture = SDL_CreateTextureFromSurface( renderer, text );
    SDL_FRect dest = { 0, 0, (float)text->w, (float)text->h };
    SDL_FRect dstRect = { 0, 480 - static_cast<float>(TTF_GetFontHeight(font)), static_cast<float>(text->w)  , static_cast<float>(text->h)};
    SDL_RenderTexture( renderer, text_texture, &dest , &dstRect);
    if ( !text ) {
        std::cout << "Failed to render text: " << SDL_GetError() << std::endl;
    }
    
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_EVENT_QUIT:
                {
                    running = false;
                    break;
                }
            case SDL_EVENT_KEY_DOWN:
                {
                    std::cout << "Key down" << std::endl;
                    if (event.key.scancode == SDL_SCANCODE_ESCAPE)
                    {
                        std::cout << "Quitting" << std::endl;
                        running = false;
                        break;
                    }
                }
            default: continue;;
            }
        }
        //SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
        //SDL_RenderClear(renderer);
        SDL_RenderPresent(renderer);
        SDL_DestroyTexture( text_texture );
        SDL_DestroySurface( text );
    }
}

int main(int argc, char **argv)
{
    createWindow();
    std::cout << "delete Window" << std::endl;
    // Unload glad before exiting TBD
        
    if (glContext)
    {
        // Once finished with OpenGL functions, the SDL_GLContext can be destroyed.
        SDL_GL_DestroyContext(glContext);  
    }
        
    // Destroy Renderer
    if (renderer)
    {
        SDL_DestroyRenderer(renderer);
    }

    // Close and destroy the window
    if (window)
    {
        SDL_DestroyWindow(window);
    }

    // Clean up SDL
    SDL_Quit();
    return 0;
}*/
