#define SDL_MAIN_USE_CALLBACKS
#include <iostream>
#include <ostream>
#include <vector>
#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_main.h>
#include <SDL3_image/SDL_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>  // for pointer to matrix or vector

static float t = 0;

typedef struct PositionTextureVertex
{
    float x, y, z;
    float u, v;
} PositionTextureVertex;

static SDL_Window* window = nullptr;
static SDL_GPUDevice* device = nullptr;
static SDL_GPUGraphicsPipeline* Pipeline = nullptr;
static SDL_Surface* imageData;
static SDL_Surface* imageData2;

struct GPUResources
{
    SDL_GPUBuffer* vertexBuffer;
    SDL_GPUBuffer* indexBuffer;
    SDL_GPUTexture* texture;
    SDL_GPUSampler* sampler;
    SDL_GPUTransferBuffer* bufferTransferBuffer;
    SDL_GPUTransferBuffer* textureTransferBuffer;
};

// A vector to store multiple sets of GPU resources
std::vector<GPUResources> gpuResourcesList;


static SDL_Surface* LoadImage(const char* imageFilename, int desiredChannels)
{
    char fullPath[256];
    SDL_Surface* result;
    SDL_PixelFormat format;

    SDL_snprintf(fullPath, sizeof(fullPath), "%sContent/Images/%s", SDL_GetBasePath(), imageFilename);

    result = IMG_Load(fullPath);
    if (result == nullptr)
    {
        SDL_Log("Failed to load Image: %s", SDL_GetError());
        return nullptr;
    }
    SDL_FlipSurface(result, SDL_FLIP_VERTICAL);
    if (desiredChannels == 4)
    {
        format = SDL_PIXELFORMAT_ABGR8888;
    }
    else
    {
        SDL_assert(!"Unexpected desiredChannels");
        SDL_DestroySurface(result);
        return nullptr;
    }
    if (result->format != format)
    {
        SDL_Surface* next = SDL_ConvertSurface(result, format);
        SDL_DestroySurface(result);
        result = next;
    }

    return result;
}

static SDL_GPUShader* LoadShader(
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
    SDL_GPUShaderFormat format = SDL_GPU_SHADERFORMAT_INVALID;
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
        return nullptr;
    }

    size_t codeSize;
    void* code = SDL_LoadFile(fullPath, &codeSize);
    if (code == nullptr)
    {
        SDL_Log("Failed to load shader from disk! %s", fullPath);
        return nullptr;
    }

    SDL_GPUShaderCreateInfo shaderInfo = {
        .code_size = codeSize,
        .code = static_cast<Uint8*>(code),
        .entrypoint = entrypoint,
        .format = format,
        .stage = stage,
        .num_samplers = samplerCount,
        .num_storage_textures = storageTextureCount,
        .num_storage_buffers = storageBufferCount,
        .num_uniform_buffers = uniformBufferCount
    };

    SDL_GPUShader* shader = SDL_CreateGPUShader(device, &shaderInfo);
    if (shader == nullptr)
    {
        SDL_Log("Failed to create shader!");
        SDL_free(code);
        return nullptr;
    }

    SDL_free(code);
    return shader;
}

static void drawQuadTexture(SDL_Surface* imageData)
{
    if (imageData == nullptr)
    {
        //white texture as fallback if no image
        // Get the pixel format details for RGBA32 (replace SDL_PIXELFORMAT_RGBA32)
        const SDL_PixelFormatDetails* pixelFormatDetails = SDL_GetPixelFormatDetails(SDL_PIXELFORMAT_RGBA32);
        SDL_Surface* whiteSurface = SDL_CreateSurface(50, 50, SDL_PIXELFORMAT_RGBA32);
        SDL_FillSurfaceRect(whiteSurface, nullptr, SDL_MapRGBA(pixelFormatDetails, nullptr, 255, 255, 255, 255));
        imageData = whiteSurface;
    }


    SDL_GPUBuffer* VertexBuffer = nullptr;
    SDL_GPUBuffer* IndexBuffer = nullptr;
    SDL_GPUTexture* Texture = nullptr;
    SDL_GPUSampler* Sampler = nullptr;
    // Store the GPU resources
    GPUResources resources = {
        .vertexBuffer = VertexBuffer,
        .indexBuffer = IndexBuffer,
        .texture = Texture,
        .sampler = Sampler,
    };

    SDL_GPUBufferCreateInfo buffer_create_infoVertex = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = sizeof(PositionTextureVertex) * 4
    };

    // Create the GPU resources
    VertexBuffer = SDL_CreateGPUBuffer(device, &buffer_create_infoVertex);
    resources.vertexBuffer = VertexBuffer;
    SDL_GPUBufferCreateInfo buffer_create_infoIndex{
        .usage = SDL_GPU_BUFFERUSAGE_INDEX,
        .size = sizeof(Uint16) * 6
    };

    IndexBuffer = SDL_CreateGPUBuffer(device, &buffer_create_infoIndex);
    resources.indexBuffer = IndexBuffer;
    SDL_GPUTextureCreateInfo texture_create_info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = static_cast<unsigned>(imageData->w),
        .height = static_cast<unsigned>(imageData->h),
        .layer_count_or_depth = 1,
        .num_levels = 1
    };

    Texture = SDL_CreateGPUTexture(device, &texture_create_info);
    resources.texture = Texture;
    SDL_GPUSamplerCreateInfo sampler_create_info = {
        .min_filter = SDL_GPU_FILTER_NEAREST,
        .mag_filter = SDL_GPU_FILTER_NEAREST,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
    };

    Sampler = SDL_CreateGPUSampler(device, &sampler_create_info);
    resources.sampler = Sampler;
    SDL_GPUTransferBufferCreateInfo transfer_buffer_create_info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = (sizeof(PositionTextureVertex) * 4) + (sizeof(Uint16) * 6)
    };

    // Set up buffer data
    SDL_GPUTransferBuffer* bufferTransferBuffer = SDL_CreateGPUTransferBuffer(device, &transfer_buffer_create_info);

    PositionTextureVertex* transferData = static_cast<PositionTextureVertex*>(SDL_MapGPUTransferBuffer(
        device,
        bufferTransferBuffer,
        false
    ));

    transferData[0] = PositionTextureVertex{-0.5f, -0.5f, 0, 0, 0}; // bottom left
    transferData[1] = PositionTextureVertex{0.5f, -0.5f, 0, 1, 0}; // bottom right
    transferData[2] = PositionTextureVertex{0.5f, 0.5f, 0, 1, 1}; // top right
    transferData[3] = PositionTextureVertex{-0.5f, 0.5f, 0, 0, 1}; // top left

    Uint16* indexData = reinterpret_cast<Uint16*>(&transferData[4]);
    indexData[0] = 0;
    indexData[1] = 1;
    indexData[2] = 2;
    indexData[3] = 0;
    indexData[4] = 2;
    indexData[5] = 3;

    SDL_UnmapGPUTransferBuffer(device, bufferTransferBuffer);

    // Set up texture data
    const Uint32 imageSizeInBytes = imageData->w * imageData->h * 4;
    SDL_GPUTransferBufferCreateInfo transfer_buffer_create_info2 = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = imageSizeInBytes
    };

    SDL_GPUTransferBuffer* textureTransferBuffer = SDL_CreateGPUTransferBuffer(device, &transfer_buffer_create_info2);

    Uint8* textureTransferPtr = static_cast<Uint8*>(SDL_MapGPUTransferBuffer(
        device,
        textureTransferBuffer,
        false
    ));

    SDL_memcpy(textureTransferPtr, imageData->pixels, imageSizeInBytes);
    SDL_UnmapGPUTransferBuffer(device, textureTransferBuffer);

    // Upload the transfer data to the GPU resources
    SDL_GPUCommandBuffer* uploadCmdBuf = SDL_AcquireGPUCommandBuffer(device);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(uploadCmdBuf);

    SDL_GPUTransferBufferLocation transfer_buffer_location = {
        .transfer_buffer = bufferTransferBuffer,
        .offset = 0
    };

    SDL_GPUBufferRegion buffer_region = {
        .buffer = VertexBuffer,
        .offset = 0,
        .size = sizeof(PositionTextureVertex) * 4
    };

    SDL_UploadToGPUBuffer(copyPass, &transfer_buffer_location, &buffer_region, false);

    SDL_GPUTransferBufferLocation transfer_buffer_location2 = {
        .transfer_buffer = bufferTransferBuffer,
        .offset = sizeof(PositionTextureVertex) * 4
    };

    SDL_GPUBufferRegion buffer_region2 = {
        .buffer = IndexBuffer,
        .offset = 0,
        .size = sizeof(Uint16) * 6
    };

    SDL_UploadToGPUBuffer(copyPass, &transfer_buffer_location2, &buffer_region2, false);

    SDL_GPUTextureTransferInfo texture_transfer_info = {
        .transfer_buffer = textureTransferBuffer,
        .offset = 0,
    };

    SDL_GPUTextureRegion texture_region = {
        .texture = Texture,
        .w = static_cast<unsigned>(imageData->w),
        .h = static_cast<unsigned>(imageData->h),
        .d = 1
    };

    SDL_UploadToGPUTexture(copyPass, &texture_transfer_info, &texture_region, false);

    SDL_DestroySurface(imageData);
    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(uploadCmdBuf);
    SDL_ReleaseGPUTransferBuffer(device, bufferTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, textureTransferBuffer);

    gpuResourcesList.emplace_back(resources);
}

static int Draw()
{
    SDL_GPUCommandBuffer* cmdbuf = SDL_AcquireGPUCommandBuffer(device);
    if (cmdbuf == nullptr)
    {
        SDL_Log("AcquireGPUCommandBuffer failed: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_GPUTexture* swapchainTexture;
    if (!SDL_AcquireGPUSwapchainTexture(cmdbuf, window, &swapchainTexture, nullptr, nullptr))
    {
        SDL_Log("WaitAndAcquireGPUSwapchainTexture failed: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    if (swapchainTexture != nullptr)
    {
        SDL_GPUColorTargetInfo colorTargetInfo = {nullptr};
        colorTargetInfo.texture = swapchainTexture;
        colorTargetInfo.clear_color = SDL_FColor{0.212f, 0.075f, 0.541f, 1.0f};
        //colorTargetInfo.clear_color = SDL_FColor { 0.0f, 0.0f, 0.0f, 1.0f };
        colorTargetInfo.load_op = SDL_GPU_LOADOP_CLEAR;
        colorTargetInfo.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(cmdbuf, &colorTargetInfo, 1, nullptr);
        SDL_BindGPUGraphicsPipeline(renderPass, Pipeline);

        SDL_GPUBufferBinding buffer_binding1 = {.buffer = gpuResourcesList[1].vertexBuffer, .offset = 0};
        SDL_BindGPUVertexBuffers(renderPass, 0, &buffer_binding1, 1);
        SDL_GPUBufferBinding buffer_binding3 = {.buffer = gpuResourcesList[1].indexBuffer, .offset = 0};
        SDL_BindGPUIndexBuffer(renderPass, &buffer_binding3, SDL_GPU_INDEXELEMENTSIZE_16BIT);
        SDL_GPUTextureSamplerBinding texture_sampler_binding2 = {
            .texture = gpuResourcesList[1].texture, .sampler = gpuResourcesList[1].sampler
        };
        SDL_BindGPUFragmentSamplers(renderPass, 0, &texture_sampler_binding2, 1);

        // Matrix math
        glm::mat4 rotationMatrix2 = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f * t), glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 translationMatrix2 = glm::translate(glm::mat4(1.0f), glm::vec3(-0.5f, 0.5f, 0.0f));
        glm::mat4 matrixUniforms2 = translationMatrix2 * rotationMatrix2;

        // Update fragment uniform color
        glm::vec4 fragUniformBottomRights2 = glm::vec4(1.0f, 0.5f + SDL_sinf(t) * 1.0f, 1.0f, 1.0f);

        // Push the matrix and fragment uniform data to the GPU
        SDL_PushGPUVertexUniformData(cmdbuf, 0, glm::value_ptr(matrixUniforms2), sizeof(glm::mat4));
        SDL_PushGPUFragmentUniformData(cmdbuf, 0, glm::value_ptr(fragUniformBottomRights2), sizeof(glm::vec4));
        SDL_DrawGPUIndexedPrimitives(renderPass, 6, 1, 0, 0, 0);

        SDL_GPUBufferBinding buffer_binding = {.buffer = gpuResourcesList[0].vertexBuffer, .offset = 0};
        SDL_BindGPUVertexBuffers(renderPass, 0, &buffer_binding, 1);
        SDL_GPUBufferBinding buffer_binding2 = {.buffer = gpuResourcesList[0].indexBuffer, .offset = 0};
        SDL_BindGPUIndexBuffer(renderPass, &buffer_binding2, SDL_GPU_INDEXELEMENTSIZE_16BIT);
        SDL_GPUTextureSamplerBinding texture_sampler_binding = {
            .texture = gpuResourcesList[0].texture, .sampler = gpuResourcesList[0].sampler
        };
        SDL_BindGPUFragmentSamplers(renderPass, 0, &texture_sampler_binding, 1);

        // Matrix math
        glm::mat4 rotationMatrix4 = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f * t), glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 translationMatrix4 = glm::translate(glm::mat4(1.0f), glm::vec3(-0.5f, -0.5f, 0.0f));
        glm::mat4 matrixUniforms4 = translationMatrix4 * rotationMatrix4;

        // Update fragment uniform color
        glm::vec4 fragUniformBottomRights4 = glm::vec4(1.0f, 0.5f + SDL_sinf(t) * 1.0f, 1.0f, 1.0f);

        // Push the matrix and fragment uniform data to the GPU
        SDL_PushGPUVertexUniformData(cmdbuf, 0, glm::value_ptr(matrixUniforms4), sizeof(glm::mat4));
        SDL_PushGPUFragmentUniformData(cmdbuf, 0, glm::value_ptr(fragUniformBottomRights4), sizeof(glm::vec4));
        SDL_DrawGPUIndexedPrimitives(renderPass, 6, 1, 0, 0, 0);

        SDL_GPUBufferBinding buffer_binding4 = {.buffer = gpuResourcesList[2].vertexBuffer, .offset = 0};
        SDL_BindGPUVertexBuffers(renderPass, 0, &buffer_binding4, 1);
        SDL_GPUBufferBinding buffer_binding5 = {.buffer = gpuResourcesList[2].indexBuffer, .offset = 0};
        SDL_BindGPUIndexBuffer(renderPass, &buffer_binding5, SDL_GPU_INDEXELEMENTSIZE_16BIT);
        SDL_GPUTextureSamplerBinding texture_sampler_binding3 = {
            .texture = gpuResourcesList[2].texture, .sampler = gpuResourcesList[2].sampler
        };
        SDL_BindGPUFragmentSamplers(renderPass, 0, &texture_sampler_binding3, 1);

        // Matrix math
        glm::mat4 scale = glm::scale(glm::mat4(1.0f), 0.5f * glm::vec3(1.0f, 1.0f, 1.0f));
        glm::mat4 rotationMatrix3 = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f * t), glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 translationMatrix3 = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, -0.5f, 0.0f));
        glm::mat4 matrixUniforms3 = translationMatrix3 * rotationMatrix3 * scale;
        // Update fragment uniform color
        glm::vec4 fragUniformBottomRights3 = glm::vec4(1.0f, 0.5f + cos(t) * 1.0f, 1.0f, 1.0f);

        /*// Push the matrix and fragment uniform data to the GPU
        SDL_PushGPUVertexUniformData(cmdbuf, 0, glm::value_ptr(matrixUniforms3), sizeof(glm::mat4));
        SDL_PushGPUFragmentUniformData(cmdbuf, 0, glm::value_ptr(fragUniformBottomRights3), sizeof(glm::vec4));
        SDL_DrawGPUIndexedPrimitives(renderPass, 6, 1, 0, 0, 0);
        SDL_GPUBufferBinding buffer_binding6 = {.buffer = gpuResourcesList[3].vertexBuffer, .offset = 0};
        SDL_BindGPUVertexBuffers(renderPass, 0, &buffer_binding6, 1);
        SDL_GPUBufferBinding buffer_binding7 = {.buffer = gpuResourcesList[3].indexBuffer, .offset = 0};
        SDL_BindGPUIndexBuffer(renderPass, &buffer_binding7, SDL_GPU_INDEXELEMENTSIZE_16BIT);
        SDL_GPUTextureSamplerBinding texture_sampler_binding4 = {
            .texture = gpuResourcesList[3].texture, .sampler = gpuResourcesList[3].sampler
        };
        SDL_BindGPUFragmentSamplers(renderPass, 0, &texture_sampler_binding4, 1);*/
        // Matrix math
        glm::mat4 scale2 = glm::scale(glm::mat4(1.0f), 0.5f * glm::vec3(1.0f, 1.0f, 1.0f));
        glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.0f));
        glm::mat4 matrixUniforms = translationMatrix * rotationMatrix * scale2;

        // Update fragment uniform color
        glm::vec4 fragUniformBottomRights = glm::vec4(1.0f, 0.5f + cos(t) * 1.0f, 1.0f, 1.0f);

        // Push the matrix and fragment uniform data to the GPU
        SDL_PushGPUVertexUniformData(cmdbuf, 0, glm::value_ptr(matrixUniforms), sizeof(glm::mat4));
        SDL_PushGPUFragmentUniformData(cmdbuf, 0, glm::value_ptr(fragUniformBottomRights), sizeof(glm::vec4));
        SDL_DrawGPUIndexedPrimitives(renderPass, 6, 1, 0, 0, 0);

        SDL_EndGPURenderPass(renderPass);
    }

    SDL_SubmitGPUCommandBuffer(cmdbuf);

    return SDL_APP_CONTINUE;
}

SDL_Surface* imageData3;
/* This function runs once at startup. */
SDL_AppResult SDL_AppInit(void** appstate, int argc, char** argv)
{
    SDL_SetAppMetadata("Example Renderer Clear", "1.0", "com.example.renderer-clear");

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    window = SDL_CreateWindow("examples/renderer/clear", 640, 480, SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        SDL_Log("Couldn't create window: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_DXIL | SDL_GPU_SHADERFORMAT_MSL,
                                 true, nullptr);
    if (!device)
    {
        SDL_Log("GPUCreateDevice failed");
        return SDL_APP_FAILURE;
    }

    const char* driver = SDL_GetGPUDeviceDriver(device);
    SDL_Log("GPU driver: %s", driver);

    if (!SDL_ClaimWindowForGPUDevice(device, window))
    {
        SDL_Log("SDL_ClaimWindowForGPUDevice failed: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    //----------------------------------------------------------

    // Create the shaders
    SDL_GPUShader* vertexShader = LoadShader(device, "shader.vert", 0, 1, 0, 0);
    if (vertexShader == nullptr)
    {
        SDL_Log("Failed to create vertex shader!");
        return SDL_APP_FAILURE;
    }

    SDL_GPUShader* fragmentShader = LoadShader(device, "shader.frag", 1, 1, 0, 0);
    if (fragmentShader == nullptr)
    {
        SDL_Log("Failed to create fragment shader!");
        return SDL_APP_FAILURE;
    }

    // Define the color target descriptions separately
    SDL_GPUColorTargetDescription color_target_descriptions[] = {
        {
            .format = SDL_GetGPUSwapchainTextureFormat(device, window),
            .blend_state = {
                .src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
                .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .color_blend_op = SDL_GPU_BLENDOP_ADD,
                .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
                .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
                .enable_blend = true
            }
        }
    };

    // Define the vertex buffer descriptions separately
    SDL_GPUVertexBufferDescription vertex_buffer_descriptions[] = {
        {
            .slot = 0,
            .pitch = sizeof(PositionTextureVertex),
            .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
            .instance_step_rate = 0
        }
    };

    // Define the vertex attributes separately
    SDL_GPUVertexAttribute vertex_attributes[] = {
        {
            .location = 0,
            .buffer_slot = 0,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
            .offset = 0
        },
        {
            .location = 1,
            .buffer_slot = 0,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
            .offset = sizeof(float) * 3
        },
    };

    // Create the pipeline
    SDL_GPUGraphicsPipelineCreateInfo pipelineCreateInfo = {
        .vertex_shader = vertexShader,
        .fragment_shader = fragmentShader,
        .vertex_input_state = {
            .vertex_buffer_descriptions = vertex_buffer_descriptions,
            .num_vertex_buffers = 1,
            .vertex_attributes = vertex_attributes,
            .num_vertex_attributes = 2
        },
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .target_info = {
            .color_target_descriptions = color_target_descriptions,
            .num_color_targets = 1
        },
    };

    Pipeline = SDL_CreateGPUGraphicsPipeline(device, &pipelineCreateInfo);

    if (Pipeline == nullptr)
    {
        SDL_Log("Failed to create pipeline!");
        return SDL_APP_FAILURE;
    }

    SDL_ReleaseGPUShader(device, vertexShader);
    SDL_ReleaseGPUShader(device, fragmentShader);

    // Load the images
    imageData = LoadImage("char2.png", 4);
    if (imageData == nullptr)
    {
        SDL_Log("Could not load first image data!");
        return SDL_APP_FAILURE;
    }

    // Load the images
    imageData2 = LoadImage("char.png", 4);
    if (imageData2 == nullptr)
    {
        SDL_Log("Could not load second image data!");
        return SDL_APP_FAILURE;
    }

    // Load the images
    imageData3 = LoadImage("char3(1).png", 4);
    if (imageData3 == nullptr)
    {
        SDL_Log("Could not load second image data!");
        return SDL_APP_FAILURE;
    }

    drawQuadTexture(imageData);
    drawQuadTexture(imageData2);
    drawQuadTexture(imageData3);


    return SDL_APP_CONTINUE;
}

/* This function runs when a new event (mouse input, keypresses, etc) occurs. */
SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event)
{
    if (event->type == SDL_EVENT_QUIT)
    {
        return SDL_APP_SUCCESS; /* end the program, reporting success to the OS. */
    }
    if (event->type == SDL_EVENT_KEY_DOWN)
    {
        if (event->key.scancode == SDL_SCANCODE_ESCAPE)
        {
            return SDL_APP_SUCCESS;
        }
    }

    return SDL_APP_CONTINUE;
}

/* This function runs once per frame, and is the heart of the program. */
SDL_AppResult SDL_AppIterate(void* appstate)
{
    Draw();
    t += (1.0f / 1000.0f);

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result)
{
    /* SDL will clean up the window for us. */
    SDL_ReleaseGPUGraphicsPipeline(device, Pipeline);
    for (const auto& resources : gpuResourcesList)
    {
        SDL_ReleaseGPUTexture(device, resources.texture);
        SDL_ReleaseGPUSampler(device, resources.sampler);
        SDL_ReleaseGPUBuffer(device, resources.vertexBuffer);
        SDL_ReleaseGPUBuffer(device, resources.indexBuffer);
    }

    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyWindow(window);
    SDL_DestroyGPUDevice(device);
}
