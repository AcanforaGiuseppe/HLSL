import compushady
import compushady.config
from compushady.formats import R8G8B8A8_UNORM, B8G8R8A8_UNORM
from compushady import (
    Buffer,
    HEAP_UPLOAD,
    HEAP_DEFAULT,
    HEAP_READBACK,
    Compute,
    Texture2D,
    Swapchain,
)
from compushady.shaders import hlsl
import glfw
import struct

compushady.config.set_debug(True)

for device in compushady.get_discovered_devices():
    print(
        device.name,
        device.dedicated_video_memory,
        device.shared_system_memory,
        device.is_hardware,
        device.is_discrete,
    )


buffer001 = Buffer(1024, HEAP_DEFAULT, stride=4)

texture = Texture2D(1024, 1024, B8G8R8A8_UNORM)

shader = """
float multiplier : register(b0); // CBV 0
RWBuffer<uint> destination : register(u0); // UAV 0
RWTexture2D<float4> target : register(u1); // UAV 1
RWBuffer<float3> positions : register(u2); // UAV 2

float mandlebrot(float2 xy)
{
    const uint max_iterations = 100;
    xy = (xy - 0.5) * 2 - float2(1, 0);
    float2 z = float2(0, 0);
    for(uint i = 0; i < max_iterations; i++)
    {
        z = float2(z.x * z.x - z.y * z.y, z.x * z.y * 2) + xy;
        if (length(z) > multiplier * 2) return float(i) / max_iterations;
    }

    return 1; // white
 }

void move_pivot(float3 pivot)
{
    positions[index] *= pivot;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    destination[tid.x] = tid.x;
    //target[tid.xy] = float4(1, 0, 0, 1);
    uint width;
    uint height;

    target.GetDimensions(width, height);

    float2 uv = tid.xy / float2(width, height);

    float m = mandlebrot(uv);

    target[tid.xy] = float4(m, 0, 0.5, 1);
}
"""

multiplier = Buffer(4, HEAP_UPLOAD)

compiled_shader = hlsl.compile(shader)

compute = Compute(compiled_shader, cbv=[multiplier], uav=[buffer001, texture])

compute.dispatch(texture.width // 8, texture.height // 8, 1)

buffer002 = Buffer(1024, HEAP_READBACK)

buffer001.copy_to(buffer002)

print(buffer002.readback())

buffer003 = Buffer(texture.size, HEAP_READBACK)
texture.copy_to(buffer003)
print(buffer003.readback(1024))

glfw.init()
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

window = glfw.create_window(texture.width, texture.height, "Compute Shader", None, None)

swapchain = Swapchain(glfw.get_win32_window(window), B8G8R8A8_UNORM, 3)

value = 0
while not glfw.window_should_close(window):
    glfw.poll_events()
    multiplier.upload(struct.pack('f', value))
    compute.dispatch(texture.width // 8, texture.height // 8, 1)
    swapchain.present(texture)
    value += 0.01
    if value > 1:
        value = 0
