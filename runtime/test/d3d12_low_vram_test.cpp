// Live GPU stress test for low-VRAM survival mode (VK_EXT_memory_budget) and LEONIDA_RT_AS=off.
// Proves on real hardware (e.g. 4 GB GTX 1650) that:
//   1. Low-VRAM mode enables cleanly (auto on <= 4.5 GiB discrete GPUs, or via LEONIDA_LOW_VRAM=1).
//   2. Allocating spillable 2D textures past the VRAM headroom reserve steers overflow textures to system memory
//      without failing E_OUTOFMEMORY, while keeping critical Render Targets and Depth Buffers pinned in DEVICE_LOCAL.
//   3. LEONIDA_RT_AS=off cleanly disables software/hardware RT acceleration structure builds.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../shims/d3d12_com.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                     \
            return 1;                                                                                \
        }                                                                                            \
    } while (0)

namespace {

HRESULT create_texture(ID3D12Device* dev, UINT width, UINT height, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags,
                       D3D12_RESOURCE_STATES initial_state, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = fmt;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    return dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial_state, nullptr,
                                        IID_PPV_ARGS(out));
}

}  // namespace

int main() {
    // Part 1: Verify Low-VRAM mode + proactive budget steering + critical RT/DS VRAM pinning.
    // Force LEONIDA_LOW_VRAM=1 only if unset? On <=4.5 GiB GPUs (like GTX 1650) it auto-enables with no env var!
    // We also pick a high enough reserve when running on >4.5 GiB GPUs so the stress test stays fast (< 2 GiB).
    ID3D12Device* dev = nullptr;
    HRESULT hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev));
    if (FAILED(hr) || !dev) {
        std::printf("SKIP (no Vulkan device: 0x%08lx)\n", static_cast<unsigned long>(hr));
        return 77;
    }

    auto stats = rt::d3d12::device_vram_stats(dev);
    std::printf("Device VRAM: %llu MiB | low_vram_mode=%d (auto=%d) | budget=%d priority=%d pageable=%d | reserve=%llu MiB\n",
                static_cast<unsigned long long>(stats.device_local_bytes >> 20), int(stats.low_vram_mode),
                int(stats.auto_detected), int(stats.memory_budget_ext), int(stats.memory_priority_ext),
                int(stats.pageable_memory_ext), static_cast<unsigned long long>(stats.reserve_bytes >> 20));

    if (!stats.memory_budget_ext) {
        std::printf("SKIP (GPU/driver lacks VK_EXT_memory_budget)\n");
        dev->Release();
        return 77;
    }

    // On a <= 4.5 GiB discrete GPU, auto-detection MUST turn low_vram_mode on automatically.
    if (stats.device_local_bytes > 0 && stats.device_local_bytes <= (4608ull << 20)) {
        CHECK(stats.low_vram_mode);
        CHECK(stats.reserve_bytes >= (64ull << 20));
    }

    // If running on a huge (> 4.5 GiB) GPU where auto-mode stayed off, re-create with LEONIDA_LOW_VRAM=1 and a large
    // reserve so we can still verify budget steering quickly without allocating 16 GiB.
    if (!stats.low_vram_mode) {
        dev->Release();
        dev = nullptr;
        _putenv_s("LEONIDA_LOW_VRAM", "1");
        _putenv_s("LEONIDA_VRAM_RESERVE_MB", "2048");
        hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev));
        CHECK(SUCCEEDED(hr) && dev);
        stats = rt::d3d12::device_vram_stats(dev);
        CHECK(stats.low_vram_mode);
    }

    // Allocate 2048x2048 RGBA8 textures (16 MiB each) up to VRAM + 512 MiB (capped at 5120 MiB) while interleaving
    // critical 1920x1080 Render Targets & Depth Buffers to prove critical resources stay in DEVICE_LOCAL VRAM even
    // after spillable textures begin steering to system memory.
    const std::uint64_t target_mib =
        std::min<std::uint64_t>((stats.device_local_bytes >> 20) + 512, 5120);
    const std::size_t tex_count = static_cast<std::size_t>(target_mib / 16);

    std::vector<ID3D12Resource*> spillable_textures;
    std::vector<ID3D12Resource*> critical_targets;
    spillable_textures.reserve(tex_count);

    std::size_t spillable_in_vram = 0;
    std::size_t spillable_in_sysmem = 0;
    std::size_t critical_in_vram = 0;
    std::size_t critical_in_sysmem = 0;

    for (std::size_t i = 0; i < tex_count; ++i) {
        ID3D12Resource* tex = nullptr;
        hr = create_texture(dev, 2048, 2048, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE,
                            D3D12_RESOURCE_STATE_COMMON, &tex);
        CHECK(SUCCEEDED(hr) && tex);
        const VkMemoryPropertyFlags flags = rt::d3d12::vulkan_memory_flags(tex);
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            ++spillable_in_vram;
        else
            ++spillable_in_sysmem;
        spillable_textures.push_back(tex);

        // Periodically allocate a critical Render Target and Depth Buffer even AFTER spillover kicks in!
        if (i == 0 || i == tex_count / 2 || i == tex_count - 1) {
            ID3D12Resource* rt = nullptr;
            hr = create_texture(dev, 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                D3D12_RESOURCE_STATE_RENDER_TARGET, &rt);
            CHECK(SUCCEEDED(hr) && rt);
            if (rt::d3d12::vulkan_memory_flags(rt) & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                ++critical_in_vram;
            else
                ++critical_in_sysmem;
            critical_targets.push_back(rt);

            ID3D12Resource* ds = nullptr;
            hr = create_texture(dev, 1920, 1080, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                                D3D12_RESOURCE_STATE_DEPTH_WRITE, &ds);
            CHECK(SUCCEEDED(hr) && ds);
            if (rt::d3d12::vulkan_memory_flags(ds) & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                ++critical_in_vram;
            else
                ++critical_in_sysmem;
            critical_targets.push_back(ds);
        }
    }

    const auto after = rt::d3d12::device_vram_stats(dev);
    std::printf("Stress result (%zu x 16 MiB = %llu MiB spillable + %zu critical RT/DS):\n"
                "  Spillable textures : %zu in VRAM, %zu steered/spilled to system RAM\n"
                "  Critical RT/DS     : %zu in VRAM, %zu in system RAM\n"
                "  Stats              : steered=%llu (%llu MiB), forced_spills=%llu (%llu MiB)\n",
                tex_count, static_cast<unsigned long long>(tex_count * 16), critical_targets.size(),
                spillable_in_vram, spillable_in_sysmem, critical_in_vram, critical_in_sysmem,
                static_cast<unsigned long long>(after.steered_allocations),
                static_cast<unsigned long long>(after.steered_bytes >> 20),
                static_cast<unsigned long long>(after.forced_spill_allocations),
                static_cast<unsigned long long>(after.forced_spill_bytes >> 20));

    // All critical RT/DS targets MUST stay pinned in DEVICE_LOCAL VRAM!
    CHECK(critical_in_vram == critical_targets.size());
    CHECK(critical_in_sysmem == 0);
    // Overflow textures past the VRAM reserve MUST have been proactively steered to system memory!
    CHECK(after.steered_allocations > 0);
    CHECK(spillable_in_sysmem == after.steered_allocations + after.forced_spill_allocations);

    for (auto* r : critical_targets) r->Release();
    for (auto* r : spillable_textures) r->Release();
    dev->Release();

    // Part 2: Verify LEONIDA_RT_AS=off cleanly disables RT acceleration structures AND returns safe 256-byte
    // minimal prebuild sizes (never 0) so titles querying prebuild sizes do not crash.
    _putenv_s("LEONIDA_RT_AS", "off");
    ID3D12Device* dev_no_rt = nullptr;
    hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev_no_rt));
    CHECK(SUCCEEDED(hr) && dev_no_rt);
    const auto* info = rt::d3d12::vulkan_device_info(dev_no_rt);
    CHECK(info != nullptr);
    CHECK(!info->xdxr_acceleration_structures);
    CHECK(!info->ray_tracing);
    CHECK(info->rt_disabled);
    ID3D12Device5* dev5 = nullptr;
    CHECK(SUCCEEDED(dev_no_rt->QueryInterface(IID_PPV_ARGS(&dev5))) && dev5);
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.NumDescs = 4;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild{};
    dev5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuild);
    CHECK(prebuild.ResultDataMaxSizeInBytes == 256);
    CHECK(prebuild.ScratchDataSizeInBytes == 256);
    CHECK(prebuild.UpdateScratchDataSizeInBytes == 256);
    dev5->Release();
    dev_no_rt->Release();

    std::printf("d3d12_low_vram_test: ALL CHECKS PASSED\n");
    return 0;
}
