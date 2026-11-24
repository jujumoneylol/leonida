#pragma once

// Memory-type policy for the D3D12 layer that depends only on VkPhysicalDeviceMemoryProperties, so it can be checked
// without a GPU (d3d12_state_map_test) on the heap layouts real cards report.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace leonida {

// Without Resizable BAR the CPU sees at most a 256 MiB window of VRAM: the DEVICE_LOCAL | HOST_VISIBLE type sits in a
// heap of that size or smaller (256 MiB on AMD and on RADV, 214 MiB on an RTX 3090 with driver 616.56). Anything
// larger is Resizable BAR (or Smart Access Memory), or a GPU whose memory is all device-local and host-visible.
inline constexpr VkDeviceSize kBarWindowBytes = VkDeviceSize(256) << 20;

// Discrete GPUs with at most 4.5 GiB in their largest DEVICE_LOCAL heap (e.g. 4 GB GTX 1650 / RX 6400 / RTX 3050
// Laptop) automatically enable budget-aware spillover unless LEONIDA_LOW_VRAM=0 turns it off.
inline constexpr VkDeviceSize kLowVramAutoThresholdBytes = VkDeviceSize(4608) << 20;
inline constexpr VkDeviceSize kDefaultVramReserveMiB = 512;
inline constexpr VkDeviceSize kMinVramReserveMiB = 64;
inline constexpr VkDeviceSize kMaxVramReserveMiB = 2048;

// Allocation priority for budget-aware placement on low-VRAM GPUs.
//   critical : render targets, depth-stencil, MSAA, swapchain, bindless tables, scratch. Always prefers DEVICE_LOCAL.
//   spillable: sampled textures, default-heap buffers, sparse tile heaps. Proactively steered to system memory once
//              device-local usage would cross into the VRAM headroom reserve.
enum class MemoryPriority {
    critical,
    spillable,
};

struct VramPolicy {
    bool enabled = false;
    bool auto_detected = false;
    VkDeviceSize device_local_bytes = 0;  // largest DEVICE_LOCAL heap on the physical device
    VkDeviceSize reserve_bytes = 0;       // headroom kept free in DEVICE_LOCAL for critical resources
    VkDeviceSize vram_cap_bytes = 0;      // optional explicit soft ceiling (LEONIDA_VRAM_CAP_MB), 0 = uncapped
};

// VK_EXT_memory_priority hint for Vulkan's kernel/WDDM memory pager: critical render targets and depth buffers get
// maximum residency priority (1.0), while spillable textures/buffers/tile heaps yield first under pressure (0.25).
inline float memory_allocation_priority(MemoryPriority priority) {
    return priority == MemoryPriority::critical ? 1.0f : 0.25f;
}

// Largest DEVICE_LOCAL heap reported by the physical device (avoids double-counting the small <=256 MiB BAR window
// heap that non-ReBAR NVIDIA drivers expose alongside the main VRAM heap).
inline VkDeviceSize max_device_local_heap_bytes(const VkPhysicalDeviceMemoryProperties& memory) {
    VkDeviceSize largest = 0;
    for (std::uint32_t i = 0; i < memory.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; ++i) {
        const VkMemoryHeap& heap = memory.memoryHeaps[i];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && heap.size > largest) largest = heap.size;
    }
    return largest;
}

// Pure policy decision from heap layout, UMA flag, LEONIDA_LOW_VRAM ("1"/"0"/null), LEONIDA_VRAM_RESERVE_MB, and
// optional LEONIDA_VRAM_CAP_MB (explicit soft VRAM ceiling in MiB, clamped 256..65536).
inline VramPolicy decide_vram_policy(const VkPhysicalDeviceMemoryProperties& memory, bool uma,
                                     const char* low_vram_env, const char* reserve_mb_env,
                                     const char* cap_mb_env = nullptr) {
    VramPolicy policy{};
    policy.device_local_bytes = max_device_local_heap_bytes(memory);

    const bool cap_set = cap_mb_env && *cap_mb_env;
    if (cap_set) {
        const VkDeviceSize cap_mib = std::clamp<VkDeviceSize>(std::strtoull(cap_mb_env, nullptr, 10), 256, 65536);
        policy.vram_cap_bytes = cap_mib << 20;
    }

    const bool env_set = low_vram_env && *low_vram_env;
    if (env_set) {
        const bool off = std::strcmp(low_vram_env, "0") == 0 || std::strcmp(low_vram_env, "off") == 0 ||
                         std::strcmp(low_vram_env, "false") == 0;
        policy.enabled = !off;
        policy.auto_detected = false;
    } else if (cap_set) {
        policy.enabled = true;
        policy.auto_detected = false;
    } else if (!uma && policy.device_local_bytes > 0 && policy.device_local_bytes <= kLowVramAutoThresholdBytes) {
        policy.enabled = true;
        policy.auto_detected = true;
    }

    if (policy.enabled) {
        VkDeviceSize reserve_mib = kDefaultVramReserveMiB;
        if (reserve_mb_env && *reserve_mb_env) {
            reserve_mib = std::clamp<VkDeviceSize>(std::strtoull(reserve_mb_env, nullptr, 10), kMinVramReserveMiB,
                                                   kMaxVramReserveMiB);
        } else if (policy.device_local_bytes > 0) {
            // Never reserve more than a quarter of total VRAM on unusually small synthetic/virtual heaps.
            const VkDeviceSize quarter_mib = (policy.device_local_bytes >> 20) / 4;
            if (quarter_mib > 0 && reserve_mib > quarter_mib) reserve_mib = std::max<VkDeviceSize>(quarter_mib, kMinVramReserveMiB);
        }
        policy.reserve_bytes = reserve_mib << 20;
    }
    return policy;
}

// True when the device has DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory in a device-local heap larger than
// the 256 MiB BAR window (D3D-96): then CPU-written memory that the GPU reads, such as an upload heap, can live in
// VRAM without exhausting a small window that every such allocation would share.
inline bool resizable_bar(const VkPhysicalDeviceMemoryProperties& memory) {
    constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
        const VkMemoryType& type = memory.memoryTypes[i];
        if ((type.propertyFlags & wanted) != wanted || type.heapIndex >= memory.memoryHeapCount) continue;
        const VkMemoryHeap& heap = memory.memoryHeaps[type.heapIndex];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && heap.size > kBarWindowBytes) return true;
    }
    return false;
}

// The memory types to try for an allocation of `size` bytes, best first: those with `required | preferred`, then
// those with only `required`. A type whose heap is smaller than the allocation is left out, as the allocation would
// break VUID-vkAllocateMemory-pAllocateInfo-01713 (some drivers still return VK_SUCCESS). Without Resizable BAR the
// heap of DEVICE_LOCAL | HOST_VISIBLE memory is typically 256 MiB or less, so a larger host-visible allocation falls
// through to system memory. A heap that is big enough but already partly used is left to the driver: a failed
// vkAllocateMemory moves on to the next type.
inline std::vector<std::uint32_t> memory_type_order(const VkPhysicalDeviceMemoryProperties& memory,
                                                    std::uint32_t type_bits, VkMemoryPropertyFlags required,
                                                    VkMemoryPropertyFlags preferred, VkDeviceSize size) {
    std::vector<std::uint32_t> order;
    for (const VkMemoryPropertyFlags flags : {required | preferred, required}) {
        for (std::uint32_t i = 0; i < memory.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
            const VkMemoryType& type = memory.memoryTypes[i];
            if (!(type_bits & (1u << i)) || (type.propertyFlags & flags) != flags) continue;
            if (type.heapIndex >= memory.memoryHeapCount || memory.memoryHeaps[type.heapIndex].size < size) continue;
            if (std::find(order.begin(), order.end(), i) == order.end()) order.push_back(i);
        }
    }
    return order;
}

// D3D-299: true when an allocation that preferred DEVICE_LOCAL without needing host access (a texture, a default-heap
// buffer outside the arena) got a type without it, so the GPU now reads it from system memory. A host-visible
// allocation (the arena, an upload heap) in system memory is the memory type policy without Resizable BAR, not this.
inline bool spilled_to_host(VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred,
                            VkMemoryPropertyFlags chosen) {
    return (preferred & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
           !(chosen & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// Budget-aware memory type ordering for low-VRAM GPUs (VK_EXT_memory_budget).
// When `policy.enabled`, `priority == spillable`, the caller prefers DEVICE_LOCAL without requiring host access, and
// `budget` is non-null: any DEVICE_LOCAL type whose heap would breach `heapBudget - policy.reserve_bytes` is demoted
// *behind* non-device-local (system memory) types instead of tried first. Critical allocations (render targets, depth
// buffers, MSAA) always keep DEVICE_LOCAL first; demoted DEVICE_LOCAL types remain at the back as a last-resort
// fallback so a valid allocation is never needlessly refused.
inline std::vector<std::uint32_t> memory_type_order_budgeted(
    const VkPhysicalDeviceMemoryProperties& memory, std::uint32_t type_bits, VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred, VkDeviceSize size, MemoryPriority priority,
    const VkPhysicalDeviceMemoryBudgetPropertiesEXT* budget, const VramPolicy& policy, bool* steered = nullptr) {
    if (steered) *steered = false;
    std::vector<std::uint32_t> base = memory_type_order(memory, type_bits, required, preferred, size);
    const bool check_budget = policy.enabled && priority == MemoryPriority::spillable && budget != nullptr &&
                              (preferred & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                              !(required & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                              !(required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (!check_budget || base.size() < 2) return base;

    std::vector<std::uint32_t> head_local;
    std::vector<std::uint32_t> system_mem;
    std::vector<std::uint32_t> demoted_local;
    for (const std::uint32_t i : base) {
        const VkMemoryType& type = memory.memoryTypes[i];
        const bool is_local = (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        if (!is_local) {
            system_mem.push_back(i);
            continue;
        }
        const std::uint32_t h = type.heapIndex;
        VkDeviceSize heap_budget = h < VK_MAX_MEMORY_HEAPS ? budget->heapBudget[h] : 0;
        if (policy.vram_cap_bytes > 0 && heap_budget > policy.vram_cap_bytes) heap_budget = policy.vram_cap_bytes;
        const VkDeviceSize heap_usage = h < VK_MAX_MEMORY_HEAPS ? budget->heapUsage[h] : 0;
        const VkDeviceSize available = heap_budget > heap_usage ? (heap_budget - heap_usage) : 0;
        const VkDeviceSize needed = (size > UINT64_MAX - policy.reserve_bytes) ? UINT64_MAX : (size + policy.reserve_bytes);
        if (available < needed) {
            demoted_local.push_back(i);
        } else {
            head_local.push_back(i);
        }
    }
    if (head_local.empty() && !system_mem.empty() && !demoted_local.empty()) {
        if (steered) *steered = true;
        std::vector<std::uint32_t> out;
        out.reserve(base.size());
        out.insert(out.end(), system_mem.begin(), system_mem.end());
        out.insert(out.end(), demoted_local.begin(), demoted_local.end());
        return out;
    }
    if (!demoted_local.empty() && !head_local.empty()) {
        std::vector<std::uint32_t> out;
        out.reserve(base.size());
        out.insert(out.end(), head_local.begin(), head_local.end());
        out.insert(out.end(), system_mem.begin(), system_mem.end());
        out.insert(out.end(), demoted_local.begin(), demoted_local.end());
        return out;
    }
    return base;
}

}  // namespace leonida
