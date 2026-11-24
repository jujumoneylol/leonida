// Live GPU host-parity test for GPU-driven graphics ExecuteIndirect (D3D-EXECIND-GRAPHICS):
//   - ExecuteIndirect with D3D12_INDIRECT_ARGUMENT_TYPE_DRAW (fixed count 1, multi-draw count 2, padded stride 24)
//   - ExecuteIndirect with D3D12_INDIRECT_ARGUMENT_TYPE_DRAW + GPU CountBuffer (count = 0, 1, clamped 5 > max=2)
//   - ExecuteIndirect with D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED (multi-draw with BaseVertexLocation, padded stride)
//   - ExecuteIndirect with D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED + GPU CountBuffer (count = 1, 2)
//   - ExecuteIndirect with MaxCommandCount = 0 (no-op)
// Every case renders an 8x8 RGBA8_UNORM target with real vertex & index buffers on both our Vulkan device and the
// host Windows D3D12 runtime on the same adapter (matched by LUID), comparing all 64 pixels byte-for-byte.
// Self-contained: builds minimal SPIR-V 1.3 shaders in memory (no glslangValidator or llvm-mc needed) and compiles
// matching HLSL for the host with d3dcompiler_47. Synthetic data only.
#include "vk_test_device.h"
#include "host_wine.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using d3d12test::Ref;

#define CHECK(expr)                                                                                  \
    do {                                                                                             \
        if (!(expr)) {                                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);                     \
            return 1;                                                                                \
        }                                                                                            \
    } while (0)

struct Vertex {
    float x, y;
    float r, g, b, a;
};

void emit_op(std::vector<std::uint32_t>& out, std::uint16_t opcode, std::initializer_list<std::uint32_t> ops) {
    out.push_back((std::uint32_t(ops.size() + 1) << 16) | opcode);
    out.insert(out.end(), ops.begin(), ops.end());
}

// Minimal valid SPIR-V 1.3 vertex shader: reads vec2 POSITION (loc 0) and vec4 COLOR (loc 1), writes gl_Position and
// passes COLOR to fragment loc 0.
std::vector<std::uint32_t> make_vb_vs_spv() {
    std::vector<std::uint32_t> w = {0x07230203u, 0x00010300u, 0u, 32u, 0u};
    emit_op(w, 17, {1});                                       // OpCapability Shader
    emit_op(w, 14, {0, 1});                                    // OpMemoryModel Logical GLSL450
    // Entry point: Vertex %main(4) "main" %per_vertex(10) %out_col(11) %in_pos(12) %in_col(13).
    emit_op(w, 15, {0, 4, 0x6e69616du, 0u, 10, 11, 12, 13});
    // Decorations.
    emit_op(w, 71, {20, 2});                                   // OpDecorate %20 Block
    emit_op(w, 72, {20, 0, 11, 0});                            // OpMemberDecorate %20 0 BuiltIn Position
    emit_op(w, 71, {11, 30, 0});                               // OpDecorate %out_col Location 0
    emit_op(w, 71, {12, 30, 0});                               // OpDecorate %in_pos Location 0
    emit_op(w, 71, {13, 30, 1});                               // OpDecorate %in_col Location 1
    // Types & variables.
    emit_op(w, 19, {1});                                       // %1 = OpTypeVoid
    emit_op(w, 33, {2, 1});                                    // %2 = OpTypeFunction %1
    emit_op(w, 22, {3, 32});                                   // %3 = OpTypeFloat 32
    emit_op(w, 23, {5, 3, 2});                                 // %5 = OpTypeVector %3 2
    emit_op(w, 23, {6, 3, 4});                                 // %6 = OpTypeVector %3 4
    emit_op(w, 30, {20, 6});                                   // %20 = OpTypeStruct %6
    emit_op(w, 32, {21, 3, 20});                               // %21 = OpTypePointer Output %20
    emit_op(w, 32, {7, 3, 6});                                 // %7 = OpTypePointer Output %6
    emit_op(w, 32, {8, 1, 5});                                 // %8 = OpTypePointer Input %5
    emit_op(w, 32, {9, 1, 6});                                 // %9 = OpTypePointer Input %6
    emit_op(w, 21, {14, 32, 1});                               // %14 = OpTypeInt 32 1
    emit_op(w, 59, {21, 10, 3});                               // %10 = OpVariable %21 Output
    emit_op(w, 59, {7, 11, 3});                                // %11 = OpVariable %7 Output
    emit_op(w, 59, {8, 12, 1});                                // %12 = OpVariable %8 Input
    emit_op(w, 59, {9, 13, 1});                                // %13 = OpVariable %9 Input
    emit_op(w, 43, {14, 15, 0});                               // %15 = OpConstant %14 0
    emit_op(w, 43, {3, 16, std::bit_cast<std::uint32_t>(0.0f)});
    emit_op(w, 43, {3, 17, std::bit_cast<std::uint32_t>(1.0f)});
    // Function %main(4).
    emit_op(w, 54, {1, 4, 0, 2});
    emit_op(w, 248, {18});                                     // OpLabel
    emit_op(w, 61, {5, 22, 12});                               // %22 = OpLoad %5 %12 (pos2)
    emit_op(w, 81, {3, 23, 22, 0});                            // %23 = OpCompositeExtract x
    emit_op(w, 81, {3, 24, 22, 1});                            // %24 = OpCompositeExtract y
    emit_op(w, 80, {6, 25, 23, 24, 16, 17});                   // %25 = vec4(x, y, 0.0, 1.0)
    emit_op(w, 65, {7, 26, 10, 15});                           // %26 = OpAccessChain %7 %10 0
    emit_op(w, 62, {26, 25});                                  // OpStore %26 %25
    emit_op(w, 61, {6, 27, 13});                               // %27 = OpLoad %6 %13 (col)
    emit_op(w, 62, {11, 27});                                  // OpStore %11 %27
    emit_op(w, 253, {});
    emit_op(w, 56, {});
    return w;
}

// Minimal valid SPIR-V 1.3 fragment shader: passes vec4 COLOR (loc 0) straight to RT0 (loc 0).
std::vector<std::uint32_t> make_col_ps_spv() {
    std::vector<std::uint32_t> w = {0x07230203u, 0x00010300u, 0u, 16u, 0u};
    emit_op(w, 17, {1});                                       // OpCapability Shader
    emit_op(w, 14, {0, 1});                                    // OpMemoryModel Logical GLSL450
    emit_op(w, 15, {4, 4, 0x6e69616du, 0u, 8, 9});             // OpEntryPoint Fragment %4 "main" %8 %9
    emit_op(w, 16, {4, 7});                                    // OpExecutionMode %4 OriginUpperLeft
    emit_op(w, 71, {8, 30, 0});                                // OpDecorate %in_col Location 0
    emit_op(w, 71, {9, 30, 0});                                // OpDecorate %out_rt Location 0
    emit_op(w, 19, {1});                                       // %1 = OpTypeVoid
    emit_op(w, 33, {2, 1});                                    // %2 = OpTypeFunction %1
    emit_op(w, 22, {3, 32});                                   // %3 = OpTypeFloat 32
    emit_op(w, 23, {5, 3, 4});                                 // %5 = OpTypeVector %3 4
    emit_op(w, 32, {6, 1, 5});                                 // %6 = OpTypePointer Input %5
    emit_op(w, 32, {7, 3, 5});                                 // %7 = OpTypePointer Output %5
    emit_op(w, 59, {6, 8, 1});                                 // %8 = OpVariable %6 Input
    emit_op(w, 59, {7, 9, 3});                                 // %9 = OpVariable %7 Output
    emit_op(w, 54, {1, 4, 0, 2});
    emit_op(w, 248, {10});                                     // OpLabel
    emit_op(w, 61, {5, 11, 8});                                // %11 = OpLoad %5 %8
    emit_op(w, 62, {9, 11});                                   // OpStore %9 %11
    emit_op(w, 253, {});
    emit_op(w, 56, {});
    return w;
}

bool write_spv(const std::filesystem::path& path, const std::vector<std::uint32_t>& words) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    return bool(f);
}

constexpr char kHostVs[] =
    "struct VsIn { float2 pos : POSITION; float4 col : COLOR; };\n"
    "struct VsOut { float4 pos : SV_Position; float4 col : COLOR; };\n"
    "VsOut main(VsIn i) { VsOut o; o.pos = float4(i.pos, 0.0, 1.0); o.col = i.col; return o; }\n";
constexpr char kHostPs[] = "float4 main(float4 pos : SV_Position, float4 col : COLOR) : SV_Target { return col; }\n";

bool compile_hlsl(pD3DCompile compile, const char* source, std::vector<std::uint8_t>& out, bool vertex) {
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", vertex ? "vs_5_0" : "ps_5_0",
                               0, 0, &code, &errors);
    if (errors) {
        std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
        errors->Release();
    }
    if (FAILED(hr) || !code) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(code->GetBufferPointer());
    out.assign(bytes, bytes + code->GetBufferSize());
    code->Release();
    return true;
}

bool host_device_for_vk(const vktest::Device& vk, Ref<ID3D12Device>& out, std::string& name) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &props);
    Ref<IDXGIFactory4> factory;
    if (!id.deviceLUIDValid || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    Ref<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter.p)))) return false;
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    char utf8[256]{};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), nullptr, nullptr);
    name = utf8;
    return SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&out.p)));
}

struct Programs {
    std::vector<std::uint8_t> root, vs, ps;
};

struct IndirectCase {
    const char* name;
    bool indexed = false;
    bool use_count_buffer = false;
    UINT stride = 16;
    UINT max_count = 1;
    UINT gpu_count = 1;
    bool expect_left = false;
    bool expect_right = false;
};

// Left half of the 8x8 target (x in [-1, 0]): red (1, 0, 0, 1).
// Right half of the 8x8 target (x in [0, 1]): green (0, 1, 0, 1).
constexpr Vertex kVertices[12] = {
    // Vertices 0..5: left half quad (2 triangles) in red.
    {-1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
    {-1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},  {-1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
    // Vertices 6..11: right half quad (2 triangles) in green.
    {0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f},  {1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
    {0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f},  {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},  {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f},
};
// Index buffer for 2 triangles (0..5); BaseVertexLocation=0 draws left half, BaseVertexLocation=6 draws right half.
constexpr std::uint16_t kIndices[6] = {0, 1, 2, 3, 4, 5};

bool upload_bytes(ID3D12Resource* upload, const void* data, std::size_t size) {
    void* mapped = nullptr;
    const D3D12_RANGE empty{0, 0};
    if (FAILED(upload->Map(0, &empty, &mapped)) || !mapped) return false;
    std::memcpy(mapped, data, size);
    upload->Unmap(0, nullptr);
    return true;
}

bool run_case(ID3D12Device* dev, const Programs& p, const IndirectCase& c, std::vector<std::uint8_t>& pixels_out) {
    Ref<ID3D12RootSignature> root;
    if (FAILED(dev->CreateRootSignature(0, p.root.data(), p.root.size(), IID_PPV_ARGS(&root.p)))) return false;

    const D3D12_INPUT_ELEMENT_DESC layout[2] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.p;
    pd.VS = {p.vs.data(), p.vs.size()};
    pd.PS = {p.ps.data(), p.ps.size()};
    for (auto& rt : pd.BlendState.RenderTarget) {
        rt.SrcBlend = rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlend = rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOp = rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.DepthStencilState.DepthEnable = FALSE;
    pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    pd.InputLayout = {layout, 2};
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pd.SampleDesc = {1, 0};

    Ref<ID3D12PipelineState> pso;
    if (FAILED(dev->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso.p)))) return false;

    D3D12_INDIRECT_ARGUMENT_DESC arg_desc{};
    arg_desc.Type = c.indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    D3D12_COMMAND_SIGNATURE_DESC sig_desc{};
    sig_desc.ByteStride = c.stride;
    sig_desc.NumArgumentDescs = 1;
    sig_desc.pArgumentDescs = &arg_desc;
    Ref<ID3D12CommandSignature> sig;
    if (FAILED(dev->CreateCommandSignature(&sig_desc, nullptr, IID_PPV_ARGS(&sig.p)))) return false;

    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES up_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = td.Height = 8;
    td.DepthOrArraySize = td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc = {1, 0};
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    Ref<ID3D12Resource> target;
    if (FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                            nullptr, IID_PPV_ARGS(&target.p))))
        return false;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    dev->GetCopyableFootprints(&td, 0, 1, 0, &foot, &rows, &row_bytes, &total);
    Ref<ID3D12Resource> readback;
    const auto rb_desc = d3d12test::buffer_desc(total);
    if (FAILED(dev->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                            nullptr, IID_PPV_ARGS(&readback.p))))
        return false;

    // Upload buffers -> default buffers for VB, IB, indirect arguments, and optional count buffer.
    Ref<ID3D12Resource> vb_up, vb_def, ib_up, ib_def, arg_up, arg_def, cnt_up, cnt_def;
    const auto vb_desc = d3d12test::buffer_desc(sizeof(kVertices));
    const auto ib_desc = d3d12test::buffer_desc(sizeof(kIndices));
    const std::size_t arg_bytes = std::max<std::size_t>(256, std::size_t(c.stride) * 4);
    const auto arg_buf_desc = d3d12test::buffer_desc(arg_bytes);
    const auto cnt_buf_desc = d3d12test::buffer_desc(64);
    if (FAILED(dev->CreateCommittedResource(&up_heap, D3D12_HEAP_FLAG_NONE, &vb_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                            nullptr, IID_PPV_ARGS(&vb_up.p))) ||
        FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &vb_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                            nullptr, IID_PPV_ARGS(&vb_def.p))) ||
        FAILED(dev->CreateCommittedResource(&up_heap, D3D12_HEAP_FLAG_NONE, &ib_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                            nullptr, IID_PPV_ARGS(&ib_up.p))) ||
        FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &ib_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                            nullptr, IID_PPV_ARGS(&ib_def.p))) ||
        FAILED(dev->CreateCommittedResource(&up_heap, D3D12_HEAP_FLAG_NONE, &arg_buf_desc,
                                            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&arg_up.p))) ||
        FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &arg_buf_desc,
                                            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&arg_def.p))) ||
        FAILED(dev->CreateCommittedResource(&up_heap, D3D12_HEAP_FLAG_NONE, &cnt_buf_desc,
                                            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cnt_up.p))) ||
        FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &cnt_buf_desc,
                                            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&cnt_def.p))))
        return false;

    std::vector<std::uint8_t> arg_data(arg_bytes, 0);
    if (c.indexed) {
        const D3D12_DRAW_INDEXED_ARGUMENTS a0{6, 1, 0, 0, 0};  // left half (BaseVertexLocation=0)
        const D3D12_DRAW_INDEXED_ARGUMENTS a1{6, 1, 0, 6, 0};  // right half (BaseVertexLocation=6)
        std::memcpy(arg_data.data(), &a0, sizeof(a0));
        std::memcpy(arg_data.data() + c.stride, &a1, sizeof(a1));
    } else {
        const D3D12_DRAW_ARGUMENTS a0{6, 1, 0, 0};  // left half (StartVertexLocation=0)
        const D3D12_DRAW_ARGUMENTS a1{6, 1, 6, 0};  // right half (StartVertexLocation=6)
        std::memcpy(arg_data.data(), &a0, sizeof(a0));
        std::memcpy(arg_data.data() + c.stride, &a1, sizeof(a1));
    }
    const std::uint32_t count_val = c.gpu_count;
    if (!upload_bytes(vb_up.p, kVertices, sizeof(kVertices)) || !upload_bytes(ib_up.p, kIndices, sizeof(kIndices)) ||
        !upload_bytes(arg_up.p, arg_data.data(), arg_data.size()) || !upload_bytes(cnt_up.p, &count_val, sizeof(count_val)))
        return false;

    Ref<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC rh{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    if (FAILED(dev->CreateDescriptorHeap(&rh, IID_PPV_ARGS(&rtv_heap.p)))) return false;
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(target.p, nullptr, rtv);

    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, pso.p, IID_PPV_ARGS(&list.p))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;

    list->CopyResource(vb_def.p, vb_up.p);
    list->CopyResource(ib_def.p, ib_up.p);
    list->CopyResource(arg_def.p, arg_up.p);
    list->CopyResource(cnt_def.p, cnt_up.p);
    const D3D12_RESOURCE_BARRIER barriers[4] = {
        d3d12test::transition(vb_def.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
        d3d12test::transition(ib_def.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER),
        d3d12test::transition(arg_def.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        d3d12test::transition(cnt_def.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
    };
    list->ResourceBarrier(4, barriers);

    const FLOAT clear_blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    list->ClearRenderTargetView(rtv, clear_blue, 0, nullptr);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_VIEWPORT vp{0.0f, 0.0f, 8.0f, 8.0f, 0.0f, 1.0f};
    const D3D12_RECT sc{0, 0, 8, 8};
    list->RSSetViewports(1, &vp);
    list->RSSetScissorRects(1, &sc);
    list->SetGraphicsRootSignature(root.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vb_def->GetGPUVirtualAddress(), sizeof(kVertices), sizeof(Vertex)};
    list->IASetVertexBuffers(0, 1, &vbv);
    if (c.indexed) {
        const D3D12_INDEX_BUFFER_VIEW ibv{ib_def->GetGPUVirtualAddress(), sizeof(kIndices), DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&ibv);
    }

    list->ExecuteIndirect(sig.p, c.max_count, arg_def.p, 0, c.use_count_buffer ? cnt_def.p : nullptr, 0);

    const auto to_src = d3d12test::transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                              D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_src);
    D3D12_TEXTURE_COPY_LOCATION src_loc{target.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    src_loc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION dst_loc{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst_loc.PlacedFootprint = foot;
    list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
    if (FAILED(list->Close())) return false;

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;

    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
    if (FAILED(readback->Map(0, &range, &mapped)) || !mapped) return false;
    pixels_out.resize(8 * 8 * 4);
    for (UINT y = 0; y < 8; ++y) {
        std::memcpy(pixels_out.data() + y * 32,
                    static_cast<const std::uint8_t*>(mapped) + foot.Offset + y * foot.Footprint.RowPitch, 32);
    }
    const D3D12_RANGE empty{0, 0};
    readback->Unmap(0, &empty);
    return true;
}

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    const auto vs_program = d3d12test::fssm_program({0x5653e101u, 0xbf810000u});
    const auto ps_program = d3d12test::fssm_program({0x5053e101u, 0xbf810000u});
    const auto library = std::filesystem::temp_directory_path() /
                         ("leonida_indirect_draw_test_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(library);
    CHECK(write_spv(library / "vs.spv", make_vb_vs_spv()));
    CHECK(write_spv(library / "ps.spv", make_col_ps_spv()));
    {
        std::ofstream index(library / "index.json", std::ios::binary | std::ios::trunc);
        index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 0, \"system_value\": 0},"
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 1, \"system_value\": 0}]},"
                 " \"ps\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"inputs\": [0], \"outputs\": [0]}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.bindless = vk.bindless;
    info.bindless_partially_bound = vk.partially_bound;
    info.host_image_copy = vk.host_image_copy;
    info.buffer_device_address = vk.buffer_device_address;
    info.features = vk.enabled;
    info.shader_dir = library.string();

    Ref<ID3D12Device> ours;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    Programs our_programs;
    our_programs.root = d3d12test::serialize_root_signature(root_desc);
    CHECK(!our_programs.root.empty());
    our_programs.vs = d3d12test::shader_container(vs_program);
    our_programs.ps = d3d12test::shader_container(ps_program);

    Ref<ID3D12Device> host;
    std::string adapter;
    Programs host_programs;
    bool have_host = false;
    if (!hostwine::host_is_wine() && host_device_for_vk(vk, host, adapter)) {
        const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
                                            GetProcAddress(compiler, "D3DCompile")))
                                      : nullptr;
        host_programs.root = our_programs.root;
        if (compile && compile_hlsl(compile, kHostVs, host_programs.vs, true) &&
            compile_hlsl(compile, kHostPs, host_programs.ps, false)) {
            have_host = true;
            std::printf("host reference: %s\n", adapter.c_str());
        }
    }

    const IndirectCase cases[] = {
        {"DRAW fixed max=1 stride=16 (left red)", false, false, 16, 1, 1, true, false},
        {"DRAW multi max=2 stride=16 (left red + right green)", false, false, 16, 2, 2, true, true},
        {"DRAW multi max=2 padded stride=24", false, false, 24, 2, 2, true, true},
        {"DRAW count=0 max=2 (all clear blue)", false, true, 16, 2, 0, false, false},
        {"DRAW count=1 max=2 (left red only)", false, true, 16, 2, 1, true, false},
        {"DRAW count=5 clamped to max=2 (both)", false, true, 16, 2, 5, true, true},
        {"DRAW_INDEXED multi max=2 stride=20 (BaseVertex)", true, false, 20, 2, 2, true, true},
        {"DRAW_INDEXED multi max=2 padded stride=28", true, false, 28, 2, 2, true, true},
        {"DRAW_INDEXED count=1 max=2 (left red only)", true, true, 20, 2, 1, true, false},
        {"DRAW_INDEXED count=2 max=2 (both)", true, true, 20, 2, 2, true, true},
        {"DRAW max=0 no-op (all clear blue)", false, false, 16, 0, 0, false, false},
    };

    for (const IndirectCase& c : cases) {
        std::vector<std::uint8_t> ours_px;
        CHECK(run_case(ours.p, our_programs, c, ours_px));
        // Sample center of left half (x=2, y=4) and center of right half (x=6, y=4).
        const std::uint8_t* left = ours_px.data() + (4 * 8 + 2) * 4;
        const std::uint8_t* right = ours_px.data() + (4 * 8 + 6) * 4;
        const bool left_ok = c.expect_left ? (left[0] == 255 && left[1] == 0 && left[2] == 0)
                                           : (left[0] == 0 && left[1] == 0 && left[2] == 255);
        const bool right_ok = c.expect_right ? (right[0] == 0 && right[1] == 255 && right[2] == 0)
                                             : (right[0] == 0 && right[1] == 0 && right[2] == 255);
        std::printf("  %-52s L=(%3u,%3u,%3u) R=(%3u,%3u,%3u)", c.name, left[0], left[1], left[2], right[0], right[1],
                    right[2]);
        CHECK(left_ok && right_ok);
        if (have_host) {
            std::vector<std::uint8_t> host_px;
            CHECK(run_case(host.p, host_programs, c, host_px));
            CHECK(ours_px == host_px);
            std::printf(" [64/64 px EXACT vs host]");
        }
        std::printf(" OK\n");
    }

    std::filesystem::remove_all(library);
    CHECK(vk.validation_errors() == 0);
    std::printf("d3d12_indirect_draw_test: ALL %zu CASES PASSED%s\n", std::size(cases),
                have_host ? " (EXACT MATCH vs host D3D12)" : "");
    return 0;
}
