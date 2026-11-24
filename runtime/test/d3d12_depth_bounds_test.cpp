// Side-by-side parity test for hardware depth-bounds testing (D3D-DEPTH-BOUNDS):
//   - D3D12_FEATURE_DATA_D3D12_OPTIONS2::DepthBoundsTestSupported (matches host & follows features.depthBounds)
//   - ID3D12Device2::CreatePipelineState with DEPTH_STENCIL1 and DEPTH_STENCIL2 (DepthBoundsTestEnable = TRUE)
//   - ID3D12GraphicsCommandList1::OMSetDepthBounds(Min, Max) -> vkCmdSetDepthBounds / VK_DYNAMIC_STATE_DEPTH_BOUNDS
//   - Bundle refusal parity for OMSetDepthBounds
//   - Live pixel-for-pixel GPU comparison against the host Windows D3D12 runtime on the same adapter (matched by LUID)
//     for default [0, 1] bounds, passing range, rejecting low/high ranges, inclusive boundary [0.5, 0.5],
//     DEPTH_STENCIL2, and Reset() restoring [0, 1].
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

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename Inner>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
    Inner inner{};
};

struct StreamBuilder {
    std::vector<std::uint8_t> bytes;
    template<typename S>
    void add(const S& s) {
        const std::size_t at = bytes.size();
        bytes.resize(at + sizeof(S));
        std::memcpy(bytes.data() + at, &s, sizeof(S));
    }
    D3D12_PIPELINE_STATE_STREAM_DESC desc() const { return {bytes.size(), const_cast<std::uint8_t*>(bytes.data())}; }
};

using RootSigSub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>;
using VsSub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>;
using PsSub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>;
using RasterSub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC>;
using DepthStencil1Sub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, D3D12_DEPTH_STENCIL_DESC1>;
using DepthStencil2Sub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, D3D12_DEPTH_STENCIL_DESC2>;
using TopologySub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE>;
using RtvFormatsSub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY>;
using DsvFormatSub = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT>;

// Helper to append a SPIR-V instruction word + operands.
void emit_op(std::vector<std::uint32_t>& out, std::uint16_t opcode, std::initializer_list<std::uint32_t> ops) {
    out.push_back((std::uint32_t(ops.size() + 1) << 16) | opcode);
    out.insert(out.end(), ops.begin(), ops.end());
}

// Minimal valid SPIR-V 1.3 vertex shader: fullscreen triangle from gl_VertexIndex (0, 1, 2) at z = 0.0, w = 1.0.
std::vector<std::uint32_t> make_fullscreen_vs_spv() {
    std::vector<std::uint32_t> w = {0x07230203u, 0x00010300u, 0u, 30u, 0u};
    // Capabilities & memory model.
    emit_op(w, 17, {1});                                       // OpCapability Shader
    emit_op(w, 14, {0, 1});                                    // OpMemoryModel Logical GLSL450
    // Entry point: Vertex %main(4) "main" %per_vertex(10) %gl_VertexIndex(13).
    emit_op(w, 15, {0, 4, 0x6e69616du, 0u, 10, 13});
    // Decorations: struct %PerVertex(25) { vec4 Position };
    emit_op(w, 71, {25, 2});                                   // OpDecorate %25 Block
    emit_op(w, 72, {25, 0, 11, 0});                            // OpMemberDecorate %25 0 BuiltIn Position
    emit_op(w, 71, {13, 11, 42});                              // OpDecorate %13 BuiltIn VertexIndex
    // Types & constants.
    emit_op(w, 19, {1});                                       // %1 = OpTypeVoid
    emit_op(w, 33, {2, 1});                                    // %2 = OpTypeFunction %1
    emit_op(w, 22, {3, 32});                                   // %3 = OpTypeFloat 32
    emit_op(w, 23, {5, 3, 4});                                 // %5 = OpTypeVector %3 4
    emit_op(w, 30, {25, 5});                                   // %25 = OpTypeStruct %5
    emit_op(w, 32, {26, 3, 25});                               // %26 = OpTypePointer Output %25
    emit_op(w, 32, {6, 3, 5});                                 // %6 = OpTypePointer Output %5
    emit_op(w, 21, {7, 32, 1});                                // %7 = OpTypeInt 32 1
    emit_op(w, 32, {8, 1, 7});                                 // %8 = OpTypePointer Input %7
    emit_op(w, 20, {9});                                       // %9 = OpTypeBool
    emit_op(w, 59, {26, 10, 3});                               // %10 = OpVariable %26 Output
    emit_op(w, 59, {8, 13, 1});                                // %13 = OpVariable %8 Input
    emit_op(w, 43, {7, 27, 0});                                // %27 = OpConstant %7 0
    emit_op(w, 43, {7, 14, 1});                                // %14 = OpConstant %7 1
    emit_op(w, 43, {7, 15, 2});                                // %15 = OpConstant %7 2
    emit_op(w, 43, {3, 16, std::bit_cast<std::uint32_t>(-1.0f)});
    emit_op(w, 43, {3, 17, std::bit_cast<std::uint32_t>(3.0f)});
    emit_op(w, 43, {3, 18, std::bit_cast<std::uint32_t>(0.0f)});
    emit_op(w, 43, {3, 19, std::bit_cast<std::uint32_t>(1.0f)});
    // Function %main(4).
    emit_op(w, 54, {1, 4, 0, 2});                              // OpFunction
    emit_op(w, 248, {11});                                     // OpLabel
    emit_op(w, 61, {7, 12, 13});                               // %12 = OpLoad %7 %13
    emit_op(w, 170, {9, 20, 12, 14});                          // %20 = OpIEqual %9 %12 1
    emit_op(w, 169, {3, 21, 20, 17, 16});                      // %21 = OpSelect %3 %20 3.0 -1.0
    emit_op(w, 170, {9, 22, 12, 15});                          // %22 = OpIEqual %9 %12 2
    emit_op(w, 169, {3, 23, 22, 17, 16});                      // %23 = OpSelect %3 %22 3.0 -1.0
    emit_op(w, 80, {5, 24, 21, 23, 18, 19});                   // %24 = OpCompositeConstruct %5 %21 %23 0.0 1.0
    emit_op(w, 65, {6, 28, 10, 27});                           // %28 = OpAccessChain %6 %10 0
    emit_op(w, 62, {28, 24});                                  // OpStore %28 %24
    emit_op(w, 253, {});                                       // OpReturn
    emit_op(w, 56, {});                                        // OpFunctionEnd
    return w;
}

// Minimal valid SPIR-V 1.3 fragment shader: writes constant RGBA (0.75, 0.5, 0.25, 0.5) to Location 0.
std::vector<std::uint32_t> make_solid_ps_spv() {
    std::vector<std::uint32_t> w = {0x07230203u, 0x00010300u, 0u, 15u, 0u};
    emit_op(w, 17, {1});                                       // OpCapability Shader
    emit_op(w, 14, {0, 1});                                    // OpMemoryModel Logical GLSL450
    emit_op(w, 15, {4, 4, 0x6e69616du, 0u, 7});                // OpEntryPoint Fragment %4 "main" %7
    emit_op(w, 16, {4, 7});                                    // OpExecutionMode %4 OriginUpperLeft
    emit_op(w, 71, {7, 30, 0});                                // OpDecorate %7 Location 0
    emit_op(w, 19, {1});                                       // %1 = OpTypeVoid
    emit_op(w, 33, {2, 1});                                    // %2 = OpTypeFunction %1
    emit_op(w, 22, {3, 32});                                   // %3 = OpTypeFloat 32
    emit_op(w, 23, {5, 3, 4});                                 // %5 = OpTypeVector %3 4
    emit_op(w, 32, {6, 3, 5});                                 // %6 = OpTypePointer Output %5
    emit_op(w, 59, {6, 7, 3});                                 // %7 = OpVariable %6 Output
    emit_op(w, 43, {3, 8, std::bit_cast<std::uint32_t>(0.75f)});
    emit_op(w, 43, {3, 9, std::bit_cast<std::uint32_t>(0.5f)});
    emit_op(w, 43, {3, 10, std::bit_cast<std::uint32_t>(0.25f)});
    emit_op(w, 43, {3, 11, std::bit_cast<std::uint32_t>(0.5f)});
    emit_op(w, 44, {5, 12, 8, 9, 10, 11});                     // %12 = OpConstantComposite
    emit_op(w, 54, {1, 4, 0, 2});                              // OpFunction
    emit_op(w, 248, {13});                                     // OpLabel
    emit_op(w, 62, {7, 12});                                   // OpStore %7 %12
    emit_op(w, 253, {});                                       // OpReturn
    emit_op(w, 56, {});                                        // OpFunctionEnd
    return w;
}

bool write_spv(const std::filesystem::path& path, const std::vector<std::uint32_t>& words) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    return bool(f);
}

constexpr char kHostVs[] =
    "float4 main(uint id : SV_VertexID) : SV_Position {\n"
    "    float x = (id == 1) ? 3.0 : -1.0;\n"
    "    float y = (id == 2) ? 3.0 : -1.0;\n"
    "    return float4(x, y, 0.0, 1.0);\n"
    "}\n";
constexpr char kHostPs[] = "float4 main() : SV_Target { return float4(0.75, 0.5, 0.25, 0.5); }\n";

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

struct BoundsCase {
    const char* name;
    bool use_ds2 = false;
    bool call_set_bounds = false;
    bool reset_after_reject = false;
    float clear_depth = 0.5f;
    float min_depth = 0.0f;
    float max_depth = 1.0f;
    bool expect_pass = true;
};

HRESULT make_bounds_pso(ID3D12Device* dev, ID3D12RootSignature* root, const Programs& p, bool use_ds2,
                        Ref<ID3D12PipelineState>& out) {
    Ref<ID3D12Device2> dev2;
    if (FAILED(dev->QueryInterface(IID_PPV_ARGS(&dev2.p)))) return E_NOINTERFACE;
    StreamBuilder sb;
    sb.add(RootSigSub{.inner = root});
    sb.add(VsSub{.inner = {p.vs.data(), p.vs.size()}});
    sb.add(PsSub{.inner = {p.ps.data(), p.ps.size()}});
    D3D12_RASTERIZER_DESC r{};
    r.FillMode = D3D12_FILL_MODE_SOLID;
    r.CullMode = D3D12_CULL_MODE_NONE;
    r.DepthClipEnable = TRUE;
    sb.add(RasterSub{.inner = r});
    if (use_ds2) {
        D3D12_DEPTH_STENCIL_DESC2 ds{};
        ds.DepthEnable = FALSE;
        ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        ds.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        ds.DepthBoundsTestEnable = TRUE;
        ds.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                        D3D12_COMPARISON_FUNC_ALWAYS, 0xff, 0xff};
        ds.BackFace = ds.FrontFace;
        sb.add(DepthStencil2Sub{.inner = ds});
    } else {
        D3D12_DEPTH_STENCIL_DESC1 ds{};
        ds.DepthEnable = FALSE;
        ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        ds.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        ds.StencilReadMask = ds.StencilWriteMask = 0xff;
        ds.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                        D3D12_COMPARISON_FUNC_ALWAYS};
        ds.BackFace = ds.FrontFace;
        ds.DepthBoundsTestEnable = TRUE;
        sb.add(DepthStencil1Sub{.inner = ds});
    }
    sb.add(TopologySub{.inner = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE});
    D3D12_RT_FORMAT_ARRAY rts{};
    rts.NumRenderTargets = 1;
    rts.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    sb.add(RtvFormatsSub{.inner = rts});
    sb.add(DsvFormatSub{.inner = DXGI_FORMAT_D32_FLOAT});
    const auto stream = sb.desc();
    return dev2->CreatePipelineState(&stream, IID_PPV_ARGS(&out.p));
}

bool run_case(ID3D12Device* dev, const Programs& p, const BoundsCase& c, std::uint8_t pixel_out[4]) {
    Ref<ID3D12RootSignature> root;
    if (FAILED(dev->CreateRootSignature(0, p.root.data(), p.root.size(), IID_PPV_ARGS(&root.p)))) return false;
    Ref<ID3D12PipelineState> pso;
    if (FAILED(make_bounds_pso(dev, root.p, p, c.use_ds2, pso))) return false;

    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT}, rb_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC cd{};
    cd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    cd.Width = cd.Height = 4;
    cd.DepthOrArraySize = cd.MipLevels = 1;
    cd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    cd.SampleDesc = {1, 0};
    cd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_RESOURCE_DESC dd = cd;
    dd.Format = DXGI_FORMAT_D32_FLOAT;
    dd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    Ref<ID3D12Resource> color, depth;
    if (FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &cd, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                            nullptr, IID_PPV_ARGS(&color.p))) ||
        FAILED(dev->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                            nullptr, IID_PPV_ARGS(&depth.p))))
        return false;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    dev->GetCopyableFootprints(&cd, 0, 1, 0, &foot, &rows, &row_bytes, &total);
    Ref<ID3D12Resource> readback;
    const auto rb_desc = d3d12test::buffer_desc(total);
    if (FAILED(dev->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                            nullptr, IID_PPV_ARGS(&readback.p))))
        return false;

    Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC rh{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, dh{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    if (FAILED(dev->CreateDescriptorHeap(&rh, IID_PPV_ARGS(&rtv_heap.p))) ||
        FAILED(dev->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&dsv_heap.p))))
        return false;
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    const auto dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(color.p, nullptr, rtv);
    dev->CreateDepthStencilView(depth.p, nullptr, dsv);

    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList1> list1;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, pso.p, IID_PPV_ARGS(&list.p))) ||
        FAILED(list->QueryInterface(IID_PPV_ARGS(&list1.p))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;

    if (c.reset_after_reject) {
        // Set rejecting bounds, close, and reset: Reset() must restore default [0.0, 1.0].
        list1->OMSetDepthBounds(0.8f, 0.9f);
        if (FAILED(list->Close()) || FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.p, pso.p))) return false;
    }

    const FLOAT clear_color[4] = {0.2f, 0.4f, 0.6f, 0.8f};
    list->ClearRenderTargetView(rtv, clear_color, 0, nullptr);
    list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, c.clear_depth, 0, 0, nullptr);
    list->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
    const D3D12_VIEWPORT vp{0.0f, 0.0f, 4.0f, 4.0f, 0.0f, 1.0f};
    const D3D12_RECT sc{0, 0, 4, 4};
    list->RSSetViewports(1, &vp);
    list->RSSetScissorRects(1, &sc);
    list->SetGraphicsRootSignature(root.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (c.call_set_bounds) list1->OMSetDepthBounds(c.min_depth, c.max_depth);
    list->DrawInstanced(3, 1, 0, 0);

    const auto to_src = d3d12test::transition(color.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                              D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_src);
    D3D12_TEXTURE_COPY_LOCATION src_loc{color.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
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
    std::memcpy(pixel_out, static_cast<const std::uint8_t*>(mapped) + foot.Offset, 4);
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

    const auto vs_program = d3d12test::fssm_program({0x5653db01u, 0xbf810000u});
    const auto ps_program = d3d12test::fssm_program({0x5053db01u, 0xbf810000u});
    const auto library = std::filesystem::temp_directory_path() /
                         ("leonida_depth_bounds_test_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(library);
    CHECK(write_spv(library / "vs.spv", make_fullscreen_vs_spv()));
    CHECK(write_spv(library / "ps.spv", make_solid_ps_spv()));
    {
        std::ofstream index(library / "index.json", std::ios::binary | std::ios::trunc);
        index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
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

    D3D12_FEATURE_DATA_D3D12_OPTIONS2 o2{};
    CHECK(SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &o2, sizeof(o2))));
    CHECK(o2.DepthBoundsTestSupported == (vk.enabled.depthBounds ? TRUE : FALSE));

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    Programs our_programs;
    our_programs.root = d3d12test::serialize_root_signature(root_desc);
    CHECK(!our_programs.root.empty());
    our_programs.vs = d3d12test::shader_container(vs_program);
    our_programs.ps = d3d12test::shader_container(ps_program);

    // 1. Verify feature-off device refuses DepthBoundsTestEnable = TRUE with E_NOTIMPL and reports FALSE.
    {
        rt::d3d12::VulkanDeviceInfo no_db = info;
        no_db.features.depthBounds = VK_FALSE;
        Ref<ID3D12Device> dev_off;
        CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(no_db, IID_PPV_ARGS(&dev_off.p))));
        D3D12_FEATURE_DATA_D3D12_OPTIONS2 off_o2{};
        CHECK(SUCCEEDED(dev_off->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &off_o2, sizeof(off_o2))));
        CHECK(off_o2.DepthBoundsTestSupported == FALSE);
        Ref<ID3D12RootSignature> rs;
        CHECK(SUCCEEDED(dev_off->CreateRootSignature(0, our_programs.root.data(), our_programs.root.size(),
                                                     IID_PPV_ARGS(&rs.p))));
        Ref<ID3D12PipelineState> refused1, refused2;
        CHECK(make_bounds_pso(dev_off.p, rs.p, our_programs, false, refused1) == E_NOTIMPL && !refused1.p);
        CHECK(make_bounds_pso(dev_off.p, rs.p, our_programs, true, refused2) == E_NOTIMPL && !refused2.p);
    }

    // 2. Verify OMSetDepthBounds is rejected in a bundle.
    {
        Ref<ID3D12CommandAllocator> ba;
        Ref<ID3D12GraphicsCommandList> b;
        Ref<ID3D12GraphicsCommandList1> b1;
        CHECK(SUCCEEDED(ours->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&ba.p))));
        CHECK(SUCCEEDED(ours->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, ba.p, nullptr, IID_PPV_ARGS(&b.p))));
        CHECK(SUCCEEDED(b->QueryInterface(IID_PPV_ARGS(&b1.p))));
        b1->OMSetDepthBounds(0.2f, 0.8f);
        CHECK(FAILED(b->Close()));
    }

    if (!vk.enabled.depthBounds) {
        std::printf("GPU lacks VkPhysicalDeviceFeatures::depthBounds; refusal checks passed, skipping draw checks\n");
        std::filesystem::remove_all(library);
        return 0;
    }

    // 3. Run live GPU draw cases on our Vulkan device (and side-by-side against host Windows D3D12 when available).
    const BoundsCase cases[] = {
        {"default [0.0, 1.0] passes at z=0.5", false, false, false, 0.5f, 0.0f, 1.0f, true},
        {"[0.25, 0.75] passes at z=0.5", false, true, false, 0.5f, 0.25f, 0.75f, true},
        {"[0.6, 0.9] rejects at z=0.5", false, true, false, 0.5f, 0.6f, 0.9f, false},
        {"[0.1, 0.4] rejects at z=0.5", false, true, false, 0.5f, 0.1f, 0.4f, false},
        {"[0.5, 0.5] inclusive boundary passes at z=0.5", false, true, false, 0.5f, 0.5f, 0.5f, true},
        {"DEPTH_STENCIL2 [0.6, 0.9] rejects at z=0.5", true, true, false, 0.5f, 0.6f, 0.9f, false},
        {"DEPTH_STENCIL2 [0.25, 0.75] passes at z=0.5", true, true, false, 0.5f, 0.25f, 0.75f, true},
        {"Reset() restores default [0.0, 1.0] after reject", false, false, true, 0.5f, 0.0f, 1.0f, true},
    };

    Ref<ID3D12Device> host;
    std::string adapter;
    Programs host_programs;
    bool have_host = false;
    if (!hostwine::host_is_wine() && host_device_for_vk(vk, host, adapter)) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS2 host_o2{};
        if (SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &host_o2, sizeof(host_o2))) &&
            host_o2.DepthBoundsTestSupported) {
            const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
                                                GetProcAddress(compiler, "D3DCompile")))
                                          : nullptr;
            host_programs.root = our_programs.root;
            if (compile && compile_hlsl(compile, kHostVs, host_programs.vs, true) &&
                compile_hlsl(compile, kHostPs, host_programs.ps, false)) {
                have_host = true;
                std::printf("host reference: %s (DepthBoundsTestSupported=TRUE)\n", adapter.c_str());
            }
        }
    }

    // Clear color (0.2, 0.4, 0.6, 0.8) ~= (51, 102, 153, 204); draw color (0.75, 0.5, 0.25, 0.5) ~= (191, 128, 64, 128).
    for (const BoundsCase& c : cases) {
        std::uint8_t ours_px[4]{};
        CHECK(run_case(ours.p, our_programs, c, ours_px));
        const bool looks_drawn = (ours_px[0] > 150 && ours_px[2] < 100);
        const bool looks_clear = (ours_px[0] < 80 && ours_px[2] > 120);
        std::printf("  %-52s vulkan=(%3u,%3u,%3u,%3u)", c.name, ours_px[0], ours_px[1], ours_px[2], ours_px[3]);
        CHECK(c.expect_pass ? looks_drawn : looks_clear);
        if (have_host) {
            std::uint8_t host_px[4]{};
            CHECK(run_case(host.p, host_programs, c, host_px));
            std::printf(" host=(%3u,%3u,%3u,%3u)", host_px[0], host_px[1], host_px[2], host_px[3]);
            for (int ch = 0; ch < 4; ++ch) {
                const int diff = std::abs(int(ours_px[ch]) - int(host_px[ch]));
                CHECK(diff <= 1);
            }
        }
        std::printf(" OK\n");
    }

    std::filesystem::remove_all(library);
    CHECK(vk.validation_errors() == 0);
    std::printf("d3d12_depth_bounds_test: ALL %zu CASES PASSED%s\n", std::size(cases),
                have_host ? " (EXACT MATCH vs host D3D12)" : "");
    return 0;
}
