//! Vulkan C ABI definitions and structures for Vantage Vulkan ICD.
//! Matches Vulkan 1.0/1.3 specifications.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::ffi::{c_char, c_void};

pub const ICD_LOADER_MAGIC: usize = 0x01FACDE4;
pub const VANTAGE_VENDOR_ID: u32 = 0x10006;
pub const VANTAGE_DEVICE_ID: u32 = 0x0001;

pub type VkBool32 = u32;
pub type VkDeviceSize = u64;
pub type VkFlags = u32;
pub type VkSampleCountFlags = VkFlags;
pub type VkQueueFlags = VkFlags;
pub type VkMemoryPropertyFlags = VkFlags;
pub type VkMemoryHeapFlags = VkFlags;
pub type VkInstanceCreateFlags = VkFlags;
pub type VkDeviceCreateFlags = VkFlags;
pub type VkDeviceQueueCreateFlags = VkFlags;
pub type VkCommandPoolCreateFlags = VkFlags;
pub type VkCommandPoolResetFlags = VkFlags;
pub type VkCommandBufferResetFlags = VkFlags;

pub const VK_TRUE: VkBool32 = 1;
pub const VK_FALSE: VkBool32 = 0;

pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
pub const VK_UUID_SIZE: usize = 16;
pub const VK_MAX_EXTENSION_NAME_SIZE: usize = 256;
pub const VK_MAX_DESCRIPTION_SIZE: usize = 256;
pub const VK_MAX_MEMORY_TYPES: usize = 32;
pub const VK_MAX_MEMORY_HEAPS: usize = 16;

pub const VK_QUEUE_GRAPHICS_BIT: VkQueueFlags = 0x00000001;
pub const VK_QUEUE_COMPUTE_BIT: VkQueueFlags = 0x00000002;
pub const VK_QUEUE_TRANSFER_BIT: VkQueueFlags = 0x00000004;

pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlags = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlags = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlags = 0x00000004;
pub const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: VkMemoryHeapFlags = 0x00000001;

pub const fn VK_MAKE_API_VERSION(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

pub const VK_API_VERSION_1_0: u32 = VK_MAKE_API_VERSION(0, 1, 0, 0);
pub const VK_API_VERSION_1_1: u32 = VK_MAKE_API_VERSION(0, 1, 1, 0);
pub const VK_API_VERSION_1_2: u32 = VK_MAKE_API_VERSION(0, 1, 2, 0);
pub const VK_API_VERSION_1_3: u32 = VK_MAKE_API_VERSION(0, 1, 3, 0);

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkResult {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
    VK_EVENT_SET = 3,
    VK_EVENT_RESET = 4,
    VK_INCOMPLETE = 5,
    VK_ERROR_OUT_OF_HOST_MEMORY = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST = -4,
    VK_ERROR_MEMORY_MAP_FAILED = -5,
    VK_ERROR_LAYER_NOT_PRESENT = -6,
    VK_ERROR_EXTENSION_NOT_PRESENT = -7,
    VK_ERROR_FEATURE_NOT_PRESENT = -8,
    VK_ERROR_INCOMPATIBLE_DRIVER = -9,
    VK_ERROR_TOO_MANY_OBJECTS = -10,
    VK_ERROR_FORMAT_NOT_SUPPORTED = -11,
    VK_ERROR_UNKNOWN = -13,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkStructureType {
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6,
    VK_STRUCTURE_TYPE_BIND_SPARSE_INFO = 7,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO = 14,
    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO = 19,
    VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO = 20,
    VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO = 22,
    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO = 23,
    VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO = 24,
    VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO = 25,
    VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO = 26,
    VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO = 27,
    VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO = 28,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO = 37,
    VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO = 38,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO = 41,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO = 43,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkPhysicalDeviceType {
    VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
    VK_PHYSICAL_DEVICE_TYPE_CPU = 4,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkFormat {
    VK_FORMAT_UNDEFINED = 0,
    VK_FORMAT_R8G8B8A8_UNORM = 37,
    VK_FORMAT_B8G8R8A8_UNORM = 44,
    VK_FORMAT_D32_SFLOAT = 126,
    VK_FORMAT_S8_UINT = 127,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkCommandBufferLevel {
    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
    VK_COMMAND_BUFFER_LEVEL_SECONDARY = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkIndexType {
    VK_INDEX_TYPE_UINT16 = 0,
    VK_INDEX_TYPE_UINT32 = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkPipelineBindPoint {
    VK_PIPELINE_BIND_POINT_GRAPHICS = 0,
    VK_PIPELINE_BIND_POINT_COMPUTE = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkSubpassContents {
    VK_SUBPASS_CONTENTS_INLINE = 0,
    VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS = 1,
}

pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;

pub type VkDeviceMemory = u64;
pub type VkBuffer = u64;
pub type VkImage = u64;
pub type VkImageView = u64;
pub type VkShaderModule = u64;
pub type VkPipeline = u64;
pub type VkPipelineLayout = u64;
pub type VkRenderPass = u64;
pub type VkFramebuffer = u64;
pub type VkCommandPool = u64;
pub type VkFence = u64;
pub type VkSemaphore = u64;

pub type PFN_vkVoidFunction = Option<unsafe extern "C" fn()>;

#[repr(C)]
pub struct VkExtensionProperties {
    pub extensionName: [c_char; VK_MAX_EXTENSION_NAME_SIZE],
    pub specVersion: u32,
}

#[repr(C)]
pub struct VkLayerProperties {
    pub layerName: [c_char; VK_MAX_EXTENSION_NAME_SIZE],
    pub specVersion: u32,
    pub implementationVersion: u32,
    pub description: [c_char; VK_MAX_DESCRIPTION_SIZE],
}

#[repr(C)]
pub struct VkApplicationInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub pApplicationName: *const c_char,
    pub applicationVersion: u32,
    pub pEngineName: *const c_char,
    pub engineVersion: u32,
    pub apiVersion: u32,
}

#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkInstanceCreateFlags,
    pub pApplicationInfo: *const VkApplicationInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
}

#[repr(C)]
pub struct VkQueueFamilyProperties {
    pub queueFlags: VkQueueFlags,
    pub queueCount: u32,
    pub timestampValidBits: u32,
    pub minImageTransferGranularity: VkExtent3D,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VkExtent2D {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VkOffset2D {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VkRect2D {
    pub offset: VkOffset2D,
    pub extent: VkExtent2D,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VkViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub minDepth: f32,
    pub maxDepth: f32,
}

#[repr(C)]
pub struct VkMemoryType {
    pub propertyFlags: VkMemoryPropertyFlags,
    pub heapIndex: u32,
}

#[repr(C)]
pub struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: VkMemoryHeapFlags,
}

#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memoryTypeCount: u32,
    pub memoryTypes: [VkMemoryType; VK_MAX_MEMORY_TYPES],
    pub memoryHeapCount: u32,
    pub memoryHeaps: [VkMemoryHeap; VK_MAX_MEMORY_HEAPS],
}

#[repr(C)]
pub struct VkPhysicalDeviceLimits {
    pub maxImageDimension1D: u32,
    pub maxImageDimension2D: u32,
    pub maxImageDimension3D: u32,
    pub maxImageDimensionCube: u32,
    pub maxImageArrayLayers: u32,
    pub maxTexelBufferElements: u32,
    pub maxUniformBufferRange: u32,
    pub maxStorageBufferRange: u32,
    pub maxPushConstantsSize: u32,
    pub maxMemoryAllocationCount: u32,
    pub maxSamplerAllocationCount: u32,
    pub bufferImageGranularity: VkDeviceSize,
    pub sparseAddressSpaceSize: VkDeviceSize,
    pub maxBoundDescriptorSets: u32,
    pub maxPerStageDescriptorSamplers: u32,
    pub maxPerStageDescriptorUniformBuffers: u32,
    pub maxPerStageDescriptorStorageBuffers: u32,
    pub maxPerStageDescriptorSampledImages: u32,
    pub maxPerStageDescriptorStorageImages: u32,
    pub maxPerStageDescriptorInputAttachments: u32,
    pub maxPerStageResources: u32,
    pub maxDescriptorSetSamplers: u32,
    pub maxDescriptorSetUniformBuffers: u32,
    pub maxDescriptorSetUniformBuffersDynamic: u32,
    pub maxDescriptorSetStorageBuffers: u32,
    pub maxDescriptorSetStorageBuffersDynamic: u32,
    pub maxDescriptorSetSampledImages: u32,
    pub maxDescriptorSetStorageImages: u32,
    pub maxDescriptorSetInputAttachments: u32,
    pub maxVertexInputAttributes: u32,
    pub maxVertexInputBindings: u32,
    pub maxVertexInputAttributeOffset: u32,
    pub maxVertexInputBindingStride: u32,
    pub maxVertexOutputComponents: u32,
    pub maxTessellationGenerationLevel: u32,
    pub maxTessellationPatchSize: u32,
    pub maxTessellationControlPerVertexInputComponents: u32,
    pub maxTessellationControlPerVertexOutputComponents: u32,
    pub maxTessellationControlPerPatchOutputComponents: u32,
    pub maxTessellationControlTotalOutputComponents: u32,
    pub maxTessellationEvaluationInputComponents: u32,
    pub maxTessellationEvaluationOutputComponents: u32,
    pub maxGeometryShaderInvocations: u32,
    pub maxGeometryInputComponents: u32,
    pub maxGeometryOutputComponents: u32,
    pub maxGeometryOutputVertices: u32,
    pub maxGeometryTotalOutputComponents: u32,
    pub maxFragmentInputComponents: u32,
    pub maxFragmentOutputAttachments: u32,
    pub maxFragmentDualSrcAttachments: u32,
    pub maxFragmentCombinedOutputResources: u32,
    pub maxComputeSharedMemorySize: u32,
    pub maxComputeWorkGroupCount: [u32; 3],
    pub maxComputeWorkGroupInvocations: u32,
    pub maxComputeWorkGroupSize: [u32; 3],
    pub subPixelPrecisionBits: u32,
    pub subTexelPrecisionBits: u32,
    pub mipmapPrecisionBits: u32,
    pub maxDrawIndexedIndexValue: u32,
    pub maxDrawIndirectCount: u32,
    pub maxSamplerLodBias: f32,
    pub maxSamplerAnisotropy: f32,
    pub maxViewports: u32,
    pub maxViewportDimensions: [u32; 2],
    pub viewportBoundsRange: [f32; 2],
    pub viewportSubPixelBits: u32,
    pub minMemoryMapAlignment: usize,
    pub minTexelBufferOffsetAlignment: VkDeviceSize,
    pub minUniformBufferOffsetAlignment: VkDeviceSize,
    pub minStorageBufferOffsetAlignment: VkDeviceSize,
    pub minTexelOffset: i32,
    pub maxTexelOffset: u32,
    pub minTexelGatherOffset: i32,
    pub maxTexelGatherOffset: u32,
    pub minInterpolationOffset: f32,
    pub maxInterpolationOffset: f32,
    pub subPixelInterpolationOffsetBits: u32,
    pub maxFramebufferWidth: u32,
    pub maxFramebufferHeight: u32,
    pub maxFramebufferLayers: u32,
    pub framebufferColorSampleCounts: VkSampleCountFlags,
    pub framebufferDepthSampleCounts: VkSampleCountFlags,
    pub framebufferStencilSampleCounts: VkSampleCountFlags,
    pub framebufferNoAttachmentsSampleCounts: VkSampleCountFlags,
    pub maxColorAttachments: u32,
    pub sampledImageColorSampleCounts: VkSampleCountFlags,
    pub sampledImageIntegerSampleCounts: VkSampleCountFlags,
    pub sampledImageDepthSampleCounts: VkSampleCountFlags,
    pub sampledImageStencilSampleCounts: VkSampleCountFlags,
    pub storageImageSampleCounts: VkSampleCountFlags,
    pub maxSampleMaskWords: u32,
    pub timestampComputeAndGraphics: VkBool32,
    pub timestampPeriod: f32,
    pub maxClipDistances: u32,
    pub maxCullDistances: u32,
    pub maxCombinedClipAndCullDistances: u32,
    pub discreteQueuePriorities: u32,
    pub pointSizeRange: [f32; 2],
    pub lineWidthRange: [f32; 2],
    pub pointSizeGranularity: f32,
    pub lineWidthGranularity: f32,
    pub strictLines: VkBool32,
    pub standardSampleLocations: VkBool32,
    pub optimalBufferCopyOffsetAlignment: VkDeviceSize,
    pub optimalBufferCopyRowPitchAlignment: VkDeviceSize,
    pub nonCoherentAtomSize: VkDeviceSize,
}

#[repr(C)]
pub struct VkPhysicalDeviceProperties {
    pub apiVersion: u32,
    pub driverVersion: u32,
    pub vendorID: u32,
    pub deviceID: u32,
    pub deviceType: VkPhysicalDeviceType,
    pub deviceName: [c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pub pipelineCacheUUID: [u8; VK_UUID_SIZE],
    pub limits: VkPhysicalDeviceLimits,
    pub sparseProperties: [u32; 5],
}

#[repr(C)]
pub struct VkPhysicalDeviceFeatures {
    pub robustBufferAccess: VkBool32,
    pub fullDrawIndexUint32: VkBool32,
    pub imageCubeArray: VkBool32,
    pub independentBlend: VkBool32,
    pub geometryShader: VkBool32,
    pub tessellationShader: VkBool32,
    pub sampleRateShading: VkBool32,
    pub dualSrcBlend: VkBool32,
    pub logicOp: VkBool32,
    pub multiDrawIndirect: VkBool32,
    pub drawIndirectFirstInstance: VkBool32,
    pub depthClamp: VkBool32,
    pub depthBiasClamp: VkBool32,
    pub fillModeNonSolid: VkBool32,
    pub depthBounds: VkBool32,
    pub wideLines: VkBool32,
    pub largePoints: VkBool32,
    pub alphaToOne: VkBool32,
    pub multiViewport: VkBool32,
    pub samplerAnisotropy: VkBool32,
    pub textureCompressionETC2: VkBool32,
    pub textureCompressionASTC_LDR: VkBool32,
    pub textureCompressionBC: VkBool32,
    pub occlusionQueryPrecise: VkBool32,
    pub pipelineStatisticsQuery: VkBool32,
    pub vertexPipelineStoresAndAtomics: VkBool32,
    pub fragmentStoresAndAtomics: VkBool32,
    pub shaderTessellationAndGeometryPointSize: VkBool32,
    pub shaderImageGatherExtended: VkBool32,
    pub shaderStorageImageExtendedFormats: VkBool32,
    pub shaderStorageImageMultisample: VkBool32,
    pub shaderStorageImageReadWithoutFormat: VkBool32,
    pub shaderStorageImageWriteWithoutFormat: VkBool32,
    pub shaderUniformBufferArrayDynamicIndexing: VkBool32,
    pub shaderSampledImageArrayDynamicIndexing: VkBool32,
    pub shaderStorageBufferArrayDynamicIndexing: VkBool32,
    pub shaderStorageImageArrayDynamicIndexing: VkBool32,
    pub shaderClipDistance: VkBool32,
    pub shaderCullDistance: VkBool32,
    pub shaderFloat64: VkBool32,
    pub shaderInt64: VkBool32,
    pub shaderInt16: VkBool32,
    pub shaderResourceResidency: VkBool32,
    pub shaderResourceMinLod: VkBool32,
    pub sparseBinding: VkBool32,
    pub sparseResidencyBuffer: VkBool32,
    pub sparseResidencyImage2D: VkBool32,
    pub sparseResidencyImage3D: VkBool32,
    pub sparseResidency2Samples: VkBool32,
    pub sparseResidency4Samples: VkBool32,
    pub sparseResidency8Samples: VkBool32,
    pub sparseResidency16Samples: VkBool32,
    pub sparseResidencyAliased: VkBool32,
    pub variableMultisampleRate: VkBool32,
    pub inheritedQueries: VkBool32,
}

#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkDeviceQueueCreateFlags,
    pub queueFamilyIndex: u32,
    pub queueCount: u32,
    pub pQueuePriorities: *const f32,
}

#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkDeviceCreateFlags,
    pub queueCreateInfoCount: u32,
    pub pQueueCreateInfos: *const VkDeviceQueueCreateInfo,
    pub enabledLayerCount: u32,
    pub ppEnabledLayerNames: *const *const c_char,
    pub enabledExtensionCount: u32,
    pub ppEnabledExtensionNames: *const *const c_char,
    pub pEnabledFeatures: *const VkPhysicalDeviceFeatures,
}

#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkCommandPoolCreateFlags,
    pub queueFamilyIndex: u32,
}

#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub commandPool: VkCommandPool,
    pub level: VkCommandBufferLevel,
    pub commandBufferCount: u32,
}

#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub pInheritanceInfo: *const c_void,
}

#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub allocationSize: VkDeviceSize,
    pub memoryTypeIndex: u32,
}

#[repr(C)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memoryTypeBits: u32,
}

#[repr(C)]
pub struct VkBufferCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub size: VkDeviceSize,
    pub usage: VkFlags,
    pub sharingMode: u32,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
}

#[repr(C)]
pub struct VkImageCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub imageType: u32,
    pub format: VkFormat,
    pub extent: VkExtent3D,
    pub mipLevels: u32,
    pub arrayLayers: u32,
    pub samples: u32,
    pub tiling: u32,
    pub usage: VkFlags,
    pub sharingMode: u32,
    pub queueFamilyIndexCount: u32,
    pub pQueueFamilyIndices: *const u32,
    pub initialLayout: u32,
}

#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub codeSize: usize,
    pub pCode: *const u32,
}

#[repr(C)]
pub struct VkSubmitInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub waitSemaphoreCount: u32,
    pub pWaitSemaphores: *const VkSemaphore,
    pub pWaitDstStageMask: *const VkFlags,
    pub commandBufferCount: u32,
    pub pCommandBuffers: *const VkCommandBuffer,
    pub signalSemaphoreCount: u32,
    pub pSignalSemaphores: *const VkSemaphore,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union VkClearColorValue {
    pub float32: [f32; 4],
    pub int32: [i32; 4],
    pub uint32: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkClearDepthStencilValue {
    pub depth: f32,
    pub stencil: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union VkClearValue {
    pub color: VkClearColorValue,
    pub depthStencil: VkClearDepthStencilValue,
}

#[repr(C)]
pub struct VkRenderPassBeginInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub renderPass: VkRenderPass,
    pub framebuffer: VkFramebuffer,
    pub renderArea: VkRect2D,
    pub clearValueCount: u32,
    pub pClearValues: *const VkClearValue,
}

#[repr(C)]
pub struct VkRenderPassCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub attachmentCount: u32,
    pub pAttachments: *const c_void,
    pub subpassCount: u32,
    pub pSubpasses: *const c_void,
    pub dependencyCount: u32,
    pub pDependencies: *const c_void,
}

#[repr(C)]
pub struct VkFramebufferCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub renderPass: VkRenderPass,
    pub attachmentCount: u32,
    pub pAttachments: *const VkImageView,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

#[repr(C)]
pub struct VkImageViewCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub image: VkImage,
    pub viewType: u32,
    pub format: VkFormat,
    pub components: [u32; 4],
    pub subresourceRange: [u32; 5],
}

#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub stage: u32,
    pub module: VkShaderModule,
    pub pName: *const c_char,
    pub pSpecializationInfo: *const c_void,
}

#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub setLayoutCount: u32,
    pub pSetLayouts: *const c_void,
    pub pushConstantRangeCount: u32,
    pub pPushConstantRanges: *const c_void,
}

#[repr(C)]
pub struct VkGraphicsPipelineCreateInfo {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub flags: VkFlags,
    pub stageCount: u32,
    pub pStages: *const VkPipelineShaderStageCreateInfo,
    pub pVertexInputState: *const c_void,
    pub pInputAssemblyState: *const c_void,
    pub pTessellationState: *const c_void,
    pub pViewportState: *const c_void,
    pub pRasterizationState: *const c_void,
    pub pMultisampleState: *const c_void,
    pub pDepthStencilState: *const c_void,
    pub pColorBlendState: *const c_void,
    pub pDynamicState: *const c_void,
    pub layout: VkPipelineLayout,
    pub renderPass: VkRenderPass,
    pub subpass: u32,
    pub basePipelineHandle: VkPipeline,
    pub basePipelineIndex: i32,
}
