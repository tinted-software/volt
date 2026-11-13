//! Vantage Vulkan ICD Implementation.
//!
//! Provides the external C ABI entry points and Vulkan Loader (v5) interface
//! backed by `vantage-hal` and `vantage-raster`.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    non_snake_case,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::missing_transmute_annotations
)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod vk;

use alloc::{boxed::Box, vec, vec::Vec};

use core::ffi::{c_char, c_void};
use core::ptr;
use hashbrown::HashMap;
use smallvec::smallvec;
use vk::*;
use volt_hal::{
    BufferId, Cmd, CommandBuffer, Device as HalDevice, Format as HalFormat, ImageId, IndexType,
    Pipeline as HalPipeline, PipelineId,
};
use volt_shader_spirv::SpirvModule;

// ============================================================================
// Dispatchable Handles & Structures
// ============================================================================

/// Loader-dispatchable instance.
#[repr(C)]
pub struct Instance {
    pub loader_magic: usize,
    pub physical_device: PhysicalDevice,
}

/// Loader-dispatchable physical device.
#[repr(C)]
pub struct PhysicalDevice {
    pub loader_magic: usize,
    pub instance: *mut Instance,
}

/// Logical device handle.
#[repr(C)]
pub struct Device {
    pub loader_magic: usize,
    pub hal: HalDevice,
    pub queue: Queue,
    pub memory_allocations: HashMap<VkDeviceMemory, Vec<u8>>,
    pub buffers: HashMap<VkBuffer, BufferBinding>,
    pub images: HashMap<VkImage, ImageBinding>,
    pub image_views: HashMap<VkImageView, VkImage>,
    pub framebuffers: HashMap<VkFramebuffer, FramebufferInfo>,
    pub shader_modules: HashMap<VkShaderModule, SpirvModule>,
    pub pipelines: HashMap<VkPipeline, PipelineId>,
    pub next_handle: u64,
}

pub struct BufferBinding {
    pub hal_id: BufferId,
    pub size: VkDeviceSize,
    pub bound_memory: Option<(VkDeviceMemory, VkDeviceSize)>,
}

pub struct ImageBinding {
    pub hal_id: ImageId,
    pub format: VkFormat,
    pub width: u32,
    pub height: u32,
    pub bound_memory: Option<(VkDeviceMemory, VkDeviceSize)>,
}

pub struct FramebufferInfo {
    pub attachments: Vec<VkImageView>,
    pub width: u32,
    pub height: u32,
}

/// Logical queue handle.
#[repr(C)]
pub struct Queue {
    pub loader_magic: usize,
    pub device: *mut Device,
}

/// Command buffer handle.
#[repr(C)]
pub struct CommandBufferHandle {
    pub loader_magic: usize,
    pub device: *mut Device,
    pub commands: CommandBuffer,
}

// ============================================================================
// Loader Entry Point: vk_icdGetInstanceProcAddr
// ============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vk_icdGetInstanceProcAddr(
    _instance: VkInstance,
    pName: *const c_char,
) -> PFN_vkVoidFunction {
    if pName.is_null() {
        return None;
    }
    let name = core::ffi::CStr::from_ptr(pName).to_bytes();

    match name {
        b"vkGetInstanceProcAddr" => Some(core::mem::transmute(
            vkGetInstanceProcAddr as *const () as usize,
        )),
        b"vk_icdGetInstanceProcAddr" => Some(core::mem::transmute(
            vk_icdGetInstanceProcAddr as *const () as usize,
        )),
        b"vk_icdGetPhysicalDeviceProcAddr" => Some(core::mem::transmute(
            vk_icdGetPhysicalDeviceProcAddr as *const () as usize,
        )),
        b"vk_icdNegotiateLoaderICDInterfaceVersion" => Some(core::mem::transmute(
            vk_icdNegotiateLoaderICDInterfaceVersion as *const () as usize,
        )),
        b"vkCreateInstance" => Some(core::mem::transmute(vkCreateInstance as *const () as usize)),
        b"vkDestroyInstance" => Some(core::mem::transmute(
            vkDestroyInstance as *const () as usize,
        )),
        b"vkEnumeratePhysicalDevices" => Some(core::mem::transmute(
            vkEnumeratePhysicalDevices as *const () as usize,
        )),
        b"vkGetPhysicalDeviceProperties" => Some(core::mem::transmute(
            vkGetPhysicalDeviceProperties as *const () as usize,
        )),
        b"vkGetPhysicalDeviceFeatures" => Some(core::mem::transmute(
            vkGetPhysicalDeviceFeatures as *const () as usize,
        )),
        b"vkGetPhysicalDeviceMemoryProperties" => Some(core::mem::transmute(
            vkGetPhysicalDeviceMemoryProperties as *const () as usize,
        )),
        b"vkGetPhysicalDeviceQueueFamilyProperties" => Some(core::mem::transmute(
            vkGetPhysicalDeviceQueueFamilyProperties as *const () as usize,
        )),
        b"vkCreateDevice" => Some(core::mem::transmute(vkCreateDevice as *const () as usize)),
        b"vkDestroyDevice" => Some(core::mem::transmute(vkDestroyDevice as *const () as usize)),
        b"vkGetDeviceQueue" => Some(core::mem::transmute(vkGetDeviceQueue as *const () as usize)),
        b"vkGetDeviceProcAddr" => Some(core::mem::transmute(
            vkGetDeviceProcAddr as *const () as usize,
        )),
        b"vkAllocateMemory" => Some(core::mem::transmute(vkAllocateMemory as *const () as usize)),
        b"vkFreeMemory" => Some(core::mem::transmute(vkFreeMemory as *const () as usize)),
        b"vkMapMemory" => Some(core::mem::transmute(vkMapMemory as *const () as usize)),
        b"vkUnmapMemory" => Some(core::mem::transmute(vkUnmapMemory as *const () as usize)),
        b"vkCreateBuffer" => Some(core::mem::transmute(vkCreateBuffer as *const () as usize)),
        b"vkDestroyBuffer" => Some(core::mem::transmute(vkDestroyBuffer as *const () as usize)),
        b"vkGetBufferMemoryRequirements" => Some(core::mem::transmute(
            vkGetBufferMemoryRequirements as *const () as usize,
        )),
        b"vkBindBufferMemory" => Some(core::mem::transmute(
            vkBindBufferMemory as *const () as usize,
        )),
        b"vkCreateImage" => Some(core::mem::transmute(vkCreateImage as *const () as usize)),
        b"vkDestroyImage" => Some(core::mem::transmute(vkDestroyImage as *const () as usize)),
        b"vkGetImageMemoryRequirements" => Some(core::mem::transmute(
            vkGetImageMemoryRequirements as *const () as usize,
        )),
        b"vkBindImageMemory" => Some(core::mem::transmute(
            vkBindImageMemory as *const () as usize,
        )),
        b"vkCreateImageView" => Some(core::mem::transmute(
            vkCreateImageView as *const () as usize,
        )),
        b"vkDestroyImageView" => Some(core::mem::transmute(
            vkDestroyImageView as *const () as usize,
        )),
        b"vkCreateShaderModule" => Some(core::mem::transmute(
            vkCreateShaderModule as *const () as usize,
        )),
        b"vkDestroyShaderModule" => Some(core::mem::transmute(
            vkDestroyShaderModule as *const () as usize,
        )),
        b"vkCreateRenderPass" => Some(core::mem::transmute(
            vkCreateRenderPass as *const () as usize,
        )),
        b"vkDestroyRenderPass" => Some(core::mem::transmute(
            vkDestroyRenderPass as *const () as usize,
        )),
        b"vkCreateFramebuffer" => Some(core::mem::transmute(
            vkCreateFramebuffer as *const () as usize,
        )),
        b"vkDestroyFramebuffer" => Some(core::mem::transmute(
            vkDestroyFramebuffer as *const () as usize,
        )),
        b"vkCreatePipelineLayout" => Some(core::mem::transmute(
            vkCreatePipelineLayout as *const () as usize,
        )),
        b"vkDestroyPipelineLayout" => Some(core::mem::transmute(
            vkDestroyPipelineLayout as *const () as usize,
        )),
        b"vkCreateGraphicsPipelines" => Some(core::mem::transmute(
            vkCreateGraphicsPipelines as *const () as usize,
        )),
        b"vkDestroyPipeline" => Some(core::mem::transmute(
            vkDestroyPipeline as *const () as usize,
        )),
        b"vkCreateCommandPool" => Some(core::mem::transmute(
            vkCreateCommandPool as *const () as usize,
        )),
        b"vkDestroyCommandPool" => Some(core::mem::transmute(
            vkDestroyCommandPool as *const () as usize,
        )),
        b"vkAllocateCommandBuffers" => Some(core::mem::transmute(
            vkAllocateCommandBuffers as *const () as usize,
        )),
        b"vkFreeCommandBuffers" => Some(core::mem::transmute(
            vkFreeCommandBuffers as *const () as usize,
        )),
        b"vkBeginCommandBuffer" => Some(core::mem::transmute(
            vkBeginCommandBuffer as *const () as usize,
        )),
        b"vkEndCommandBuffer" => Some(core::mem::transmute(
            vkEndCommandBuffer as *const () as usize,
        )),
        b"vkCmdBeginRenderPass" => Some(core::mem::transmute(
            vkCmdBeginRenderPass as *const () as usize,
        )),
        b"vkCmdEndRenderPass" => Some(core::mem::transmute(
            vkCmdEndRenderPass as *const () as usize,
        )),
        b"vkCmdBindPipeline" => Some(core::mem::transmute(
            vkCmdBindPipeline as *const () as usize,
        )),
        b"vkCmdSetViewport" => Some(core::mem::transmute(vkCmdSetViewport as *const () as usize)),
        b"vkCmdSetScissor" => Some(core::mem::transmute(vkCmdSetScissor as *const () as usize)),
        b"vkCmdBindVertexBuffers" => Some(core::mem::transmute(
            vkCmdBindVertexBuffers as *const () as usize,
        )),
        b"vkCmdBindIndexBuffer" => Some(core::mem::transmute(
            vkCmdBindIndexBuffer as *const () as usize,
        )),
        b"vkCmdDraw" => Some(core::mem::transmute(vkCmdDraw as *const () as usize)),
        b"vkCmdDrawIndexed" => Some(core::mem::transmute(vkCmdDrawIndexed as *const () as usize)),
        b"vkQueueSubmit" => Some(core::mem::transmute(vkQueueSubmit as *const () as usize)),
        b"vkQueueWaitIdle" => Some(core::mem::transmute(vkQueueWaitIdle as *const () as usize)),
        b"vkDeviceWaitIdle" => Some(core::mem::transmute(vkDeviceWaitIdle as *const () as usize)),
        _ => None,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetInstanceProcAddr(
    instance: VkInstance,
    pName: *const c_char,
) -> PFN_vkVoidFunction {
    vk_icdGetInstanceProcAddr(instance, pName)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetDeviceProcAddr(
    _device: VkDevice,
    pName: *const c_char,
) -> PFN_vkVoidFunction {
    vk_icdGetInstanceProcAddr(ptr::null_mut(), pName)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vk_icdGetPhysicalDeviceProcAddr(
    _instance: VkInstance,
    pName: *const c_char,
) -> PFN_vkVoidFunction {
    vk_icdGetInstanceProcAddr(ptr::null_mut(), pName)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vk_icdNegotiateLoaderICDInterfaceVersion(
    pSupportedVersion: *mut u32,
) -> VkResult {
    if pSupportedVersion.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let requested = *pSupportedVersion;
    *pSupportedVersion = requested.min(5);
    VkResult::VK_SUCCESS
}

// ============================================================================
// Core Vulkan Entry Points
// ============================================================================

/// Builds a new instance. Safe to call; ownership stays with the caller.
fn create_instance_inner() -> Box<Instance> {
    let mut inst = Box::new(Instance {
        loader_magic: ICD_LOADER_MAGIC,
        physical_device: PhysicalDevice {
            loader_magic: ICD_LOADER_MAGIC,
            instance: ptr::null_mut(),
        },
    });
    inst.physical_device.instance = &mut *inst as *mut Instance;
    inst
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateInstance(
    _pCreateInfo: *const VkInstanceCreateInfo,
    _pAllocator: *const c_void,
    pInstance: *mut VkInstance,
) -> VkResult {
    if pInstance.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    *pInstance = Box::into_raw(create_instance_inner()) as VkInstance;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyInstance(instance: VkInstance, _pAllocator: *const c_void) {
    if !instance.is_null() {
        drop(Box::from_raw(instance as *mut Instance));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkEnumeratePhysicalDevices(
    instance: VkInstance,
    pPhysicalDeviceCount: *mut u32,
    pPhysicalDevices: *mut VkPhysicalDevice,
) -> VkResult {
    if pPhysicalDeviceCount.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let count = 1;
    if pPhysicalDevices.is_null() {
        *pPhysicalDeviceCount = count;
        return VkResult::VK_SUCCESS;
    }

    if *pPhysicalDeviceCount < count {
        *pPhysicalDeviceCount = 0;
        return VkResult::VK_INCOMPLETE;
    }

    let inst = &mut *(instance as *mut Instance);
    *pPhysicalDevices = &mut inst.physical_device as *mut PhysicalDevice as VkPhysicalDevice;
    *pPhysicalDeviceCount = count;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetPhysicalDeviceProperties(
    _physicalDevice: VkPhysicalDevice,
    pProperties: *mut VkPhysicalDeviceProperties,
) {
    if pProperties.is_null() {
        return;
    }

    let props = &mut *pProperties;
    props.apiVersion = VK_API_VERSION_1_0;
    props.driverVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    props.vendorID = VANTAGE_VENDOR_ID;
    props.deviceID = VANTAGE_DEVICE_ID;
    props.deviceType = VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU;

    let name = b"Vantage Software Rasterizer\0";
    for (i, &b) in name.iter().enumerate() {
        props.deviceName[i] = b as c_char;
    }

    props.limits.maxImageDimension2D = 4096;
    props.limits.maxImageDimension1D = 4096;
    props.limits.maxImageDimension3D = 512;
    props.limits.maxBoundDescriptorSets = 4;
    props.limits.maxVertexInputBindings = 8;
    props.limits.maxVertexInputAttributes = 16;
    props.limits.maxColorAttachments = 1;
    props.limits.maxFramebufferWidth = 4096;
    props.limits.maxFramebufferHeight = 4096;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetPhysicalDeviceFeatures(
    _physicalDevice: VkPhysicalDevice,
    pFeatures: *mut VkPhysicalDeviceFeatures,
) {
    if !pFeatures.is_null() {
        core::ptr::write_bytes(pFeatures, 0, 1);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetPhysicalDeviceQueueFamilyProperties(
    _physicalDevice: VkPhysicalDevice,
    pQueueFamilyPropertyCount: *mut u32,
    pQueueFamilyProperties: *mut VkQueueFamilyProperties,
) {
    if pQueueFamilyPropertyCount.is_null() {
        return;
    }

    if pQueueFamilyProperties.is_null() {
        *pQueueFamilyPropertyCount = 1;
        return;
    }

    if *pQueueFamilyPropertyCount >= 1 {
        let qf = &mut *pQueueFamilyProperties;
        qf.queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        qf.queueCount = 1;
        qf.timestampValidBits = 0;
        qf.minImageTransferGranularity = VkExtent3D {
            width: 1,
            height: 1,
            depth: 1,
        };
        *pQueueFamilyPropertyCount = 1;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetPhysicalDeviceMemoryProperties(
    _physicalDevice: VkPhysicalDevice,
    pMemoryProperties: *mut VkPhysicalDeviceMemoryProperties,
) {
    if pMemoryProperties.is_null() {
        return;
    }

    let mem = &mut *pMemoryProperties;
    mem.memoryTypeCount = 1;
    mem.memoryTypes[0] = VkMemoryType {
        propertyFlags: VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        heapIndex: 0,
    };
    mem.memoryHeapCount = 1;
    mem.memoryHeaps[0] = VkMemoryHeap {
        size: 1024 * 1024 * 1024,
        flags: VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
    };
}

/// Builds a new device. Safe to call; ownership stays with the caller.
fn create_device_inner() -> Box<Device> {
    let mut dev = Box::new(Device {
        loader_magic: ICD_LOADER_MAGIC,
        hal: HalDevice::new(),
        queue: Queue {
            loader_magic: ICD_LOADER_MAGIC,
            device: ptr::null_mut(),
        },
        memory_allocations: HashMap::new(),
        buffers: HashMap::new(),
        images: HashMap::new(),
        image_views: HashMap::new(),
        framebuffers: HashMap::new(),
        shader_modules: HashMap::new(),
        pipelines: HashMap::new(),
        next_handle: 1,
    });
    dev.queue.device = &mut *dev as *mut Device;
    dev
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateDevice(
    _physicalDevice: VkPhysicalDevice,
    _pCreateInfo: *const VkDeviceCreateInfo,
    _pAllocator: *const c_void,
    pDevice: *mut VkDevice,
) -> VkResult {
    if pDevice.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    *pDevice = Box::into_raw(create_device_inner()) as VkDevice;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyDevice(device: VkDevice, _pAllocator: *const c_void) {
    if !device.is_null() {
        drop(Box::from_raw(device as *mut Device));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetDeviceQueue(
    device: VkDevice,
    _queueFamilyIndex: u32,
    _queueIndex: u32,
    pQueue: *mut VkQueue,
) {
    if !device.is_null() && !pQueue.is_null() {
        let dev = &mut *(device as *mut Device);
        *pQueue = &mut dev.queue as *mut Queue as VkQueue;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkAllocateMemory(
    device: VkDevice,
    pAllocateInfo: *const VkMemoryAllocateInfo,
    _pAllocator: *const c_void,
    pMemory: *mut VkDeviceMemory,
) -> VkResult {
    if device.is_null() || pAllocateInfo.is_null() || pMemory.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let dev = &mut *(device as *mut Device);
    let info = &*pAllocateInfo;
    let size = info.allocationSize as usize;

    let id = dev.next_handle;
    dev.next_handle += 1;

    let buf = vec![0u8; size];
    dev.memory_allocations.insert(id, buf);

    *pMemory = id;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkFreeMemory(
    device: VkDevice,
    memory: VkDeviceMemory,
    _pAllocator: *const c_void,
) {
    if !device.is_null() {
        let dev = &mut *(device as *mut Device);
        dev.memory_allocations.remove(&memory);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkMapMemory(
    device: VkDevice,
    memory: VkDeviceMemory,
    offset: VkDeviceSize,
    _size: VkDeviceSize,
    _flags: VkFlags,
    ppData: *mut *mut c_void,
) -> VkResult {
    if device.is_null() || ppData.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let dev = &mut *(device as *mut Device);
    if let Some(mem) = dev.memory_allocations.get_mut(&memory) {
        let ptr = mem.as_mut_ptr().add(offset as usize);
        *ppData = ptr as *mut c_void;
        VkResult::VK_SUCCESS
    } else {
        VkResult::VK_ERROR_MEMORY_MAP_FAILED
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkUnmapMemory(_device: VkDevice, _memory: VkDeviceMemory) {}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateBuffer(
    device: VkDevice,
    pCreateInfo: *const VkBufferCreateInfo,
    _pAllocator: *const c_void,
    pBuffer: *mut VkBuffer,
) -> VkResult {
    if device.is_null() || pCreateInfo.is_null() || pBuffer.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let dev = &mut *(device as *mut Device);
    let info = &*pCreateInfo;

    let hal_id = dev.hal.create_buffer(info.size);
    let handle = dev.next_handle;
    dev.next_handle += 1;

    dev.buffers.insert(
        handle,
        BufferBinding {
            hal_id,
            size: info.size,
            bound_memory: None,
        },
    );

    *pBuffer = handle;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyBuffer(
    device: VkDevice,
    buffer: VkBuffer,
    _pAllocator: *const c_void,
) {
    if !device.is_null() {
        let dev = &mut *(device as *mut Device);
        if let Some(binding) = dev.buffers.remove(&buffer) {
            dev.hal.destroy_buffer(binding.hal_id);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetBufferMemoryRequirements(
    device: VkDevice,
    buffer: VkBuffer,
    pMemoryRequirements: *mut VkMemoryRequirements,
) {
    if device.is_null() || pMemoryRequirements.is_null() {
        return;
    }

    let dev = &*(device as *const Device);
    let req = &mut *pMemoryRequirements;
    if let Some(b) = dev.buffers.get(&buffer) {
        req.size = b.size;
        req.alignment = 16;
        req.memoryTypeBits = 1;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkBindBufferMemory(
    device: VkDevice,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    memoryOffset: VkDeviceSize,
) -> VkResult {
    if device.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let dev = &mut *(device as *mut Device);
    if let Some(b) = dev.buffers.get_mut(&buffer) {
        b.bound_memory = Some((memory, memoryOffset));
        VkResult::VK_SUCCESS
    } else {
        VkResult::VK_ERROR_INITIALIZATION_FAILED
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateImage(
    device: VkDevice,
    pCreateInfo: *const VkImageCreateInfo,
    _pAllocator: *const c_void,
    pImage: *mut VkImage,
) -> VkResult {
    if device.is_null() || pCreateInfo.is_null() || pImage.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let dev = &mut *(device as *mut Device);
    let info = &*pCreateInfo;

    let hal_fmt = match info.format {
        VkFormat::VK_FORMAT_B8G8R8A8_UNORM => HalFormat::B8G8R8A8Unorm,
        VkFormat::VK_FORMAT_D32_SFLOAT => HalFormat::D32Sfloat,
        VkFormat::VK_FORMAT_S8_UINT => HalFormat::S8Uint,
        _ => HalFormat::R8G8B8A8Unorm,
    };

    let hal_id = dev
        .hal
        .create_image(hal_fmt, info.extent.width, info.extent.height);
    let handle = dev.next_handle;
    dev.next_handle += 1;

    dev.images.insert(
        handle,
        ImageBinding {
            hal_id,
            format: info.format,
            width: info.extent.width,
            height: info.extent.height,
            bound_memory: None,
        },
    );

    *pImage = handle;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyImage(
    device: VkDevice,
    image: VkImage,
    _pAllocator: *const c_void,
) {
    if !device.is_null() {
        let dev = &mut *(device as *mut Device);
        if let Some(binding) = dev.images.remove(&image) {
            dev.hal.destroy_image(binding.hal_id);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkGetImageMemoryRequirements(
    device: VkDevice,
    image: VkImage,
    pMemoryRequirements: *mut VkMemoryRequirements,
) {
    if device.is_null() || pMemoryRequirements.is_null() {
        return;
    }

    let dev = &*(device as *const Device);
    let req = &mut *pMemoryRequirements;
    if let Some(im) = dev.images.get(&image) {
        let bpp = match im.format {
            VkFormat::VK_FORMAT_D32_SFLOAT => 4,
            VkFormat::VK_FORMAT_S8_UINT => 1,
            _ => 4,
        };
        req.size = (im.width * im.height * bpp) as VkDeviceSize;
        req.alignment = 64;
        req.memoryTypeBits = 1;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkBindImageMemory(
    device: VkDevice,
    image: VkImage,
    memory: VkDeviceMemory,
    memoryOffset: VkDeviceSize,
) -> VkResult {
    if device.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let dev = &mut *(device as *mut Device);
    if let Some(im) = dev.images.get_mut(&image) {
        im.bound_memory = Some((memory, memoryOffset));
        VkResult::VK_SUCCESS
    } else {
        VkResult::VK_ERROR_INITIALIZATION_FAILED
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateImageView(
    device: VkDevice,
    pCreateInfo: *const VkImageViewCreateInfo,
    _pAllocator: *const c_void,
    pView: *mut VkImageView,
) -> VkResult {
    if device.is_null() || pCreateInfo.is_null() || pView.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let dev = &mut *(device as *mut Device);
    let info = &*pCreateInfo;
    let handle = dev.next_handle;
    dev.next_handle += 1;
    dev.image_views.insert(handle, info.image);
    *pView = handle;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyImageView(
    device: VkDevice,
    imageView: VkImageView,
    _pAllocator: *const c_void,
) {
    if !device.is_null() {
        let dev = &mut *(device as *mut Device);
        dev.image_views.remove(&imageView);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateShaderModule(
    device: VkDevice,
    pCreateInfo: *const VkShaderModuleCreateInfo,
    _pAllocator: *const c_void,
    pShaderModule: *mut VkShaderModule,
) -> VkResult {
    if device.is_null() || pCreateInfo.is_null() || pShaderModule.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let dev = &mut *(device as *mut Device);
    let info = &*pCreateInfo;
    if info.codeSize == 0 || !info.codeSize.is_multiple_of(4) || info.pCode.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let byte_slice = core::slice::from_raw_parts(info.pCode as *const u8, info.codeSize);
    let module = match SpirvModule::from_bytes(byte_slice) {
        Ok(m) => m,
        Err(_) => return VkResult::VK_ERROR_INITIALIZATION_FAILED,
    };

    let handle = dev.next_handle;
    dev.next_handle += 1;
    dev.shader_modules.insert(handle, module);
    *pShaderModule = handle;

    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyShaderModule(
    device: VkDevice,
    shaderModule: VkShaderModule,
    _pAllocator: *const c_void,
) {
    if !device.is_null() {
        let dev = &mut *(device as *mut Device);
        dev.shader_modules.remove(&shaderModule);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateRenderPass(
    device: VkDevice,
    _pCreateInfo: *const VkRenderPassCreateInfo,
    _pAllocator: *const c_void,
    pRenderPass: *mut VkRenderPass,
) -> VkResult {
    if device.is_null() || pRenderPass.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let dev = &mut *(device as *mut Device);
    let handle = dev.next_handle;
    dev.next_handle += 1;
    *pRenderPass = handle;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyRenderPass(
    _device: VkDevice,
    _renderPass: VkRenderPass,
    _pAllocator: *const c_void,
) {
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateFramebuffer(
    device: VkDevice,
    pCreateInfo: *const VkFramebufferCreateInfo,
    _pAllocator: *const c_void,
    pFramebuffer: *mut VkFramebuffer,
) -> VkResult {
    if device.is_null() || pCreateInfo.is_null() || pFramebuffer.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let dev = &mut *(device as *mut Device);
    let info = &*pCreateInfo;
    let mut attachments = Vec::new();
    if info.attachmentCount > 0 && !info.pAttachments.is_null() {
        attachments.extend_from_slice(core::slice::from_raw_parts(
            info.pAttachments,
            info.attachmentCount as usize,
        ));
    }

    let handle = dev.next_handle;
    dev.next_handle += 1;
    dev.framebuffers.insert(
        handle,
        FramebufferInfo {
            attachments,
            width: info.width,
            height: info.height,
        },
    );
    *pFramebuffer = handle;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyFramebuffer(
    device: VkDevice,
    framebuffer: VkFramebuffer,
    _pAllocator: *const c_void,
) {
    if !device.is_null() {
        let dev = &mut *(device as *mut Device);
        dev.framebuffers.remove(&framebuffer);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreatePipelineLayout(
    device: VkDevice,
    _pCreateInfo: *const VkPipelineLayoutCreateInfo,
    _pAllocator: *const c_void,
    pPipelineLayout: *mut VkPipelineLayout,
) -> VkResult {
    if device.is_null() || pPipelineLayout.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let dev = &mut *(device as *mut Device);
    let handle = dev.next_handle;
    dev.next_handle += 1;
    *pPipelineLayout = handle;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyPipelineLayout(
    _device: VkDevice,
    _pipelineLayout: VkPipelineLayout,
    _pAllocator: *const c_void,
) {
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateGraphicsPipelines(
    device: VkDevice,
    _pipelineCache: u64,
    createInfoCount: u32,
    pCreateInfos: *const VkGraphicsPipelineCreateInfo,
    _pAllocator: *const c_void,
    pPipelines: *mut VkPipeline,
) -> VkResult {
    if device.is_null() || pCreateInfos.is_null() || pPipelines.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let dev = &mut *(device as *mut Device);

    for i in 0..createInfoCount as usize {
        let hal_pipe = HalPipeline {
            topology: 0,
            cull_mode: 0,
            front_face_ccw: true,
            blend_enabled: false,
            src_factor: volt_raster::gl::ONE,
            dst_factor: volt_raster::gl::ZERO,
            depth_test: false,
            depth_write: true,
            depth_func: volt_raster::gl::LESS,
            color_mask: 0x0f,
        };

        let hal_id = dev.hal.create_pipeline(hal_pipe);
        let handle = dev.next_handle;
        dev.next_handle += 1;
        dev.pipelines.insert(handle, hal_id);
        *pPipelines.add(i) = handle;
    }

    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyPipeline(
    device: VkDevice,
    pipeline: VkPipeline,
    _pAllocator: *const c_void,
) {
    if !device.is_null() {
        let dev = &mut *(device as *mut Device);
        dev.pipelines.remove(&pipeline);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCreateCommandPool(
    device: VkDevice,
    _pCreateInfo: *const VkCommandPoolCreateInfo,
    _pAllocator: *const c_void,
    pCommandPool: *mut VkCommandPool,
) -> VkResult {
    if device.is_null() || pCommandPool.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let dev = &mut *(device as *mut Device);
    let handle = dev.next_handle;
    dev.next_handle += 1;
    *pCommandPool = handle;
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDestroyCommandPool(
    _device: VkDevice,
    _commandPool: VkCommandPool,
    _pAllocator: *const c_void,
) {
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkAllocateCommandBuffers(
    device: VkDevice,
    pAllocateInfo: *const VkCommandBufferAllocateInfo,
    pCommandBuffers: *mut VkCommandBuffer,
) -> VkResult {
    if device.is_null() || pAllocateInfo.is_null() || pCommandBuffers.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }

    let info = &*pAllocateInfo;
    for i in 0..info.commandBufferCount as usize {
        let cb = Box::new(CommandBufferHandle {
            loader_magic: ICD_LOADER_MAGIC,
            device: device as *mut Device,
            commands: CommandBuffer::default(),
        });
        *pCommandBuffers.add(i) = Box::into_raw(cb) as VkCommandBuffer;
    }

    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkFreeCommandBuffers(
    _device: VkDevice,
    _commandPool: VkCommandPool,
    commandBufferCount: u32,
    pCommandBuffers: *const VkCommandBuffer,
) {
    if pCommandBuffers.is_null() {
        return;
    }
    for i in 0..commandBufferCount as usize {
        let ptr = *pCommandBuffers.add(i);
        if !ptr.is_null() {
            drop(Box::from_raw(ptr as *mut CommandBufferHandle));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkBeginCommandBuffer(
    commandBuffer: VkCommandBuffer,
    _pBeginInfo: *const VkCommandBufferBeginInfo,
) -> VkResult {
    if commandBuffer.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    cb.commands.ops.clear();
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkEndCommandBuffer(_commandBuffer: VkCommandBuffer) -> VkResult {
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdBeginRenderPass(
    commandBuffer: VkCommandBuffer,
    pRenderPassBegin: *const VkRenderPassBeginInfo,
    _contents: VkSubpassContents,
) {
    if commandBuffer.is_null() || pRenderPassBegin.is_null() {
        return;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    let dev = &*(cb.device as *const Device);
    let info = &*pRenderPassBegin;

    let mut color_id = None;
    if let Some(fb) = dev.framebuffers.get(&info.framebuffer)
        && let Some(&first_view) = fb.attachments.first()
        && let Some(&img_handle) = dev.image_views.get(&first_view)
        && let Some(img_binding) = dev.images.get(&img_handle)
    {
        color_id = Some(img_binding.hal_id);
    }

    cb.commands.ops.push(Cmd::BindAttachments {
        color: color_id,
        depth: None,
        stencil: None,
    });

    if info.clearValueCount > 0 && !info.pClearValues.is_null() {
        let cv = &*info.pClearValues;
        let c = cv.color.float32;
        cb.commands.ops.push(Cmd::ClearAttachments {
            color: c,
            depth: 1.0,
            stencil: 0,
            mask: 1, // Color mask
        });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdEndRenderPass(_commandBuffer: VkCommandBuffer) {}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdBindPipeline(
    commandBuffer: VkCommandBuffer,
    _pipelineBindPoint: VkPipelineBindPoint,
    pipeline: VkPipeline,
) {
    if commandBuffer.is_null() {
        return;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    let dev = &*(cb.device as *const Device);
    if let Some(&hal_id) = dev.pipelines.get(&pipeline) {
        cb.commands.ops.push(Cmd::BindPipeline { pipeline: hal_id });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdSetViewport(
    commandBuffer: VkCommandBuffer,
    _firstViewport: u32,
    viewportCount: u32,
    pViewports: *const VkViewport,
) {
    if commandBuffer.is_null() || pViewports.is_null() || viewportCount == 0 {
        return;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    let vp = &*pViewports;
    cb.commands.ops.push(Cmd::SetViewport {
        x: vp.x as i32,
        y: vp.y as i32,
        w: vp.width as u32,
        h: vp.height as u32,
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdSetScissor(
    commandBuffer: VkCommandBuffer,
    _firstScissor: u32,
    scissorCount: u32,
    pScissors: *const VkRect2D,
) {
    if commandBuffer.is_null() || pScissors.is_null() || scissorCount == 0 {
        return;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    let sc = &*pScissors;
    cb.commands.ops.push(Cmd::SetScissor(Some((
        sc.offset.x,
        sc.offset.y,
        sc.extent.width,
        sc.extent.height,
    ))));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdBindVertexBuffers(
    commandBuffer: VkCommandBuffer,
    firstBinding: u32,
    bindingCount: u32,
    pBuffers: *const VkBuffer,
    pOffsets: *const VkDeviceSize,
) {
    if commandBuffer.is_null() || pBuffers.is_null() || pOffsets.is_null() {
        return;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    let dev = &*(cb.device as *const Device);

    let mut buffers = smallvec![];
    for i in 0..bindingCount as usize {
        let buf_handle = *pBuffers.add(i);
        let offset = *pOffsets.add(i);
        if let Some(binding) = dev.buffers.get(&buf_handle) {
            buffers.push((binding.hal_id, offset));
        }
    }

    cb.commands.ops.push(Cmd::BindVertexBuffers {
        first: firstBinding,
        buffers,
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdBindIndexBuffer(
    commandBuffer: VkCommandBuffer,
    buffer: VkBuffer,
    offset: VkDeviceSize,
    indexType: VkIndexType,
) {
    if commandBuffer.is_null() {
        return;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    let dev = &*(cb.device as *const Device);

    if let Some(binding) = dev.buffers.get(&buffer) {
        let ty = match indexType {
            VkIndexType::VK_INDEX_TYPE_UINT16 => IndexType::U16,
            VkIndexType::VK_INDEX_TYPE_UINT32 => IndexType::U32,
        };
        cb.commands.ops.push(Cmd::BindIndexBuffer {
            buffer: binding.hal_id,
            offset,
            index_ty: ty,
        });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdDraw(
    commandBuffer: VkCommandBuffer,
    vertexCount: u32,
    _instanceCount: u32,
    firstVertex: u32,
    _firstInstance: u32,
) {
    if commandBuffer.is_null() {
        return;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    cb.commands.ops.push(Cmd::Draw {
        count: vertexCount,
        first: firstVertex,
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkCmdDrawIndexed(
    commandBuffer: VkCommandBuffer,
    indexCount: u32,
    _instanceCount: u32,
    firstIndex: u32,
    _vertexOffset: i32,
    _firstInstance: u32,
) {
    if commandBuffer.is_null() {
        return;
    }
    let cb = &mut *(commandBuffer as *mut CommandBufferHandle);
    cb.commands.ops.push(Cmd::DrawIndexed {
        count: indexCount,
        first: firstIndex,
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkQueueSubmit(
    queue: VkQueue,
    submitCount: u32,
    pSubmits: *const VkSubmitInfo,
    _fence: VkFence,
) -> VkResult {
    if queue.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    let q = &*(queue as *const Queue);
    let dev = &mut *q.device;

    if submitCount > 0 && !pSubmits.is_null() {
        for i in 0..submitCount as usize {
            let submit = &*pSubmits.add(i);
            for j in 0..submit.commandBufferCount as usize {
                let cb_ptr = *submit.pCommandBuffers.add(j);
                if !cb_ptr.is_null() {
                    let cb = &*(cb_ptr as *const CommandBufferHandle);
                    dev.hal.submit(&cb.commands);
                }
            }
        }
    }

    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkQueueWaitIdle(queue: VkQueue) -> VkResult {
    if queue.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    VkResult::VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkDeviceWaitIdle(device: VkDevice) -> VkResult {
    if device.is_null() {
        return VkResult::VK_ERROR_INITIALIZATION_FAILED;
    }
    VkResult::VK_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negotiate_interface_version() {
        let mut ver = 5;
        unsafe {
            assert_eq!(
                vk_icdNegotiateLoaderICDInterfaceVersion(&mut ver),
                VkResult::VK_SUCCESS
            );
            assert_eq!(ver, 5);
        }
    }

    #[test]
    fn test_create_instance_device_flow() {
        unsafe {
            let mut instance = ptr::null_mut();
            let res = vkCreateInstance(ptr::null(), ptr::null(), &mut instance);
            assert_eq!(res, VkResult::VK_SUCCESS);
            assert!(!instance.is_null());

            let mut count = 0;
            let res = vkEnumeratePhysicalDevices(instance, &mut count, ptr::null_mut());
            assert_eq!(res, VkResult::VK_SUCCESS);
            assert_eq!(count, 1);

            let mut phys = ptr::null_mut();
            let res = vkEnumeratePhysicalDevices(instance, &mut count, &mut phys);
            assert_eq!(res, VkResult::VK_SUCCESS);
            assert!(!phys.is_null());

            let mut props: VkPhysicalDeviceProperties = core::mem::zeroed();
            vkGetPhysicalDeviceProperties(phys, &mut props);
            assert_eq!(props.vendorID, VANTAGE_VENDOR_ID);
            assert_eq!(props.deviceID, VANTAGE_DEVICE_ID);

            let mut device = ptr::null_mut();
            let res = vkCreateDevice(phys, ptr::null(), ptr::null(), &mut device);
            assert_eq!(res, VkResult::VK_SUCCESS);
            assert!(!device.is_null());

            let mut queue = ptr::null_mut();
            vkGetDeviceQueue(device, 0, 0, &mut queue);
            assert!(!queue.is_null());

            // Allocate buffer
            let mut buffer = 0;
            let b_info = VkBufferCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                size: 256,
                usage: 0,
                sharingMode: 0,
                queueFamilyIndexCount: 0,
                pQueueFamilyIndices: ptr::null(),
            };
            assert_eq!(
                vkCreateBuffer(device, &b_info, ptr::null(), &mut buffer),
                VkResult::VK_SUCCESS
            );
            assert_ne!(buffer, 0);

            vkDestroyBuffer(device, buffer, ptr::null());
            vkDestroyDevice(device, ptr::null());
            vkDestroyInstance(instance, ptr::null());
        }
    }

    #[test]
    fn test_shader_module_and_draw_pipeline() {
        unsafe {
            let mut instance = ptr::null_mut();
            assert_eq!(
                vkCreateInstance(ptr::null(), ptr::null(), &mut instance),
                VkResult::VK_SUCCESS
            );
            let mut count = 1;
            let mut phys = ptr::null_mut();
            assert_eq!(
                vkEnumeratePhysicalDevices(instance, &mut count, &mut phys),
                VkResult::VK_SUCCESS
            );
            let mut device = ptr::null_mut();
            assert_eq!(
                vkCreateDevice(phys, ptr::null(), ptr::null(), &mut device),
                VkResult::VK_SUCCESS
            );

            // Load SPIR-V into shader module
            let spirv_bytes = include_bytes!("../../shader-spirv/testdata/testdata_flip_frag.spv");
            let sm_info = VkShaderModuleCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                codeSize: spirv_bytes.len(),
                pCode: spirv_bytes.as_ptr() as *const u32,
            };
            let mut shader_module = 0;
            assert_eq!(
                vkCreateShaderModule(device, &sm_info, ptr::null(), &mut shader_module),
                VkResult::VK_SUCCESS
            );
            assert_ne!(shader_module, 0);

            // Create Pipeline
            let mut pipeline = 0;
            let p_info = VkGraphicsPipelineCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stageCount: 0,
                pStages: ptr::null(),
                pVertexInputState: ptr::null(),
                pInputAssemblyState: ptr::null(),
                pTessellationState: ptr::null(),
                pViewportState: ptr::null(),
                pRasterizationState: ptr::null(),
                pMultisampleState: ptr::null(),
                pDepthStencilState: ptr::null(),
                pColorBlendState: ptr::null(),
                pDynamicState: ptr::null(),
                layout: 0,
                renderPass: 0,
                subpass: 0,
                basePipelineHandle: 0,
                basePipelineIndex: -1,
            };
            assert_eq!(
                vkCreateGraphicsPipelines(device, 0, 1, &p_info, ptr::null(), &mut pipeline),
                VkResult::VK_SUCCESS
            );
            assert_ne!(pipeline, 0);

            // Command buffer recording & execution
            let mut pool = 0;
            let cp_info = VkCommandPoolCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                queueFamilyIndex: 0,
            };
            assert_eq!(
                vkCreateCommandPool(device, &cp_info, ptr::null(), &mut pool),
                VkResult::VK_SUCCESS
            );

            let mut cmd_buf = ptr::null_mut();
            let cb_info = VkCommandBufferAllocateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: ptr::null(),
                commandPool: pool,
                level: VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount: 1,
            };
            assert_eq!(
                vkAllocateCommandBuffers(device, &cb_info, &mut cmd_buf),
                VkResult::VK_SUCCESS
            );

            assert_eq!(
                vkBeginCommandBuffer(cmd_buf, ptr::null()),
                VkResult::VK_SUCCESS
            );
            vkCmdBindPipeline(
                cmd_buf,
                VkPipelineBindPoint::VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline,
            );
            vkCmdDraw(cmd_buf, 3, 1, 0, 0);
            assert_eq!(vkEndCommandBuffer(cmd_buf), VkResult::VK_SUCCESS);

            let mut queue = ptr::null_mut();
            vkGetDeviceQueue(device, 0, 0, &mut queue);

            let submit = VkSubmitInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: 0,
                pWaitSemaphores: ptr::null(),
                pWaitDstStageMask: ptr::null(),
                commandBufferCount: 1,
                pCommandBuffers: &cmd_buf,
                signalSemaphoreCount: 0,
                pSignalSemaphores: ptr::null(),
            };
            assert_eq!(vkQueueSubmit(queue, 1, &submit, 0), VkResult::VK_SUCCESS);

            vkDestroyShaderModule(device, shader_module, ptr::null());
            vkDestroyPipeline(device, pipeline, ptr::null());
            vkDestroyCommandPool(device, pool, ptr::null());
            vkDestroyDevice(device, ptr::null());
            vkDestroyInstance(instance, ptr::null());
        }
    }
    #[test]
    fn test_vulkan_e2e_clear_and_render() {
        unsafe {
            let mut instance_box = create_instance_inner();
            let instance = &mut *instance_box as *mut Instance as VkInstance;
            let mut count = 1;
            let mut phys = ptr::null_mut();
            assert_eq!(
                vkEnumeratePhysicalDevices(instance, &mut count, &mut phys),
                VkResult::VK_SUCCESS
            );
            // The test owns the device; the raw handle is only passed to the C entry points.
            let mut device_box = create_device_inner();
            let device = &mut *device_box as *mut Device as VkDevice;

            // Create color image (64x64 RGBA)
            let mut image = 0;
            let img_info = VkImageCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                imageType: 2,
                format: VkFormat::VK_FORMAT_R8G8B8A8_UNORM,
                extent: VkExtent3D {
                    width: 64,
                    height: 64,
                    depth: 1,
                },
                mipLevels: 1,
                arrayLayers: 1,
                samples: 1,
                tiling: 0,
                usage: 0,
                sharingMode: 0,
                queueFamilyIndexCount: 0,
                pQueueFamilyIndices: ptr::null(),
                initialLayout: 0,
            };
            assert_eq!(
                vkCreateImage(device, &img_info, ptr::null(), &mut image),
                VkResult::VK_SUCCESS
            );

            // Create image view
            let mut view = 0;
            let view_info = VkImageViewCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                image,
                viewType: 1,
                format: VkFormat::VK_FORMAT_R8G8B8A8_UNORM,
                components: [0; 4],
                subresourceRange: [0; 5],
            };
            assert_eq!(
                vkCreateImageView(device, &view_info, ptr::null(), &mut view),
                VkResult::VK_SUCCESS
            );

            // Create renderpass and framebuffer
            let mut render_pass = 0;
            assert_eq!(
                vkCreateRenderPass(device, ptr::null(), ptr::null(), &mut render_pass),
                VkResult::VK_SUCCESS
            );

            let mut framebuffer = 0;
            let fb_info = VkFramebufferCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                renderPass: render_pass,
                attachmentCount: 1,
                pAttachments: &view,
                width: 64,
                height: 64,
                layers: 1,
            };
            assert_eq!(
                vkCreateFramebuffer(device, &fb_info, ptr::null(), &mut framebuffer),
                VkResult::VK_SUCCESS
            );

            // Command pool and buffer
            let mut pool = 0;
            let cp_info = VkCommandPoolCreateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                queueFamilyIndex: 0,
            };
            assert_eq!(
                vkCreateCommandPool(device, &cp_info, ptr::null(), &mut pool),
                VkResult::VK_SUCCESS
            );

            let mut cmd_buf = ptr::null_mut();
            let cb_info = VkCommandBufferAllocateInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: ptr::null(),
                commandPool: pool,
                level: VkCommandBufferLevel::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount: 1,
            };
            assert_eq!(
                vkAllocateCommandBuffers(device, &cb_info, &mut cmd_buf),
                VkResult::VK_SUCCESS
            );

            assert_eq!(
                vkBeginCommandBuffer(cmd_buf, ptr::null()),
                VkResult::VK_SUCCESS
            );

            let clear_value = VkClearValue {
                color: VkClearColorValue {
                    float32: [0.25, 0.5, 0.75, 1.0],
                },
            };
            let rp_begin = VkRenderPassBeginInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                pNext: ptr::null(),
                renderPass: render_pass,
                framebuffer,
                renderArea: VkRect2D {
                    offset: VkOffset2D { x: 0, y: 0 },
                    extent: VkExtent2D {
                        width: 64,
                        height: 64,
                    },
                },
                clearValueCount: 1,
                pClearValues: &clear_value,
            };
            vkCmdBeginRenderPass(
                cmd_buf,
                &rp_begin,
                VkSubpassContents::VK_SUBPASS_CONTENTS_INLINE,
            );
            vkCmdEndRenderPass(cmd_buf);
            assert_eq!(vkEndCommandBuffer(cmd_buf), VkResult::VK_SUCCESS);

            let mut queue = ptr::null_mut();
            vkGetDeviceQueue(device, 0, 0, &mut queue);

            let submit = VkSubmitInfo {
                sType: VkStructureType::VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: 0,
                pWaitSemaphores: ptr::null(),
                pWaitDstStageMask: ptr::null(),
                commandBufferCount: 1,
                pCommandBuffers: &cmd_buf,
                signalSemaphoreCount: 0,
                pSignalSemaphores: ptr::null(),
            };
            assert_eq!(vkQueueSubmit(queue, 1, &submit, 0), VkResult::VK_SUCCESS);

            // Verify that the color attachment was cleared with the clear color in HAL
            let img_binding = device_box.images.get(&image).expect("image binding");
            let hal_img = device_box.hal.image(img_binding.hal_id).expect("hal image");
            // Check first pixel RGBA values (0.25 * 255.0 = 63.75 -> round to 64, 0.5 * 255.0 = 127.5 -> round to 128, 0.75 * 255.0 = 191.25 -> round to 191, 1.0 * 255 = 255)
            assert_eq!(hal_img.data[0], 64);
            assert_eq!(hal_img.data[1], 128);
            assert_eq!(hal_img.data[2], 191);
            assert_eq!(hal_img.data[3], 255);

            vkDestroyFramebuffer(device, framebuffer, ptr::null());
            vkDestroyRenderPass(device, render_pass, ptr::null());
            vkDestroyImageView(device, view, ptr::null());
            vkDestroyImage(device, image, ptr::null());
            vkDestroyCommandPool(device, pool, ptr::null());
            // `device_box` and `instance_box` are owned by this test and dropped at scope end.
        }
    }
}
