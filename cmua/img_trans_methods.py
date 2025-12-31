import torch
import cv2
import numpy as np

from torchvision.io import encode_jpeg, decode_jpeg

import random

def apply_random_transform(original_gen_image, perturbed_gen_image):
    """
    Applies one of six options (five transformation functions + no transformation) randomly.
    
    Args:
        original_gen_image (torch.Tensor): The original generated image tensor, shape=[1, 3, 256, 256], value range -1 to 1.
        perturbed_gen_image (torch.Tensor): The perturbed generated image tensor, shape=[1, 3, 256, 256], value range -1 to 1.
        
    Returns:
        tuple: (random_original_gen_image, random_perturbed_gen_image) - Image tensors with random transformation applied.
    """
    # List of possible transformation options
    transform_options = [
        "no_transform",
        "compress_jpeg",
        "denoise_opencv",
        "median_filter",
        "random_resize_padding",
        "random_image_transforms"
    ]
    
    # Randomly select one transformation option
    selected_transform = random.choice(transform_options)
    print(f"Selected transform: {selected_transform}")
    
    # Apply transformation based on the selected option
    if selected_transform == "no_transform":
        # No transformation - return the original images
        return original_gen_image, perturbed_gen_image
    
    elif selected_transform == "compress_jpeg":
        # Apply JPEG compression
        # Apply the same compression quality to both images
        quality = random.randint(50, 95)  # Random compression quality
        compressed_original = compress_jpeg(original_gen_image, quality)
        compressed_perturbed = compress_jpeg(perturbed_gen_image, quality)
        return compressed_original, compressed_perturbed
    
    elif selected_transform == "denoise_opencv":
        # Apply OpenCV denoising
        denoised_original = denoise_opencv(original_gen_image)
        denoised_perturbed = denoise_opencv(perturbed_gen_image)
        return denoised_original, denoised_perturbed
    
    elif selected_transform == "median_filter":
        # Apply median filter
        denoised_original = denoise_scikit(original_gen_image)
        denoised_perturbed = denoise_scikit(perturbed_gen_image)
        return denoised_original, denoised_perturbed
    
    elif selected_transform == "random_resize_padding":
        # Apply random resizing and padding
        transformed_original, transformed_perturbed = random_resize_padding(original_gen_image, perturbed_gen_image)
        return transformed_original, transformed_perturbed
    
    elif selected_transform == "random_image_transforms":
        # Apply random image transformations
        transformed_original, transformed_perturbed = random_image_transforms(original_gen_image, perturbed_gen_image)
        return transformed_original, transformed_perturbed
    
    else:
        # If an unexpected option is selected, return the originals
        print("Unknown transform option selected. Returning original images.")
        return original_gen_image, perturbed_gen_image


def compress_jpeg(x_adv, quality=75):
    """
    Applies JPEG compression to a given tensor and returns the compressed tensor.
    
    Args:
        x_adv (torch.Tensor): Input tensor with shape [1, 3, 256, 256] and value range [-1, 1].
        quality (int): JPEG compression quality (1-100), lower means more compression.
    
    Returns:
        torch.Tensor: A compressed tensor with the same shape and value range as the input.
    """
    # Store the current device
    device = x_adv.device
    
    # Move the tensor to CPU for JPEG encoding (if necessary)
    x_adv_cpu = x_adv.to('cpu')
    
    # Convert from [-1, 1] range to [0, 255] range
    x_adv_uint8 = ((x_adv_cpu + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    
    # Batch processing (currently batch_size=1, but for generalization)
    compressed_batch = []
    for i in range(x_adv_uint8.size(0)):
        # JPEG encoding (compression)
        encoded = encode_jpeg(x_adv_uint8[i], quality=quality)
        
        # JPEG decoding (decompression)
        decoded = decode_jpeg(encoded)
        
        compressed_batch.append(decoded)
    
    # Stack back into a batch
    compressed = torch.stack(compressed_batch)
    
    # Convert back from [0, 255] range to [-1, 1] range
    compressed_float = (compressed.float() / 127.5) - 1
    
    # Move back to the original device
    compressed_float = compressed_float.to(device)
    
    return compressed_float


def denoise_opencv(x_adv):
    """
    Denoises a PyTorch tensor image using OpenCV.
    
    Args:
        x_adv (torch.Tensor): The image tensor to be denoised, shape=[1, 3, 256, 256], value range -1 to 1.
        
    Returns:
        torch.Tensor: The denoised image tensor, shape=[1, 3, 256, 256], value range -1 to 1.
    """
    # If the input tensor is on GPU, move it to CPU
    device = x_adv.device
    x_cpu = x_adv.detach().cpu()
    
    # Check batch size
    batch_size = x_cpu.shape[0]
    
    # Initialize a tensor to store the results
    result = torch.zeros_like(x_cpu)
    
    for i in range(batch_size):
        # 1. Convert PyTorch tensor to NumPy array
        img_np = x_cpu[i].numpy()
        
        # 2. Change channel order: [C, H, W] -> [H, W, C]
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # 3. Convert value range: -1~1 -> 0~255
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)
        
        # 4. Apply OpenCV denoising
        # h=10: Filter strength for luminance component of a color image. Higher values mean stronger noise removal.
        # hColor=10: Filter strength for color components of the image.
        # templateWindowSize=7: Size of the template patch that is used to compute weights (7 is recommended).
        # searchWindowSize=21: Size of the window that is used to compute weighted average (21 is recommended).
        denoised_img = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
        
        # 5. Convert value range: 0~255 -> -1~1
        denoised_img = (denoised_img.astype(np.float32) / 127.5) - 1
        
        # 6. Change channel order: [H, W, C] -> [C, H, W]
        denoised_img = np.transpose(denoised_img, (2, 0, 1))
        
        # 7. Convert NumPy array to PyTorch tensor
        result[i] = torch.from_numpy(denoised_img)
    
    # Move the tensor to the original device
    return result.to(device)


from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, estimate_sigma
from skimage import filters

def denoise_scikit(x_adv):
    """
    Processes a PyTorch tensor image with various scikit-image denoising algorithms.
    
    Args:
    - x_adv: torch.Tensor, shape [1, 3, 256, 256], value range -1 to 1.
    
    Returns:
    - denoised: torch.Tensor, with the same shape and value range as the input.
    """
    # 1. Convert PyTorch tensor to NumPy array
    x_np = x_adv.detach().cpu().numpy()
    
    # 2. Remove batch dimension and change channel order (PyTorch: [B, C, H, W] -> scikit-image: [H, W, C])
    x_img = np.squeeze(x_np, axis=0)  # Remove batch dimension
    x_img = np.transpose(x_img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    
    # 3. Convert value range: -1~1 -> 0~1
    x_img = (x_img + 1) / 2
    
    # Estimate noise level (used in some algorithms)
    sigma_est = estimate_sigma(x_img, channel_axis=-1)
    # print(f"[DEBUG] Estimated noise level: {sigma_est}")
    
    # 4. Apply various denoising algorithms (uncomment the desired one)
    
    # 4.1 Total Variation denoising
    # weight: Parameter to adjust the denoising strength. Higher values (e.g., 0.1 -> 0.3) result in stronger denoising but more loss of detail. Typically between 0.01 and 0.5.
    # denoised_img = denoise_tv_chambolle(x_img, weight=0.1, channel_axis=-1)
    
    # 4.2 Bilateral denoising
    # sigma_color: Denoising strength based on color similarity. Higher values smooth a wider range of colors, resulting in stronger denoising.
    # sigma_spatial: Denoising strength based on spatial similarity. Higher values consider a wider area of pixels.
    # denoised_img = denoise_bilateral(x_img, sigma_color=0.05, sigma_spatial=15, channel_axis=-1)
    
    # 4.3 Wavelet denoising
    # wavelet: Type of wavelet to use. Different wavelets ('haar', 'db2', 'sym9', etc.) provide different denoising characteristics.
    # rescale_sigma: If True, automatically adjusts to the noise level.
    # method: 'BayesShrink' uses an adaptive threshold, while 'VisuShrink' provides more aggressive denoising.
    # denoised_img = denoise_wavelet(x_img, wavelet='db1', method='BayesShrink', channel_axis=-1, rescale_sigma=True)
    
    # 4.4 Non-local Means denoising
    # h: Parameter that directly controls denoising strength. Higher values (e.g., 0.8 -> 1.5) result in stronger denoising. Typically 0.8-1.2 times the noise standard deviation (sigma_est).
    # patch_size: Size of patches to compare. Larger values preserve more structural information but increase computation cost.
    # patch_distance: Search area radius. Larger values find similar patches in a wider area but increase computation cost.
    # denoised_img = denoise_nl_means(x_img, h=0.8 * sigma_est, patch_size=5, 
    #                              patch_distance=6, channel_axis=-1)
    
    # 4.5 Median filtering
    # selem: Adjusts denoising strength by specifying the structuring element size.
    # For example, `from skimage.morphology import disk; selem=disk(2)` uses a circular mask of radius 2.
    # A larger mask results in stronger denoising.
    denoised_img = np.zeros_like(x_img)
    for i in range(x_img.shape[2]):  # Apply to each channel
        denoised_img[:, :, i] = filters.median(x_img[:, :, i])
    
    # 4.6 Gaussian filtering
    # sigma: Standard deviation of the Gaussian kernel. Directly controls denoising strength.
    # Higher values (e.g., 1 -> 2) cause stronger blurring (smoothing) to remove noise.
    # Typically between 0.5 and 3.
    # denoised_img = np.zeros_like(x_img)
    # for i in range(x_img.shape[2]):  # Apply to each channel
    #     denoised_img[:, :, i] = filters.gaussian(x_img[:, :, i], sigma=1)

    
    # 5. Convert value range back to -1~1
    denoised_img = denoised_img * 2 - 1
    
    # 6. Change dimension order and add batch dimension ([H, W, C] -> [B, C, H, W])
    denoised_img = np.transpose(denoised_img, (2, 0, 1))  # [H, W, C] -> [C, H, W]
    denoised_img = np.expand_dims(denoised_img, axis=0)  # [C, H, W] -> [1, C, H, W]
    
    # 7. Convert NumPy array to PyTorch tensor
    denoised_tensor = torch.from_numpy(denoised_img).to(x_adv.device)
    
    return denoised_tensor


import torch.nn.functional as F
import random

def random_resize_padding(x_real, x_adv):
    """
    Applies the same random resizing and padding to x_adv and x_real.
    
    Args:
        x_real (torch.Tensor): Tensor of shape [1, 3, 256, 256] (value range: -1 to 1).
        x_adv (torch.Tensor): Tensor of shape [1, 3, 256, 256] (value range: -1 to 1).
        
    Returns:
        tuple: (transformed_x_real, transformed_x_adv), each with shape [1, 3, 256, 256].
    """
    # Original image size
    original_size = 256
    
    # Randomly select one of the possible resize sizes
    resize_sizes = [208, 224, 240]
    resize_size = random.choice(resize_sizes)
    print(f"Resize selected for this step: {resize_size}")
    
    # Resize both images to the same size
    x_adv_resized = F.interpolate(x_adv, size=(resize_size, resize_size), mode='bilinear', align_corners=False)
    x_real_resized = F.interpolate(x_real, size=(resize_size, resize_size), mode='bilinear', align_corners=False)
    
    # Calculate padding size (total padding)
    pad_diff = original_size - resize_size
    
    # Randomly determine padding location
    pad_left = random.randint(0, pad_diff)
    pad_right = pad_diff - pad_left
    pad_top = random.randint(0, pad_diff)
    pad_bottom = pad_diff - pad_top
    
    # Apply padding (pad_left, pad_right, pad_top, pad_bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    
    x_adv_padded = F.pad(x_adv_resized, padding, mode='constant', value=0)
    x_real_padded = F.pad(x_real_resized, padding, mode='constant', value=0)
    
    return x_real_padded, x_adv_padded


import torchvision.transforms.functional as TF

def random_image_transforms(x_real, x_adv):
    """
    Applies the same random image transformations to x_adv and x_real.
    
    Args:
        x_real (torch.Tensor): Tensor of shape [1, 3, 256, 256] (value range: -1 to 1).
        x_adv (torch.Tensor): Tensor of shape [1, 3, 256, 256] (value range: -1 to 1).
        
    Returns:
        tuple: (transformed_x_real, transformed_x_adv), each with shape [1, 3, 256, 256].
    """
    batch_size, channels, height, width = x_adv.shape
    
    # Randomly decide whether to apply each transformation (0.5 probability)
    apply_shear = random.random() > 0.5
    apply_shift = random.random() > 0.5
    apply_zoom = random.random() > 0.5
    apply_rotation = random.random() > 0.5
    
    # Set transformation parameters
    angle = random.uniform(-15, 15) if apply_rotation else 0
    shear = (random.uniform(-10, 10) if apply_shear else 0,
            random.uniform(-10, 10) if apply_shear else 0)
    translate = (int(random.uniform(-0.1, 0.1) * width) if apply_shift else 0,
               int(random.uniform(-0.1, 0.1) * height) if apply_shift else 0)
    scale = random.uniform(0.9, 1.1) if apply_zoom else 1.0
    
    # Convert original range from [-1, 1] to [0, 1]
    x_real_01 = (x_real + 1) / 2
    x_adv_01 = (x_adv + 1) / 2
    
    # Apply transformations
    transformed_x_real = []
    transformed_x_adv = []
    
    for i in range(batch_size):
        # Apply the same transformation to each image
        trans_img_real = TF.affine(x_real_01[i], angle=angle, translate=translate, 
                            scale=scale, shear=shear,
                            interpolation=TF.InterpolationMode.BILINEAR)

        trans_img_adv = TF.affine(x_adv_01[i], angle=angle, translate=translate, 
                            scale=scale, shear=shear, 
                            interpolation=TF.InterpolationMode.BILINEAR)
        
        
        transformed_x_real.append(trans_img_real)
        transformed_x_adv.append(trans_img_adv)
    
    # Stack into a batch tensor
    transformed_x_real = torch.stack(transformed_x_real)
    transformed_x_adv = torch.stack(transformed_x_adv)
    
    # Convert range back from [0, 1] to [-1, 1]
    transformed_x_real = transformed_x_real * 2 - 1
    transformed_x_adv = transformed_x_adv * 2 - 1
    
    return transformed_x_real, transformed_x_adv
