"""
inference_cmua method to be added to SolverRainbow class in stargan_solver.py

This method performs CMUA (Cross-Model Universal Adversarial Watermark) inference.
"""

def inference_cmua(self, data_loader, result_dir):
    """
    Perform CMUA inference to generate universal adversarial perturbations.
    
    Args:
        data_loader: DataLoader for test images
        result_dir: Directory to save results
    """
    import os
    import time
    import numpy as np
    import torch
    from torchvision.utils import save_image
    from optuna_util import analyze_perturbation, print_comprehensive_metrics, visualize_actions, calculate_and_save_metrics
    from img_trans_methods import compress_jpeg, denoise_opencv, denoise_scikit, random_resize_padding, random_image_transforms, apply_random_transform
    
    # Import CMUA attack class
    from cmua_implementation import CMUAAttack
    
    os.makedirs(result_dir, exist_ok=True)
    self.G.eval()
    
    # Initialize CMUA attack
    cmua_attack = CMUAAttack(
        epsilon=self.config.cmua_epsilon,
        step_size=self.config.cmua_step_size,
        iterations=self.config.cmua_iterations,
        momentum=self.config.cmua_momentum,
        device=self.device
    )
    
    # Initialize result tracking
    total_perturbation_map = np.zeros((256, 256))
    results = {
        "원본(변형없음)": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
        "JPEG압축": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
        "OpenCV디노이즈": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
        "중간값스무딩": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
        "크기조정패딩": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
        "이미지변환": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))}
    }
    total_invisible_psnr, total_invisible_ssim, total_invisible_lpips = 0.0, 0.0, 0.0
    
    episode = 0
    
    # Phase 1: Generate universal perturbation using a batch of training images
    print(f"\n{'='*80}")
    print(f"[CMUA Phase 1] Generating Universal Perturbation")
    print(f"{'='*80}")
    
    training_images = []
    training_labels = []
    
    # Collect training batch for universal perturbation generation
    for batch_idx, (x_real, c_org, filename) in enumerate(data_loader):
        if batch_idx >= self.config.cmua_batch_size:
            break
        
        x_real = x_real.to(self.device)
        c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        
        for c_trg in c_trg_list:
            c_trg = c_trg.to(self.device)
            training_images.append(x_real)
            training_labels.append(c_trg)
    
    # Generate universal perturbation
    print(f"[CMUA] Generating universal perturbation with {len(training_images)} image-attribute pairs...")
    
    for idx, (x_train, c_train) in enumerate(zip(training_images, training_labels)):
        print(f"\n[CMUA] Processing training pair {idx+1}/{len(training_images)}")
        cmua_attack.attack_stargan(x_train, self.G, c_train)
    
    universal_perturbation = cmua_attack.get_universal_perturbation()
    print(f"\n[CMUA] Universal perturbation generated successfully!")
    print(f"[CMUA] Perturbation shape: {universal_perturbation.shape}")
    print(f"[CMUA] Perturbation L_inf norm: {universal_perturbation.abs().max().item():.6f}")
    print(f"[CMUA] Perturbation L2 norm: {universal_perturbation.norm().item():.6f}")
    
    # Save universal perturbation
    perturbation_path = os.path.join(result_dir, 'universal_perturbation.pt')
    torch.save(universal_perturbation, perturbation_path)
    print(f"[CMUA] Universal perturbation saved to: {perturbation_path}")
    
    # Phase 2: Apply universal perturbation to test images
    print(f"\n{'='*80}")
    print(f"[CMUA Phase 2] Applying Universal Perturbation to Test Images")
    print(f"{'='*80}\n")
    
    total_core_time = 0.0
    total_processing_time = 0.0
    
    for infer_img_idx, (x_real, c_org, filename) in enumerate(data_loader):
        total_start_time = time.time()
        image_core_time = 0.0
        
        x_real = x_real.to(self.device)
        
        # Lists to save results
        noattack_result_list = [x_real]
        jpeg_result_list = [x_real]
        opencv_result_list = [x_real]
        median_result_list = [x_real]
        padding_result_list = [x_real]
        transforms_result_list = [x_real]
        
        c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        
        for idx, c_trg in enumerate(c_trg_list):
            c_trg = c_trg.to(self.device)
            
            # Start core processing time
            core_start_time = time.time()
            
            # Apply universal perturbation
            perturbed_image = cmua_attack.apply_universal_perturbation(x_real)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            core_end_time = time.time()
            step_core_time = core_end_time - core_start_time
            image_core_time += step_core_time
            
            print(f"[CMUA Core Processing Time] Applied universal perturbation: {step_core_time:.5f}s")
            
            # Accumulate perturbation map
            analyzed_perturbation_array = analyze_perturbation(perturbed_image - x_real)
            total_perturbation_map += analyzed_perturbation_array
            
            # [Inference 1] No transformation (Original)
            with torch.no_grad():
                remain_perturb_array = analyze_perturbation(perturbed_image - x_real)
                results["원본(변형없음)"]["total_remain_map"] += remain_perturb_array
                original_gen_image, _ = self.G(x_real, c_trg)
                perturbed_gen_image_orig, _ = self.G(perturbed_image, c_trg)
                noattack_result_list.append(perturbed_image)
                noattack_result_list.append(original_gen_image)
                noattack_result_list.append(perturbed_gen_image_orig)
                results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_orig, "원본(변형없음)", results)
            
            # [Inference 2] JPEG Compression
            x_adv_jpeg = compress_jpeg(perturbed_image, quality=75)
            with torch.no_grad():
                remain_perturb_array = analyze_perturbation(x_adv_jpeg - x_real)
                results["JPEG압축"]["total_remain_map"] += remain_perturb_array
                perturbed_gen_image_jpeg, _ = self.G(x_adv_jpeg, c_trg)
                jpeg_result_list.append(x_adv_jpeg)
                jpeg_result_list.append(original_gen_image)
                jpeg_result_list.append(perturbed_gen_image_jpeg)
                results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_jpeg, "JPEG압축", results)
            
            # [Inference 3] OpenCV Denoising
            x_adv_opencv = denoise_opencv(perturbed_image)
            with torch.no_grad():
                remain_perturb_array = analyze_perturbation(x_adv_opencv - x_real)
                results["OpenCV디노이즈"]["total_remain_map"] += remain_perturb_array
                perturbed_gen_image_opencv, _ = self.G(x_adv_opencv, c_trg)
                opencv_result_list.append(x_adv_opencv)
                opencv_result_list.append(original_gen_image)
                opencv_result_list.append(perturbed_gen_image_opencv)
                results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_opencv, "OpenCV디노이즈", results)
            
            # [Inference 4] Median Smoothing
            x_adv_median = denoise_scikit(perturbed_image)
            with torch.no_grad():
                remain_perturb_array = analyze_perturbation(x_adv_median - x_real)
                results["중간값스무딩"]["total_remain_map"] += remain_perturb_array
                perturbed_gen_image_median, _ = self.G(x_adv_median, c_trg)
                median_result_list.append(x_adv_median)
                median_result_list.append(original_gen_image)
                median_result_list.append(perturbed_gen_image_median)
                results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_median, "중간값스무딩", results)
            
            # [Inference 5] Resize and Padding
            x_adv_padding = random_resize_padding(perturbed_image)
            with torch.no_grad():
                remain_perturb_array = analyze_perturbation(x_adv_padding - x_real)
                results["크기조정패딩"]["total_remain_map"] += remain_perturb_array
                perturbed_gen_image_padding, _ = self.G(x_adv_padding, c_trg)
                padding_result_list.append(x_adv_padding)
                padding_result_list.append(original_gen_image)
                padding_result_list.append(perturbed_gen_image_padding)
                results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_padding, "크기조정패딩", results)
            
            # [Inference 6] Random Transformations
            x_adv_transforms = apply_random_transform(perturbed_image)
            with torch.no_grad():
                remain_perturb_array = analyze_perturbation(x_adv_transforms - x_real)
                results["이미지변환"]["total_remain_map"] += remain_perturb_array
                perturbed_gen_image_transforms, _ = self.G(x_adv_transforms, c_trg)
                transforms_result_list.append(x_adv_transforms)
                transforms_result_list.append(original_gen_image)
                transforms_result_list.append(perturbed_gen_image_transforms)
                results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_transforms, "이미지변환", results)
        
        # Calculate invisibility metrics
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        
        perturbed_np = perturbed_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        x_real_np = x_real.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        invisible_psnr = psnr(x_real_np, perturbed_np, data_range=2.0)
        invisible_ssim = ssim(x_real_np, perturbed_np, data_range=2.0, channel_axis=2)
        invisible_lpips = self.lpips_loss(x_real, perturbed_image).item()
        
        total_invisible_psnr += invisible_psnr
        total_invisible_ssim += invisible_ssim
        total_invisible_lpips += invisible_lpips
        
        episode += 1
        
        # Save concatenated results
        all_result_lists = [noattack_result_list, jpeg_result_list, opencv_result_list, 
                           median_result_list, padding_result_list, transforms_result_list]
        
        row_images = []
        for result_list in all_result_lists:
            row_concat = torch.cat(result_list, dim=3)
            row_images.append(row_concat)
        
        spacing = 10
        blank_image = torch.ones_like(row_images[0][:, :, :spacing, :]) * 1.0
        
        vertical_concat_list = [row_images[0]]
        for i in range(1, len(row_images)):
            vertical_concat_list.append(blank_image)
            vertical_concat_list.append(row_images[i])
        
        x_concat = torch.cat(vertical_concat_list, dim=2)
        result_path = os.path.join(result_dir, '{}-images.jpg'.format(infer_img_idx + 1))
        save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
        print(f"[CMUA Inference] Result saved: {result_path}")
        
        # Time tracking
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        
        total_core_time += image_core_time
        total_processing_time += total_elapsed_time
        
        print(f"[CMUA Core Processing Time] Image {infer_img_idx + 1} core time: {image_core_time:.5f}s")
        print(f"[CMUA Total Processing Time] Image {infer_img_idx + 1} total time: {total_elapsed_time:.5f}s\n")
        
        if infer_img_idx >= (self.inference_image_num - 1):
            break
    
    # Print final metrics
    score = print_comprehensive_metrics(results, episode, total_invisible_psnr, total_invisible_ssim, total_invisible_lpips)
    
    print(f"\n{'='*80}")
    print(f"[CMUA Inference Complete]")
    print(f"{'='*80}")
    print(f"Total core processing time: {total_core_time:.5f}s")
    print(f"Total processing time: {total_processing_time:.5f}s")
    print(f"Average core time per image: {total_core_time / episode:.5f}s")
    print(f"Average total time per image: {total_processing_time / episode:.5f}s")
    print(f"Number of images processed: {episode}")
    print(f"{'='*80}\n")
    
    return score
