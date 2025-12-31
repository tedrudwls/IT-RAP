import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import datetime
import shutil
from sklearn.cluster import KMeans
import time
import gc
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity # LPIPS metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_perturbation(perturbation):
    """
    Analyzes and records the perturbation value returned by the perturb() function.
    Measures the magnitude of the perturbation by calculating the Euclidean distance 
    of the RGB channel perturbations on a per-pixel basis.

    Args:
        perturbation (torch.Tensor): The perturbation tensor returned from the perturb() function 
                                     (shape: torch.Size([1, 3, 256, 256])).

    Returns:
        numpy.ndarray: The Euclidean distance of the RGB channel perturbation vector for each pixel 
                       (shape: (256, 256)). Returns the recorded perturbation values.
    """
    # Remove the batch dimension (index 0)
    perturbation_no_batch = perturbation.squeeze(0) # shape: torch.Size([3, 256, 256])

    # Calculate Euclidean distance (per-pixel)
    squared_perturbation = perturbation_no_batch ** 2 # Square of perturbation for each channel
    sum_squared_perturbation = torch.sum(squared_perturbation, dim=0) # Sum of squared values along the channel dimension
    euclidean_norm_perturbation = torch.sqrt(sum_squared_perturbation) # Square root of the sum of squares (Euclidean distance) # shape: torch.Size([256, 256])

    # Convert to NumPy array
    perturbation_array = euclidean_norm_perturbation.cpu().numpy() # If on GPU, move to CPU then convert to numpy

    return perturbation_array

"""
Prints various debug logs.
x_real, perturbed_image, original_gen_image, perturbed_gen_image: shape [1, 3, 256, 256] / values [-1, 1]
"""
def print_debug(x_real, perturbed_image, original_gen_image, perturbed_gen_image):
    # Create the image with perturbation added directly here (=x_adv), not the one returned from perturb
    print(f'x_real shape: {x_real.shape}, perturbed_image shape: {perturbed_image.shape}, perturbed_image type: {type(perturbed_image)}')

    # Print min/max values for each RGB channel of x_real
    real_red_min = x_real[0, 0].min()
    real_red_max = x_real[0, 0].max()
    real_green_min = x_real[0, 1].min()
    real_green_max = x_real[0, 1].max()
    real_blue_min = x_real[0, 2].min()
    real_blue_max = x_real[0, 2].max()
    print(f"Original Image Red Min: {real_red_min}, Max: {real_red_max}")
    print(f"Original Image Green Min: {real_green_min}, Max: {real_green_max}")
    print(f"Original Image Blue Min: {real_blue_min}, Max: {real_blue_max}")

    # Print min/max values for each RGB channel of perturbed_image (x_adv)
    adv_red_min = perturbed_image[0, 0].min()
    adv_red_max = perturbed_image[0, 0].max()
    adv_green_min = perturbed_image[0, 1].min()
    adv_green_max = perturbed_image[0, 1].max()
    adv_blue_min = perturbed_image[0, 2].min()
    adv_blue_max = perturbed_image[0, 2].max()
    print(f"Adversarial Example Red Min: {adv_red_min}, Max: {adv_red_max}")
    print(f"Adversarial Example Green Min: {adv_green_min}, Max: {adv_green_max}")
    print(f"Adversarial Example Blue Min: {adv_blue_min}, Max: {adv_blue_max}")


def calculate_and_save_metrics(original_gen_image, perturbed_gen_image, transform_type, results):
    """Calculate and save metrics for each transformation method."""

    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    
    # Calculate distortion metrics (L1, L2)
    l1_error = F.l1_loss(perturbed_gen_image, original_gen_image)
    l2_error = F.mse_loss(perturbed_gen_image, original_gen_image)
    
    results[transform_type]["l1_error"] += l1_error.item()
    results[transform_type]["l2_error"] += l2_error.item()
    
    # Check if the attack was successful
    if l2_error > 0.05:  # Attack success condition (L2 Error Threshold)
        results[transform_type]["attack_success"] += 1
    
    # Calculate deepfake defense effectiveness metrics (PSNR, SSIM, LPIPS)
    original_gen_image_np = original_gen_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    perturbed_gen_image_np = perturbed_gen_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    defense_lpips = lpips_loss(original_gen_image, perturbed_gen_image).mean()
    defense_psnr = psnr(original_gen_image_np, perturbed_gen_image_np, data_range=1.0)
    defense_ssim = ssim(original_gen_image_np, perturbed_gen_image_np, data_range=1.0, win_size=3, channel_axis=2)
    
    results[transform_type]["defense_lpips"] += defense_lpips.item()
    results[transform_type]["defense_psnr"] += defense_psnr
    results[transform_type]["defense_ssim"] += defense_ssim

    return results

"""
Prints the final result metrics.
x_real, perturbed_image, original_gen_image, perturbed_gen_image: shape [1, 3, 256, 256] / values [-1, 1]
"""
def print_final_metrics(episode, total_perturbation_map, total_remain_map, total_l1_error, total_l2_error, attack_success, invisible_psnr, invisible_ssim, invisible_lpips, defense_psnr, defense_ssim, defense_lpips):
    print("\n======== Printing Metrics ========")
    # Calculate the average perturbation map
    average_perturbation_map = total_perturbation_map / episode

    print(f"--- Average Perturbation Over {episode} Iterations ---")
    # Print the average perturbation map array
    print("Average Perturbation Map (256x256 array):\n", average_perturbation_map)

    # Calculate and display the total sum of the average perturbation map (a single float value)
    total_average_perturbation_value = np.sum(average_perturbation_map)
    print("\nTotal Average Perturbation Value (Sum of all pixels in average map) : ", total_average_perturbation_value)

    # After applying image transformation, calculate and display the average of the remaining perturbation
    average_remain_map = total_remain_map / episode
    print("\nAverage Remain Perturbation Map (256x256 array):\n", average_remain_map)

    total_average_remain_perturbation_value = np.sum(average_remain_map)
    print("\nTotal Average Remain Perturbation Value (Sum of all pixels in remain map) : ", total_average_remain_perturbation_value)

    # Evaluate training results (Average L2 Error, Attack success rate)
    print(f'{episode} images. L1 error: {total_l1_error / episode:.5f}. L2 error: {total_l2_error / episode:.5f}. prop_dist: {float(attack_success) / episode:.5f}.')

    print(f'Invisibility PSNR: {invisible_psnr / episode:.5f} dB. Invisibility SSIM: {invisible_ssim / episode:.5f}. Invisibility LPIPS: {invisible_lpips / episode:.5f}\nDeepfake Defense PSNR: {defense_psnr / episode:.5f} dB. Deepfake Defense SSIM: {defense_ssim / episode:.5f}. Deepfake Defense LPIPS: {defense_lpips / episode:.5f}')


"""
Print and save final metrics for all transformation methods.
"""
def print_comprehensive_metrics(results, episode, total_invisible_psnr, total_invisible_ssim, total_invisible_lpips, num_images=None):
    """
    results: 변환 방법별 결과 dictionary
    episode: 총 처리된 속성 개수 (이미지 × 속성)
    total_invisible_psnr/ssim/lpips: 누적된 invisibility 지표
    num_images: 실제 처리된 이미지 개수 (None이면 episode 사용)
    """
    # List of strings to store the results
    output_lines = []

    # --- Header ---
    header1 = "\n" + "="*100
    header2 = "Final Results Summary by Image Transformation Method"
    header3 = "="*100
    
    print(header1)
    print(header2)
    print(header3)
    output_lines.extend([header1, header2, header3])

    # --- Invisibility Metrics ---
    # ✅ 이미지 단위로 평균
    if num_images is None:
        num_images = episode
    
    invisibility_line = f'Invisibility PSNR: {total_invisible_psnr / num_images:.5f} dB. Invisibility SSIM: {total_invisible_ssim / num_images:.5f}. Invisibility LPIPS: {total_invisible_lpips / num_images:.5f}'
    print(invisibility_line)
    output_lines.append(invisibility_line)
    
    separator = "="*100
    print(separator)
    output_lines.append(separator)

    # --- Table Header ---
    table_header = f"{'Transform Type':<15} | {'L1 Error':<10} | {'L2 Error':<10} | {'Noise Attack PSNR':<18} | {'Noise Attack SSIM':<18} | {'Noise Attack LPIPS':<18} | {'Success Rate (%)':<18} | {'Noise Residue'}"
    table_separator = "-"*130
    
    print(table_header)
    print(table_separator)
    output_lines.append(table_header)
    output_lines.append(table_separator)

    total_success_rate = 0.0
    
    for transform_type, metrics in results.items():
        # ✅ 속성 단위로 평균 (episode 사용)
        avg_l1 = metrics["l1_error"] / episode
        avg_l2 = metrics["l2_error"] / episode
        avg_def_psnr = metrics["defense_psnr"] / episode
        avg_def_ssim = metrics["defense_ssim"] / episode
        avg_def_lpips = metrics["defense_lpips"] / episode
        success_rate = metrics["attack_success"] / episode  # ✅ 속성 단위

        total_success_rate += success_rate

        average_remain_map = metrics["total_remain_map"] / episode
        average_remain_perturbation_value = np.sum(average_remain_map)
        
        # Create result line
        result_line = (f"{transform_type:<15} | {avg_l1:<10.4f} | {avg_l2:<10.4f} | "
                       f"{avg_def_psnr:<18.2f} | {avg_def_ssim:<18.4f} | {avg_def_lpips:<18.4f} | "
                       f"{success_rate:<18.2f} | {average_remain_perturbation_value:.2f}")
        print(result_line)
        output_lines.append(result_line)
    
    print(separator)
    output_lines.append(separator)

    # --- File Saving Logic ---
    output_dir = "result_test"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f"training_performance.txt")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\n[Success] Results saved to file: {file_path}")
    except IOError as e:
        print(f"\n[Error] Failed to save file: {e}")
    
    print("="*100)

    # Return objective function value for Optuna
    avg_invisible_psnr = total_invisible_psnr / num_images  # ✅ 이미지 단위
    avg_invisible_ssim = total_invisible_ssim / num_images
    avg_invisible_lpips = total_invisible_lpips / num_images
    avg_success_rate = (total_success_rate - 1) / (len(results) - 1) if len(results) > 1 else 0.0

    score = (avg_invisible_psnr / 27) + (avg_invisible_ssim) + (1 - avg_invisible_lpips) + (2 * avg_success_rate)
    return score
    

def visualize_actions(action_history, image_indices, attr_indices, step_indices):
    """
    Visualizes and analyzes the action selection patterns of the Rainbow DQN reinforcement learning algorithm.
    Works efficiently even with large image datasets (100, 1000, 10000+ images).
    """
    # Record the start time of processing
    start_time = time.time()
    
    # Set font for plotting
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'  # Use DejaVu Sans font (available on most systems)
    except:
        pass  # Use default font if DejaVu Sans is not available
    plt.rcParams['axes.unicode_minus'] = False     # Prevent minus sign from breaking
    
    # Set low resolution for large data processing
    plt.rcParams['figure.dpi'] = 100

    # Create a folder name using the current time
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join("test_result_images", current_time)
    
    # Create folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Move image files from the "result_test" folder to the newly created folder
    result_test_dir = None
    for candidate_dir in ["result_test", "result_test_attgan", "result_inference"]:
        if os.path.exists(candidate_dir):
            result_test_dir = candidate_dir
            break

    if result_test_dir:
        for filename in os.listdir(result_test_dir):
            src_path = os.path.join(result_test_dir, filename)
            dst_path = os.path.join(save_dir, filename)
            shutil.move(src_path, dst_path)
    else:
        print("'result_test', 'result_inference', or 'result_test_att' folder does not exist.")


    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame({
        'Action': action_history,
        'Image': image_indices,
        'Attribute': attr_indices,
        'Step': step_indices
    })
    
    # Check the number of images
    unique_images = df['Image'].nunique()
    unique_attrs = df['Attribute'].nunique()
    total_steps = len(action_history)
    
    print(f"Data Statistics:")
    print(f"- Total images: {unique_images}")
    print(f"- Total attributes: {unique_attrs}")
    print(f"- Total steps: {total_steps}")
    
    # Determine visualization strategy based on the number of images
    is_large_dataset = unique_images > 20
    is_very_large_dataset = unique_images > 100
    is_massive_dataset = unique_images > 1000
    is_ultra_massive_dataset = unique_images > 10000
    
    # Define action names
    ACTION_NAMES = ['PGD Space', 'Low-freq', 'Mid-freq', 'High-freq']
    
    # 1. Visualize Action Distribution - Bar Chart (useful regardless of the number of images)
    print("1. Generating action distribution visualization...")
    plt.figure(figsize=(10, 6))
    series_action_counts = df['Action'].value_counts().sort_index()
    
    # Check if all action indices exist, and add them with a value of 0 if not
    for i in range(4):
        if i not in series_action_counts.index:
            series_action_counts[i] = 0
    series_action_counts = series_action_counts.sort_index()
    
    plt.bar(series_action_counts.index, series_action_counts.values)
    plt.xticks([0, 1, 2, 3], ACTION_NAMES)
    plt.title('Action Selection Distribution Over Entire Training Process')
    plt.xlabel('Action Type')
    plt.ylabel('Selection Count')
    plt.savefig(os.path.join(save_dir, 'action_distribution.png'))
    plt.close()
    
    # Also visualize as a pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(series_action_counts.values, labels=ACTION_NAMES, 
            autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Action Selection Ratio')
    plt.savefig(os.path.join(save_dir, 'action_distribution_pie.png'))
    plt.close()
    
    # 2. Action Selection Pattern Over Time - Line Graph
    print("2. Generating action selection pattern over time visualization...")
    # Apply downsampling if the data is very large
    if is_ultra_massive_dataset:
        # Divide the data into 1000 intervals and display the average action
        max_samples = 1000
        bin_size = max(1, len(action_history) // max_samples)
        
        steps = np.array(range(len(action_history)))
        bins = steps // bin_size
        
        sampled_actions = []
        sampled_steps = []
        
        for bin_idx in range(bins.max() + 1):
            bin_mask = bins == bin_idx
            if np.any(bin_mask):
                sampled_actions.append(np.mean(np.array(action_history)[bin_mask]))
                sampled_steps.append(np.mean(steps[bin_mask]))
        
        plt.figure(figsize=(15, 6))
        plt.plot(sampled_steps, sampled_actions, '-', alpha=0.8)
        plt.yticks([0, 1, 2, 3], ACTION_NAMES)
        plt.title(f'Action Selection Pattern Over Time (Downsampled: {len(sampled_steps)} points)')
    elif is_massive_dataset:
        # Display only the line without markers
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(action_history)), action_history, '-', alpha=0.5)
        plt.yticks([0, 1, 2, 3], ACTION_NAMES)
        plt.title('Action Selection Pattern Over Time (Line Graph)')
    else:
        plt.figure(figsize=(15, 6))
        # For medium-sized datasets, make the markers smaller
        if is_very_large_dataset:
            plt.plot(range(len(action_history)), action_history, '.', markersize=1, alpha=0.3)
        else:
            plt.plot(range(len(action_history)), action_history, 'o', markersize=3, alpha=0.6)
            
        plt.yticks([0, 1, 2, 3], ACTION_NAMES)
        plt.title('Action Selection Pattern Over Time')
    
    plt.xlabel('Step')
    plt.ylabel('Selected Action')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'action_timeline.png'))
    plt.close()
    
    # 3. Change in Action Selection Over Training - Cumulative Graph
    print("3. Generating cumulative action selection visualization...")
    plt.figure(figsize=(12, 6))
    action_timeline = []
    for i in range(4):  # For the 4 actions
        action_counts = [0]
        for a in action_history:
            if a == i:
                action_counts.append(action_counts[-1] + 1)
            else:
                action_counts.append(action_counts[-1])
        action_timeline.append(action_counts[1:])  # Exclude the first 0
    
    for i in range(4):
        plt.plot(range(len(action_history)), action_timeline[i], 
                label=ACTION_NAMES[i])
    
    plt.title('Cumulative Action Selections Over Training')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Selection Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'action_cumulative.png'))
    plt.close()
    
    # 4. Analysis of Action Selection Patterns per Image
    print("4. Analyzing action selection patterns per image...")
    
    # Aggregate actions per image
    img_action_pivot = df.pivot_table(
        index='Image',
        columns='Action',
        aggfunc='size',
        fill_value=0
    )
    
    # Add missing action columns
    for i in range(4):
        if i not in img_action_pivot.columns:
            img_action_pivot[i] = 0
    
    # Sort columns
    img_action_pivot = img_action_pivot.reindex(sorted(img_action_pivot.columns), axis=1)
    
    # Set column names
    img_action_pivot.columns = ACTION_NAMES
    
    # Save per-image action statistics to CSV
    img_action_pivot.to_csv(os.path.join(save_dir, 'image_action_counts.csv'))
    
    # Calculate action ratios per image
    img_action_ratio = img_action_pivot.div(img_action_pivot.sum(axis=1), axis=0)
    img_action_ratio.to_csv(os.path.join(save_dir, 'image_action_ratios.csv'))
    
    # 5. Heatmap Visualization (strategy changes based on the number of images)
    print("5. Determining heatmap visualization strategy...")
    
    # If there are very many images (1,000+)
    if is_massive_dataset:
        print("   Starting clustering analysis for large dataset...")
        # Clustering-based analysis
        
        # For memory efficiency, cluster only a random sample of images
        sample_size = min(5000, unique_images)
        if unique_images > sample_size:
            print(f"   For memory efficiency, sampling {sample_size} images for clustering...")
            sample_images = np.random.choice(img_action_ratio.index, sample_size, replace=False)
            clustering_data = img_action_ratio.loc[sample_images]
        else:
            clustering_data = img_action_ratio
        
        # Set the number of clusters
        n_clusters = min(20, max(3, sample_size // 250))
        
        try:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(clustering_data.fillna(0))
            
            # Save cluster information
            cluster_df = pd.DataFrame({
                'Image': clustering_data.index,
                'Cluster': clusters
            })
            
            # Calculate patterns per cluster
            cluster_patterns = pd.DataFrame()
            for i in range(n_clusters):
                cluster_imgs = cluster_df[cluster_df['Cluster'] == i]['Image']
                if len(cluster_imgs) > 0:
                    pattern = clustering_data.loc[cluster_imgs].mean()
                    cluster_patterns[f'Cluster_{i}'] = pattern
            
            # Cluster pattern heatmap
            plt.figure(figsize=(max(10, n_clusters * 0.8), 6))
            sns.heatmap(cluster_patterns.T, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Average Action Selection Pattern by Cluster')
            plt.savefig(os.path.join(save_dir, 'cluster_patterns.png'))
            plt.close()
            
            # Visualize cluster sizes
            cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()
            
            plt.figure(figsize=(max(10, n_clusters * 0.8), 6))
            bars = plt.bar(cluster_sizes.index, cluster_sizes.values)
            
            # Display value above the bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}', ha='center', va='bottom')
                
            plt.title('Number of Images per Cluster')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Images')
            plt.xticks(cluster_sizes.index)
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'cluster_sizes.png'))
            plt.close()
            
            # Save cluster information to CSV
            cluster_df.to_csv(os.path.join(save_dir, 'image_clusters.csv'))
            
            # Save cluster summary to a text file
            with open(os.path.join(save_dir, 'cluster_summary.txt'), 'w', encoding='utf-8') as f:
                f.write("Image Clustering Results Summary\n")
                f.write("========================\n\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Number of images used for analysis: {len(clustering_data)}")
                if unique_images > sample_size:
                    f.write(f" (Sampled from {unique_images} total)\n")
                else:
                    f.write(" (All)\n")
                    
                f.write("\nNumber of images per cluster:\n")
                for cluster_id, size in cluster_sizes.items():
                    f.write(f"  Cluster {cluster_id}: {size} images ({size/len(clustering_data)*100:.2f}%)\n")
                
                f.write("\nMain action pattern per cluster:\n")
                for cluster_id in range(n_clusters):
                    if f'Cluster_{cluster_id}' in cluster_patterns:
                        pattern = cluster_patterns[f'Cluster_{cluster_id}']
                        main_action = pattern.idxmax()
                        main_action_ratio = pattern[main_action]
                        f.write(f"  Cluster {cluster_id}: Main action '{main_action}' ({main_action_ratio:.2f})\n")
                        f.write(f"    Full distribution: {', '.join([f'{action}: {val:.2f}' for action, val in pattern.items()])}\n")
            
            print("   Clustering analysis complete.")
        except Exception as e:
            print(f"Error during clustering: {e}")
            with open(os.path.join(save_dir, 'error_log.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Error during clustering: {e}")
    
    # If the number of images is moderately large (20-1000)
    elif is_large_dataset:
        print("   Generating sampled heatmap for medium-sized dataset...")
        
        # Image sampling
        max_images_for_heatmap = 50  # Maximum number of images to display on the heatmap
        if unique_images > max_images_for_heatmap:
            # Strategic sampling to select images with diverse action patterns
            # Calculate the main action for each image
            main_actions = img_action_ratio.idxmax(axis=1)
            
            # For balanced sampling, select an equal number of images for each action
            sampled_images = []
            for action in ACTION_NAMES:
                action_images = main_actions[main_actions == action].index.tolist()
                # If there are not enough images for an action, select as many as possible
                n_samples = min(max_images_for_heatmap // len(ACTION_NAMES), len(action_images))
                if n_samples > 0:
                    sampled_images.extend(np.random.choice(action_images, n_samples, replace=False))
            
            # If the number of samples is insufficient, randomly select more
            if len(sampled_images) < max_images_for_heatmap:
                remaining = list(set(img_action_ratio.index) - set(sampled_images))
                n_additional = min(max_images_for_heatmap - len(sampled_images), len(remaining))
                if n_additional > 0:
                    sampled_images.extend(np.random.choice(remaining, n_additional, replace=False))
            
            sampled_heatmap_data = img_action_ratio.loc[sampled_images]
        else:
            sampled_heatmap_data = img_action_ratio
        
        # Generate heatmap
        plt.figure(figsize=(12, max(8, len(sampled_heatmap_data) * 0.3)))
        sns.heatmap(sampled_heatmap_data, annot=True, cmap='viridis', fmt='.2f')
        plt.title(f'Per-Image Action Selection Ratio (Sample of {len(sampled_heatmap_data)})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'image_action_ratio_heatmap.png'))
        plt.close()
    
    # If the number of images is small (20 or less)
    else:
        print("   Generating heatmap for small dataset...")
        # Per-image, per-attribute action selection pattern heatmap (similar to original code)
        # If per-attribute analysis is possible
        if unique_attrs > 1:
            # Pivot table of main actions per image-attribute
            pivot_table = df.pivot_table(
                index='Image', 
                columns='Attribute', 
                values='Action', 
                aggfunc=lambda x: pd.Series.mode(x)[0] if len(x) > 0 else np.nan
            )
            
            # Generate heatmap
            attr_labels = [f'Attr {i+1}' for i in range(unique_attrs)]
            img_labels = [f'Image {i+1}' for i in range(unique_images)]
            
            plt.figure(figsize=(max(8, unique_attrs * 1.5), max(6, unique_images * 0.5)))
            sns.heatmap(pivot_table, annot=True, cmap='viridis', 
                        xticklabels=attr_labels[:unique_attrs],
                        yticklabels=img_labels[:unique_images])
            plt.title('Main Action Selection Pattern per Image/Attribute')
            plt.savefig(os.path.join(save_dir, 'image_attribute_action_heatmap.png'))
            plt.close()
        
        # Per-image action ratio heatmap (small dataset)
        plt.figure(figsize=(10, max(6, unique_images * 0.5)))
        sns.heatmap(img_action_ratio, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Per-Image Action Selection Ratio')
        plt.savefig(os.path.join(save_dir, 'image_action_ratio_heatmap.png'))
        plt.close()
    
    # 6. Analysis of Action Selection Patterns per Attribute
    print("6. Analyzing action selection patterns per attribute...")
    if unique_attrs > 1:  # Perform only if there are multiple attributes
        # Action selection count per attribute
        attr_action_pivot = df.pivot_table(
            index='Attribute',
            columns='Action',
            aggfunc='size',
            fill_value=0
        )
        
        # Add missing action columns
        for i in range(4):
            if i not in attr_action_pivot.columns:
                attr_action_pivot[i] = 0
        
        # Sort columns and set names
        attr_action_pivot = attr_action_pivot.reindex(sorted(attr_action_pivot.columns), axis=1)
        attr_action_pivot.columns = ACTION_NAMES
        
        # Calculate action ratios per attribute
        attr_action_ratio = attr_action_pivot.div(attr_action_pivot.sum(axis=1), axis=0)
        
        # Save to CSV
        attr_action_pivot.to_csv(os.path.join(save_dir, 'attribute_action_counts.csv'))
        attr_action_ratio.to_csv(os.path.join(save_dir, 'attribute_action_ratios.csv'))
        
        # Per-attribute action ratio heatmap
        plt.figure(figsize=(10, max(6, unique_attrs * 0.5)))
        sns.heatmap(attr_action_ratio, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Per-Attribute Action Selection Ratio')
        plt.savefig(os.path.join(save_dir, 'attribute_action_ratio_heatmap.png'))
        plt.close()
        
        # Per-attribute main action bar graph
        plt.figure(figsize=(max(10, unique_attrs * 1.2), 6))
        attr_main_actions = attr_action_ratio.idxmax(axis=1)
        
        # Representative action and ratio per attribute
        main_action_data = []
        for attr, action in attr_main_actions.items():
            ratio = attr_action_ratio.loc[attr, action]
            main_action_data.append((attr, action, ratio))
        
        # Display attribute ID, main action, and ratio
        df_main_actions = pd.DataFrame(main_action_data, columns=['Attribute', 'Main Action', 'Ratio'])
        df_main_actions.to_csv(os.path.join(save_dir, 'attribute_main_actions.csv'))
        
        # Generate bar graph
        attr_indices = range(len(attr_main_actions))
        bars = plt.bar(attr_indices, df_main_actions['Ratio'])
        
        # Display action name above the bar
        for i, bar in enumerate(bars):
            action = df_main_actions.iloc[i]['Main Action']
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    action, ha='center', va='bottom')
        
        plt.xticks(attr_indices, [f'Attr {attr+1}' for attr in attr_main_actions.index])
        plt.ylim(0, 1.1)
        plt.title('Main Action Selection Ratio by Attribute')
        plt.ylabel('Selection Ratio')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'attribute_main_action_bars.png'))
        plt.close()
    else:
        print("   Only one attribute, so per-attribute analysis is not performed.")
    
    # 7. Save Comprehensive Statistics to Text File
    print("7. Generating comprehensive statistics...")
    with open(os.path.join(save_dir, 'action_statistics.txt'), 'w', encoding='utf-8') as f:
        f.write("Rainbow DQN Action Selection Statistics\n")
        f.write("===========================\n\n")
        
        # 1. Basic Information
        f.write("1. Basic Information\n")
        f.write(f"Total images: {unique_images}\n")
        f.write(f"Total attributes: {unique_attrs}\n")
        f.write(f"Total steps: {total_steps}\n\n")
        
        # 2. Per-Action Statistics
        f.write("2. Selection Count and Ratio per Action\n")
        for i in range(4):
            action_name = ACTION_NAMES[i]
            count = series_action_counts[i] if i in series_action_counts.index else 0
            ratio = count/total_steps*100
            f.write(f"{action_name}: {count} times ({ratio:.2f}%)\n")
        f.write("\n")
        
        # 3. Special Analysis Results Based on Image Count
        if is_massive_dataset:
            f.write("3. Large Dataset Special Analysis\n")
            f.write("Clustering analysis was performed for the large dataset (1000+ images).\n")
            f.write("Refer to 'cluster_summary.txt' for detailed clustering results.\n\n")
        
        # 4. Key Patterns
        max_action = series_action_counts.idxmax()
        min_action = series_action_counts.idxmin()
        f.write("4. Key Patterns\n")
        f.write(f"Most selected action: {ACTION_NAMES[max_action]} ({series_action_counts[max_action]} times, {series_action_counts[max_action]/total_steps*100:.2f}%)\n")
        f.write(f"Least selected action: {ACTION_NAMES[min_action]} ({series_action_counts[min_action]} times, {series_action_counts[min_action]/total_steps*100:.2f}%)\n\n")
        
        # 5. Per-Image Action Trends
        if not is_massive_dataset:
            f.write("5. Main Action per Image\n")
            main_actions_by_image = img_action_ratio.idxmax(axis=1)
            
            unique_main_actions = main_actions_by_image.value_counts()
            f.write("Distribution of main actions for images:\n")
            for action, count in unique_main_actions.items():
                f.write(f"- {action}: {count} images ({count/unique_images*100:.2f}%)\n")
            
            f.write("\nMain action and ratio for each image:\n")
            for img, action in main_actions_by_image.items():
                ratio = img_action_ratio.loc[img, action]
                f.write(f"Image {img}: {action} ({ratio:.2f})\n")
        else:
            f.write("5. Main Action per Image\n")
            f.write("Detailed list omitted due to large dataset size.\n")
            f.write("Refer to 'image_action_ratios.csv' for details.\n")
    
    # 8. Record Execution Time
    end_time = time.time()
    execution_time = end_time - start_time
    
    with open(os.path.join(save_dir, 'execution_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Visualization execution time: {execution_time:.2f} seconds\n")
        f.write(f"Number of images processed: {unique_images}\n")
        f.write(f"Number of steps processed: {total_steps}\n")
    
    # Memory cleanup
    gc.collect()
    
    print(f"Analysis complete! Execution time: {execution_time:.2f} seconds")
    print(f"Results have been saved to the '{save_dir}' folder.")


# To get reward trend plots
def plot_reward_trend(reward_list, window_size=25, save_path="reward_trend.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    episodes = np.arange(len(reward_list))
    reward_array = np.array(reward_list)

    # 'valid' mode requires each window to have the same number of episodes, while 'same' or 'full' do not.
    moving_avg = np.convolve(reward_array, np.ones(window_size)/window_size, mode='same')

    plt.figure(figsize=(10, 5))
    plt.plot(episodes[:len(moving_avg)], moving_avg, label=f"Moving Average (window={window_size})", color='tab:blue')
    plt.plot(episodes, reward_array, color='lightgray', alpha=0.4, label='Episodic Return')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Trend Over Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# To output reward values to a txt file
def save_reward_moving_average_txt(reward_list, window_size=25, save_path="reward_moving_avg.txt"):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("==== Raw Episodic Return ====\n")
        for i, r in enumerate(reward_list):
            f.write(f"Episode {i}: {r:.4f}\n")

        f.write("\n\n==== Moving Average of Reward (Window size = {}) ====\n".format(window_size))
        f.write("=" * 50 + "\n")

        num_windows = len(reward_list) // window_size
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = reward_list[start:end]
            avg = sum(window) / len(window)
            f.write(f"Window {i+1} (Episode {start}~{end-1}): Average = {avg:.4f}\n")

        # Last window (less than 25 items)
        remainder = len(reward_list) % window_size
        if remainder > 0:
            start = num_windows * window_size
            end = len(reward_list)
            window = reward_list[start:end]
            avg = sum(window) / len(window)
            f.write(f"Window {num_windows+1} (Episode {start}~{end-1}): Average = {avg:.4f}\n")