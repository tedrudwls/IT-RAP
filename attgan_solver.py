from stargan_model import Generator, Discriminator
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np
import os
import random
import math
import gc
import time
from torchvision import transforms

from torchvision.models import vgg19, resnet50, VGG19_Weights, ResNet50_Weights
from torch.nn.utils import clip_grad_norm_ # Gradient Clipping
from torch.optim import Adam
from collections import namedtuple, deque
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity # Add LPIPS metric

from optuna_util import analyze_perturbation, print_debug, print_final_metrics, print_comprehensive_metrics, visualize_actions, calculate_and_save_metrics, plot_reward_trend, save_reward_moving_average_txt


from img_trans_methods import compress_jpeg, denoise_opencv, denoise_scikit, random_resize_padding, random_image_transforms, apply_random_transform

from segment_tree import MinSegmentTree, SumSegmentTree # PrioritizedReplayBuffer

from meso_net import Meso4, MesoInception4, convert_tf_weights_to_pytorch

from attgan.attgan_init import attgan_model
import attgan_attacks

from ellzaf_ml.models import GhostFaceNetsV2
import neptune

# To maintain test reproducibility
np.random.seed(0)

"""Noisy Layer"""
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        # Use cloned weight_epsilon and bias_epsilon
        we = self.weight_epsilon.clone()
        be = self.bias_epsilon.clone()
        return F.linear(x, self.weight_mu + self.weight_sigma * we, self.bias_mu + self.bias_sigma * be)

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


"""Rainbow DQN Network (DuelingNet + NoisyNet + Categorical DQN)"""
class RainbowDQNNet(nn.Module):
    def __init__(self, input_dim, action_dim, atom_size, support): # Add variables related to Categorical DQN
        super(RainbowDQNNet, self).__init__()
        self.atom_size = atom_size
        self.support = support

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        # Dueling Network + Noisy Network
        self.value_hidden = NoisyLinear(128, 128) # Use NoisyLinear
        self.value_layer = NoisyLinear(128, atom_size) # Use NoisyLinear

        self.advantage_hidden = NoisyLinear(128, 128) # Use NoisyLinear
        self.advantage_layer = NoisyLinear(128, action_dim * atom_size) # Use NoisyLinear

        self.action_dim = action_dim

    def forward(self, state):
        dist = self.get_distribution(state) # Approximate Q value function with distribution
        q_values = torch.sum(dist * self.support, dim=2) # distribution * support = Q value
        return q_values

    """Return distribution"""
    def get_distribution(self, state):
        feature = self.feature_layer(state)

        value_hid = F.relu(self.value_hidden(feature))
        value = self.value_layer(value_hid).view(-1, 1, self.atom_size) # Value Stream, output atom_size

        advantage_hid = F.relu(self.advantage_hidden(feature))
        advantage = self.advantage_layer(advantage_hid).view(-1, self.action_dim, self.atom_size) # Advantage Stream, output action_dim * atom_size

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True) # Dueling Network structure

        dist = F.softmax(q_atoms, dim=-1) # Generate probability distribution with Softmax
        dist = dist.clamp(min=1e-3, max=1.0) # Prevent NaN

        return dist

    """Noisy Layer Reset"""
    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_layer.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_layer.reset_noise()


"""Prioritized Replay Buffer (PER) with N-step Learning: Store and sample N-step Transitions"""
class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha, beta_start, beta_frames, n_step, gamma): # Add parameters related to N-step Learning
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.n_step = n_step
        self.gamma = gamma

        self.memory = deque(maxlen=capacity) # Prioritized Replay Buffer (using deque)
        self.priorities_sum_tree = SumSegmentTree(capacity) # Sum Segment Tree
        self.priorities_min_tree = MinSegmentTree(capacity) # Min Segment Tree
        self.max_priority = 1.0 # Set initial max priority
        self.n_step_buffer = deque(maxlen=n_step) # Add N-step buffer
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) # Define Transition

        assert alpha >= 0
        self._storage = []
        self._maxsize = capacity
        self._next_idx = 0

    def beta_by_frame(self, frame_idx): # Beta annealing
        beta_initial = self.beta_start
        beta_annealing_frames = self.beta_frames
        beta_final = 1.0
        return min(beta_final, beta_initial + frame_idx * (beta_final - beta_initial) / beta_annealing_frames)

    """Create and store N-step transition"""
    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) == self.n_step: # Store transition only when the N-step buffer is full
            state, action = self.n_step_buffer[0][:2] # Starting state, action
            n_step_reward = 0
            next_s = next_state # Initialization
            for i in range(self.n_step):
                r, s, d = self.n_step_buffer[i][2:]
                n_step_reward += r * (self.gamma ** i)
                next_s, done = (s, d) if d else (next_s, done) # done=True encountered, update next_state and done

            max_priority = self.max_priority # Use max priority
            n_step_transition = (state, action, n_step_reward, next_s, torch.tensor([float(done)])) # done value is a tensor
            self.memory.append(n_step_transition)

            self.priorities_sum_tree[self._next_idx] = max_priority ** self.alpha # Store priority in Sum tree (alpha squared)
            self.priorities_min_tree[self._next_idx] = max_priority ** self.alpha # Store priority in Min tree (alpha squared)
            self._next_idx = (self._next_idx + 1) % self._maxsize # Circular buffer index
            self.n_step_buffer.popleft() # N-step buffer popleft()


    def _get_priority(self, index):
        return self.priorities_sum_tree[index]

    def _set_priority(self, index, priority):
        priority_alpha = priority ** self.alpha
        self.priorities_sum_tree[index] = priority_alpha
        self.priorities_min_tree[index] = priority_alpha
        self.max_priority = max(self.max_priority, priority) # Update max priority

    """frame_idx for beta annealing"""
    def sample(self, batch_size, frame_idx=None):
        if frame_idx is not None: # Apply beta annealing
            self.beta = self.beta_by_frame(frame_idx)

        indices = self._sample_proportional(batch_size) # Proportional sampling
        transitions = [self.memory[idx] for idx in indices] # Extract transition by index

        weights = [self._calculate_weight(idx, frame_idx) for idx in indices] # Calculate Importance Sampling weight

        batch = self.Transition(*zip(*transitions)) # Create transition batch
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(device) # Convert weights to tensor, unsqueeze for element-wise multiplication
        indices_tensor = torch.LongTensor(indices).to(device) # Convert indices to tensor


        return batch, weights_tensor, indices_tensor # Return weight and index together

    """Proportional sampling"""
    def _sample_proportional(self, batch_size):
        indices = []
        p_total = self.priorities_sum_tree.sum(0, len(self) - 1) # Total priority sum
        segment = p_total / batch_size # Calculate segment

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b) # Uniform sampling within the segment
            idx = self.priorities_sum_tree.retrieve(upperbound) # Retrieve index (sum tree)
            indices.append(idx)
        return indices

    """Calculate Importance Sampling weight"""
    def _calculate_weight(self, idx, frame_idx):
        prob = self.priorities_sum_tree[idx] / self.priorities_sum_tree.sum() # Calculate sampling probability
        N = len(self)
        weight = (prob * N) ** -self.beta # Calculate IS weight, apply beta annealing
        return weight / ((self.priorities_min_tree.min() / self.priorities_sum_tree.sum() * N) ** -self.beta) # Weight normalization


    def update_priorities(self, indices, priorities): # Update priorities
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._set_priority(idx, priority) # Update priority (segment tree)


    def __len__(self):
        return len(self.memory)


"""SolverRainbow for training and testing StarGAN and Rainbow DQN Attack."""
class SolverRainbow(object):
    def __init__(self, dataset_loader, config, run = None):
        self.config = config # For Optuna
        
        # neptune logging
        self.run = run 
        
        # Parameters related to Data loader
        self.dataset_loader = dataset_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Parameters related to Model configurations
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Parameters related to Training configurations
        self.dataset = config.dataset
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.selected_attrs = config.selected_attrs
        self.training_image_num = config.training_image_num
        self.inference_image_num = config.inference_image_num 
        self.reward_weight = config.reward_weight # Reward weight (Trade-off between deepfake defense and invisibility)

        # Parameters related to Test configurations
        self.test_iters = config.test_iters

        # Parameters related to Directories
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Rainbow DQN Hyperparameters (loaded from config in new_main.py)
        self.batch_size = config.batch_size # RLAB batch_size
        self.agent_lr = config.agent_lr
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.target_update_interval = config.target_update_interval
        self.memory_capacity = config.memory_capacity
        self.max_steps_per_episode = config.max_steps_per_episode
        self.action_dim = config.action_dim
        self.noise_level = config.noise_level
        self.feature_extractor_name = config.feature_extractor_name # For State
        self.feature_extractor_frequency = config.feature_extractor_frequency # Feature extractor call frequency
        
        # Feature extraction caching
        self.step_counter = 0
        self.cached_state = None

        # Parameters related to PER
        self.alpha = config.alpha # PER alpha
        self.beta_start = config.beta_start # PER beta_start
        self.beta_frames = config.beta_frames # PER beta_frames
        self.prior_eps = config.prior_eps # PER prior_eps

        # Parameters related to Categorical DQN
        self.v_min = config.v_min # Categorical DQN v_min
        self.v_max = config.v_max # Categorical DQN v_max
        self.atom_size = config.atom_size # Categorical DQN atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device) # Create Support

        # Parameters related to N-step Learning
        self.n_step = config.n_step # N-step Learning n_step

        # Initialize LPIPS Metric
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)

        # Initialize Neptune logging
        self.run = None
        if self.config.neptune_api_token and self.config.neptune_project:
            try:
                self.run = neptune.init_run(
                    project=self.config.neptune_project,
                    api_token=self.config.neptune_api_token,
                    tags=["RLAB", "AttGAN", self.config.dataset, "GhostFaceNet"]
                )
                self.run["config"] = vars(self.config)
                print(f"[INFO] Neptune run initialized: {self.run.get_url()}")
            except Exception as e:
                print(f"[WARNING] Failed to initialize Neptune: {e}")
                self.run = None

        self.build_model() # Build StarGAN model
        self.build_rlab_agent() # Build RLAB Agent


    def inference_rainbow_dqn(self, data_loader, result_dir):
        os.makedirs(result_dir, exist_ok=True)
        self.attack_func = attgan_attacks.AttackFunction(config=self.config, model=attgan_model.G, device=self.device)
        self.rl_agent.dqn.eval()

        # Full list of 13 attributes for AttGAN (same as in train_attack)
        self.attgan_attrs = [
            'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
            'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
        ]
        
        total_perturbation_map = np.zeros((256, 256))
        total_remain_map = np.zeros((256, 256))

        # Dictionary to store results (maintaining the existing structure)
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

        # Variables for recording and visualizing action selections
        action_history = []
        image_indices = []
        attr_indices = []
        step_indices = []

        # Initialize time measurement variables
        total_core_time = 0.0
        total_processing_time = 0.0
        
        # Initialize detailed time measurement variables
        total_generator_time = 0.0
        total_feature_extraction_time = 0.0
        total_feature_cached_time = 0.0
        total_action_selection_time = 0.0
        total_attack_execution_time = 0.0
        total_feature_extractions = 0  # Count of actual feature extractions
        total_feature_cache_hits = 0   # Count of cache hits
        
        for infer_img_idx, (x_real, c_org, filename) in enumerate(data_loader):
            total_start_time = time.time()
            image_core_time = 0.0
            
            # Accumulators for detailed time per image
            image_generator_time = 0.0
            image_feature_extraction_time = 0.0
            image_feature_cached_time = 0.0
            image_action_selection_time = 0.0
            image_attack_execution_time = 0.0
            image_feature_extractions = 0
            image_feature_cache_hits = 0

            x_real = x_real.to(self.device)
            c_org = c_org.to(self.device) # Move c_org to the device as well

            # Lists to store images for final result visualization
            noattack_result_list = [x_real]
            jpeg_result_list = [x_real]
            opencv_result_list = [x_real]
            median_result_list = [x_real]
            padding_result_list = [x_real]
            transforms_result_list = [x_real]

            for idx, attr_name in enumerate(self.selected_attrs):
                print("=" * 100)
                print(f"Target Attribute ({idx + 1}/{len(self.selected_attrs)}): {attr_name}")

                # 1. Copy the original 13-dimensional attribute vector to prepare the target vector
                c_trg = c_org.clone()

                # 2. Find the index of the current target attribute in the full list of 13 attributes
                try:
                    attr_index = self.attgan_attrs.index(attr_name)
                except ValueError:
                    print(f"Warning: '{attr_name}' is not a default AttGAN attribute. Skipping this attribute.")
                    continue

                # 3. Invert the value of the corresponding attribute (0 -> 1, 1 -> 0)
                c_trg[:, attr_index] = 1 - c_trg[:, attr_index]

                perturbed_image = x_real.clone().detach_() + torch.tensor(np.random.uniform(-self.noise_level, self.noise_level, x_real.shape).astype('float32')).to(self.device)
                # Reset step counter and cached state for each new image/attribute combination
                self.step_counter = 0
                self.cached_state = None
                
                for step in range(self.max_steps_per_episode):
                    # Increment step counter for feature extraction frequency control
                    self.step_counter += 1
                    core_start_time = time.time()
                    
                    # Measure generator forward pass time
                    gen_start_time = time.time()
                    with torch.no_grad():
                        original_gen_image = attgan_model.G(x_real, c_trg, mode='enc-dec')
                        perturbed_gen_image = attgan_model.G(perturbed_image, c_trg, mode='enc-dec')
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    gen_end_time = time.time()
                    step_generator_time = gen_end_time - gen_start_time

                    # Measure feature extraction time
                    feat_start_time = time.time()
                    state, is_cached = self.get_state(perturbed_image, perturbed_gen_image)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    feat_end_time = time.time()
                    step_feature_time = feat_end_time - feat_start_time
                    
                    # Measure action selection time
                    action_start_time = time.time()
                    with torch.no_grad():
                        action = self.rl_agent.select_action(state).item()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    action_end_time = time.time()
                    step_action_time = action_end_time - action_start_time
                    print(f"[Inference] Selected action: {action}")
                    
                    action_history.append(int(action))
                    image_indices.append(infer_img_idx)
                    attr_indices.append(idx)
                    step_indices.append(step)
                    
                    # Measure attack execution time
                    attack_start_time = time.time()
                    if action == 0:
                        perturbed_image, _ = self.attack_func.PGD(perturbed_image, original_gen_image, c_trg)
                        attack_type = "PGD"
                    else:
                        freq_band = ['LOW', 'MID', 'HIGH'][action - 1]
                        perturbed_image, _ = self.attack_func.perturb_frequency_domain(perturbed_image, original_gen_image, c_trg, freq_band=freq_band)
                        attack_type = f"Freq-{freq_band}"
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    attack_end_time = time.time()
                    step_attack_time = attack_end_time - attack_start_time
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    core_end_time = time.time()
                    step_core_time = core_end_time - core_start_time
                    image_core_time += step_core_time
                    
                    # Accumulate detailed times
                    image_generator_time += step_generator_time
                    image_action_selection_time += step_action_time
                    image_attack_execution_time += step_attack_time
                    
                    if is_cached:
                        image_feature_cached_time += step_feature_time
                        image_feature_cache_hits += 1
                    else:
                        image_feature_extraction_time += step_feature_time
                        image_feature_extractions += 1
                    
                    # Print existing core time
                    print(f"[Core Processing Time] Step {step + 1} processing time: {step_core_time:.5f}s")
                    
                    # Print detailed time breakdown
                    cache_status = "cached" if is_cached else "extracted"
                    print(f"  ├─ Generator forward: {step_generator_time:.5f}s")
                    print(f"  ├─ Feature extraction: {step_feature_time:.5f}s ({cache_status})")
                    print(f"  ├─ Action selection: {step_action_time:.5f}s")
                    print(f"  └─ Attack execution: {step_attack_time:.5f}s ({attack_type})")

                analyzed_perturbation_array = analyze_perturbation(perturbed_image - x_real)
                total_perturbation_map += analyzed_perturbation_array

                # [Inference 1] No Transformation (Original)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(perturbed_image - x_real)
                    results["원본(변형없음)"]["total_remain_map"] += remain_perturb_array
                    perturbed_gen_image_orig = attgan_model.G(perturbed_image, c_trg, mode='enc-dec')
                    noattack_result_list.append(perturbed_image)
                    noattack_result_list.append(original_gen_image)
                    noattack_result_list.append(perturbed_gen_image_orig)
                    results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_orig, "원본(변형없음)", results)

                # [Inference 2] JPEG Compression
                x_adv_jpeg = compress_jpeg(perturbed_image, quality=75)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_jpeg - x_real)
                    results["JPEG압축"]["total_remain_map"] += remain_perturb_array
                    perturbed_gen_image_jpeg = attgan_model.G(x_adv_jpeg, c_trg, mode='enc-dec')
                    jpeg_result_list.append(x_adv_jpeg)
                    jpeg_result_list.append(original_gen_image)
                    jpeg_result_list.append(perturbed_gen_image_jpeg)
                    results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_jpeg, "JPEG압축", results)

                # [Inference 3] OpenCV Denoising
                x_adv_denoise_opencv = denoise_opencv(perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_denoise_opencv - x_real)
                    results["OpenCV디노이즈"]["total_remain_map"] += remain_perturb_array
                    perturbed_gen_image_opencv = attgan_model.G(x_adv_denoise_opencv, c_trg, mode='enc-dec')
                    opencv_result_list.append(x_adv_denoise_opencv)
                    opencv_result_list.append(original_gen_image)
                    opencv_result_list.append(perturbed_gen_image_opencv)
                    results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_opencv, "OpenCV디노이즈", results)

                # [Inference 4] Median Smoothing
                x_adv_median = denoise_scikit(perturbed_image)
                with torch.no_grad(): 
                    remain_perturb_array = analyze_perturbation(x_adv_median - x_real)
                    results["중간값스무딩"]["total_remain_map"] += remain_perturb_array
                    perturbed_gen_image_median = attgan_model.G(x_adv_median, c_trg, mode='enc-dec')
                    median_result_list.append(x_adv_median)
                    median_result_list.append(original_gen_image)
                    median_result_list.append(perturbed_gen_image_median)
                    results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_median, "중간값스무딩", results)

                # [Inference 5] Random Resize and Padding
                x_real_padding, x_adv_padding = random_resize_padding(x_real, perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_padding - x_real_padding)
                    results["크기조정패딩"]["total_remain_map"] += remain_perturb_array
                    original_gen_image_padding = attgan_model.G(x_real_padding, c_trg, mode='enc-dec')
                    perturbed_gen_image_padding = attgan_model.G(x_adv_padding, c_trg, mode='enc-dec')
                    padding_result_list.append(x_adv_padding)
                    padding_result_list.append(original_gen_image_padding)
                    padding_result_list.append(perturbed_gen_image_padding)
                    results = calculate_and_save_metrics(original_gen_image_padding, perturbed_gen_image_padding, "크기조정패딩", results)

                # [Inference 6] Random Image Transformations
                x_real_transforms, x_adv_transforms = random_image_transforms(x_real, perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_transforms - x_real_transforms)
                    results["이미지변환"]["total_remain_map"] += remain_perturb_array
                    original_gen_image_transforms = attgan_model.G(x_real_transforms, c_trg, mode='enc-dec')
                    perturbed_gen_image_transforms = attgan_model.G(x_adv_transforms, c_trg, mode='enc-dec')
                    transforms_result_list.append(x_adv_transforms)
                    transforms_result_list.append(original_gen_image_transforms)
                    transforms_result_list.append(perturbed_gen_image_transforms)
                    results = calculate_and_save_metrics(original_gen_image_transforms, perturbed_gen_image_transforms, "이미지변환", results)
                
                with torch.no_grad():
                    x_real_np = x_real.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    perturbed_image_np = perturbed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    invisible_lpips_value = self.lpips_loss(x_real, perturbed_image).mean()
                    invisible_psnr_value = psnr(x_real_np, perturbed_image_np, data_range=1.0)
                    invisible_ssim_value = ssim(x_real_np, perturbed_image_np, data_range=1.0, win_size=3, channel_axis=2)

                    total_invisible_lpips += invisible_lpips_value
                    total_invisible_psnr += invisible_psnr_value
                    total_invisible_ssim += invisible_ssim_value

                    episode += 1
                
            all_result_lists = [noattack_result_list, jpeg_result_list, opencv_result_list, median_result_list, padding_result_list, transforms_result_list]
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
            print(f"[Inference] Result saved: {result_path}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_end_time = time.time()
            total_elapsed_time = total_end_time - total_start_time
            
            total_core_time += image_core_time
            total_processing_time += total_elapsed_time
            total_generator_time += image_generator_time
            total_feature_extraction_time += image_feature_extraction_time
            total_feature_cached_time += image_feature_cached_time
            total_action_selection_time += image_action_selection_time
            total_attack_execution_time += image_attack_execution_time
            total_feature_extractions += image_feature_extractions
            total_feature_cache_hits += image_feature_cache_hits
            
            print(f"[Core Processing Time] Image {infer_img_idx + 1} core processing time: {image_core_time:.5f}s")
            print(f"[Total Processing Time] Image {infer_img_idx + 1} total processing time: {total_elapsed_time:.5f}s (for reference)")
            
            # Print detailed time summary per image
            print(f"\n[Detailed Time Summary] Image {infer_img_idx + 1}:")
            print(f"  Generator forward:     {image_generator_time:.5f}s ({image_generator_time/image_core_time*100:.1f}%)")
            print(f"  Feature extraction:    {image_feature_extraction_time:.5f}s ({image_feature_extractions} extractions)")
            print(f"  Feature cached:        {image_feature_cached_time:.5f}s ({image_feature_cache_hits} cache hits)")
            total_feature_time = image_feature_extraction_time + image_feature_cached_time
            print(f"  Total feature time:    {total_feature_time:.5f}s ({total_feature_time/image_core_time*100:.1f}%)")
            print(f"  Action selection:      {image_action_selection_time:.5f}s ({image_action_selection_time/image_core_time*100:.1f}%)")
            print(f"  Attack execution:      {image_attack_execution_time:.5f}s ({image_attack_execution_time/image_core_time*100:.1f}%)")
            if image_feature_extractions + image_feature_cache_hits > 0:
                cache_hit_rate = image_feature_cache_hits / (image_feature_extractions + image_feature_cache_hits) * 100
                print(f"  Feature cache hit rate: {cache_hit_rate:.1f}%")
            print()

            if infer_img_idx >= (self.inference_image_num - 1):
                break

        score = print_comprehensive_metrics(results, episode, total_invisible_psnr, total_invisible_ssim, total_invisible_lpips)
        visualize_actions(action_history, image_indices, attr_indices, step_indices)

        # Print final time summary (existing)
        print(f"\n{'='*80}")
        print(f"[Inference Complete] Total core processing time: {total_core_time:.5f}s")
        print(f"[Inference Complete] Total processing time (for reference): {total_processing_time:.5f}s")
        print(f"[Inference Complete] Average core processing time: {total_core_time / episode:.5f}s (Total {episode} processed)")
        print(f"[Inference Complete] Average total processing time (for reference): {total_processing_time / episode:.5f}s (Total {episode} processed)")
        
        # Print detailed time statistics
        print(f"\n{'='*80}")
        print(f"[Detailed Time Statistics] Total Inference Summary")
        print(f"{'='*80}")
        print(f"\n1. Total Time Breakdown:")
        print(f"   Generator forward:     {total_generator_time:.5f}s ({total_generator_time/total_core_time*100:.1f}%)")
        print(f"   Feature extraction:    {total_feature_extraction_time:.5f}s ({total_feature_extractions} extractions)")
        print(f"   Feature cached:        {total_feature_cached_time:.5f}s ({total_feature_cache_hits} cache hits)")
        total_all_feature_time = total_feature_extraction_time + total_feature_cached_time
        print(f"   Total feature time:    {total_all_feature_time:.5f}s ({total_all_feature_time/total_core_time*100:.1f}%)")
        print(f"   Action selection:      {total_action_selection_time:.5f}s ({total_action_selection_time/total_core_time*100:.1f}%)")
        print(f"   Attack execution:      {total_attack_execution_time:.5f}s ({total_attack_execution_time/total_core_time*100:.1f}%)")
        
        print(f"\n2. Average Time per Image:")
        print(f"   Generator forward:     {total_generator_time/episode:.5f}s")
        print(f"   Feature extraction:    {total_feature_extraction_time/episode:.5f}s")
        print(f"   Feature cached:        {total_feature_cached_time/episode:.5f}s")
        print(f"   Total feature time:    {total_all_feature_time/episode:.5f}s")
        print(f"   Action selection:      {total_action_selection_time/episode:.5f}s")
        print(f"   Attack execution:      {total_attack_execution_time/episode:.5f}s")
        
        print(f"\n3. Feature Extraction Statistics:")
        print(f"   Total feature calls:   {total_feature_extractions + total_feature_cache_hits}")
        print(f"   Actual extractions:    {total_feature_extractions}")
        print(f"   Cache hits:            {total_feature_cache_hits}")
        if total_feature_extractions + total_feature_cache_hits > 0:
            overall_cache_hit_rate = total_feature_cache_hits / (total_feature_extractions + total_feature_cache_hits) * 100
            print(f"   Cache hit rate:        {overall_cache_hit_rate:.1f}%")
            if total_feature_extractions > 0:
                avg_extraction_time = total_feature_extraction_time / total_feature_extractions
                print(f"   Avg extraction time:   {avg_extraction_time:.5f}s")
            if total_feature_cache_hits > 0:
                avg_cached_time = total_feature_cached_time / total_feature_cache_hits
                print(f"   Avg cached time:       {avg_cached_time:.5f}s")
                if total_feature_extractions > 0:
                    speedup = avg_extraction_time / avg_cached_time
                    print(f"   Cache speedup:         {speedup:.2f}x")
        
        print(f"\n4. Configuration:")
        print(f"   Feature extractor:     {self.feature_extractor_name}")
        print(f"   Extractor frequency:   {self.feature_extractor_frequency}")
        print(f"   Max steps per episode: {self.max_steps_per_episode}")
        print(f"   Number of images:      {episode}")
        print(f"{'='*80}\n")

        return score

    def load_rainbow_dqn_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"[INFO] Checkpoint not found at {checkpoint_path}. Starting from scratch.")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.rl_agent.dqn.load_state_dict(checkpoint['rainbow_dqn_state_dict'])
        self.rl_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[INFO] Rainbow DQN model loaded successfully: {checkpoint_path}")


    def build_model(self):
        """Create Generator and Discriminator"""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.G.to(self.device)
        self.D.to(self.device)


    """Build RLAB agent components (Rainbow DQN, Prioritized Replay Buffer, Feature Extractor)."""
    def build_rlab_agent(self):
        # Select and load Feature Extractor
        if self.feature_extractor_name == "vgg19":
            self.feature_extractor = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval() # VGG19 features
        elif self.feature_extractor_name == "resnet50":
            extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(extractor.children())[:-2]).to(device).eval()
        elif self.feature_extractor_name == "mesonet":
            # MesoNet is replaced by GhostFaceNet
            raise ValueError("MesoNet is deprecated. Use 'ghostfacenets' instead.")
        elif self.feature_extractor_name == "ghostfacenets":
            # GhostFaceNetsV2 모델을 임베딩 추출 모드로 불러옵니다.
            # 입력 이미지 크기는 256x256이며, width=1, dropout=0.으로 설정합니다.
            model = GhostFaceNetsV2(image_size=256, width=1, dropout=0.)
            self.feature_extractor = model.to(device).eval()
        elif self.feature_extractor_name == "edgeface":
            # EdgeFace 모델을 torch.hub에서 불러옵니다.
            self.feature_extractor = torch.hub.load(
                'otroshi/edgeface', 
                'edgeface_xs_gamma_06',
                source='github', 
                pretrained=True,
                trust_repo="check"
            ).to(device).eval()
        else:
            raise ValueError("Invalid FEATURE_EXTRACTOR_NAME")

        # Prioritized Replay Buffer (PER) : N-step Replay Buffer
        self.memory = PrioritizedReplayBuffer(capacity=self.memory_capacity, alpha=self.alpha, beta_start=self.beta_start, beta_frames=self.beta_frames, n_step=self.n_step, gamma=self.gamma)

        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        self.Transition = Transition # Use Transition in the SolverRainbow class

        # Configure Rainbow DQN Agent
        if (self.feature_extractor_name == "vgg19"):
            state_dim = 65536
        elif (self.feature_extractor_name == "resnet50"):
            state_dim = 262144
        elif (self.feature_extractor_name == "mesonet"):
            state_dim = 64
        elif (self.feature_extractor_name == "ghostfacenets"):
            state_dim = 1024 # 512 * 2 images
        elif (self.feature_extractor_name == "edgeface"):
            state_dim = 1024 # 512 * 2 images
        else:
            # default: ghostfacenets or edgeface = 512 * 2
            state_dim = 1024

        # Variable to be used in select_action()
        initial_ratio_steps = self.max_steps_per_episode * self.training_image_num * 5 // 10

        action_dim = self.action_dim # Action Dimension
        self.rl_agent = RainbowDQNAgent(state_dim, action_dim, self.agent_lr, self.gamma, self.epsilon_start, self.epsilon_end, self.epsilon_decay, self.target_update_interval,
                                        v_min=self.v_min, v_max=self.v_max, atom_size=self.atom_size, # Categorical DQN parameters
                                        beta_start=self.beta_start, beta_frames=self.beta_frames, prior_eps=self.prior_eps, # PER parameters
                                        n_step=self.n_step, initial_ratio_steps=initial_ratio_steps) # N-step Learning parameters
        self.rl_agent.dqn.to(self.device) # policy_net -> dqn
        self.rl_agent.dqn_target.to(self.device) # target_net -> dqn_target

    def restore_attgan_model(self, checkpoint_path):
        """Load AttGAN model from a specific checkpoint path"""
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] AttGAN checkpoint not found: {checkpoint_path}")
            print(f"[INFO] Using randomly initialized weights.")
            return
        
        print(f"[INFO] Loading AttGAN checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        if isinstance(checkpoint, dict) and 'G' in checkpoint:
            # AttGAN format: {'G': state_dict, 'D': state_dict (optional)}
            self.G.load_state_dict(checkpoint['G'], strict=False)
            print(f"[INFO] Successfully loaded Generator from AttGAN checkpoint")
            
            if 'D' in checkpoint:
                self.D.load_state_dict(checkpoint['D'], strict=False)
                print(f"[INFO] Successfully loaded Discriminator from AttGAN checkpoint")
            else:
                print(f"[INFO] Discriminator not found in checkpoint, using randomly initialized weights")
        else:
            print(f"[WARNING] Unexpected checkpoint format in {checkpoint_path}")
            print(f"[INFO] Using randomly initialized weights.")
    
    def restore_model(self, resume_iters):
        """Reset the weights of the trained generator and discriminator"""
        print('Loading the trained models from step {}...'.format(resume_iters))
        
        # Try multiple checkpoint formats
        # Format 1: Original AttGAN format (weights.{iter}.pth with 'G' key)
        attgan_pth_path = os.path.join(self.model_save_dir, 'weights.{}.pth'.format(resume_iters))
        # Format 2: Separate .ckpt files
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        # Try loading AttGAN .pth format first
        if os.path.exists(attgan_pth_path):
            print(f"[INFO] Loading AttGAN checkpoint from: {attgan_pth_path}")
            checkpoint = torch.load(attgan_pth_path, map_location=lambda storage, loc: storage)
            
            if isinstance(checkpoint, dict) and 'G' in checkpoint:
                # AttGAN format: {'G': state_dict, 'D': state_dict (optional)}
                self.G.load_state_dict(checkpoint['G'], strict=False)
                print(f"[INFO] Successfully loaded Generator from AttGAN checkpoint")
                
                if 'D' in checkpoint:
                    self.D.load_state_dict(checkpoint['D'], strict=False)
                    print(f"[INFO] Successfully loaded Discriminator from AttGAN checkpoint")
                else:
                    print(f"[INFO] Discriminator not found in checkpoint, using randomly initialized weights")
                return
            else:
                print(f"[WARNING] Unexpected checkpoint format in {attgan_pth_path}")
        
        # Fall back to separate .ckpt files
        if not os.path.exists(G_path):
            print(f"[WARNING] Generator checkpoint not found: {G_path}")
            print(f"[INFO] Skipping model loading. Using randomly initialized weights.")
            return
        
        if not os.path.exists(D_path):
            print(f"[WARNING] Discriminator checkpoint not found: {D_path}")
            print(f"[INFO] Loading only Generator weights.")
            self.load_model_weights(self.G, G_path)
            return

        self.load_model_weights(self.G, G_path)
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        print(f"[INFO] Successfully loaded Generator and Discriminator from step {resume_iters}")

    def load_model_weights(self, model, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict, strict=False)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset=None, selected_attrs=None): # dataset='CelebA' -> None
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA' or 'MAADFace':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA' or 'MAADFace':
                c_trg = c_org.clone()
                if i in hair_color_indices: # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0) # Reverse attribute value.

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    # """
    #     Calculates the reward.
    #     LPIPS: A value between [0, 1]. The lower the value, the more similar the two images are. 
    #            Generally, an LPIPS value greater than 0.5 is considered to indicate a human-observable difference between two images.
    # """
    # def calculate_reward(self, original_gen_image, perturbed_gen_image, x_real, perturbed_image, c_trg):
    #     # 1. Deepfake Defense Reward
    #     # Use the difference between StarGAN generated images (L1, L2, LPIPS) as a reward.
    #     # Randomly apply "image transformations" to original_gen_image and perturbed_gen_image each time, and calculate the difference between them.
    #     transformed_x_real, transformed_perturbed_image = apply_random_transform(x_real, perturbed_image)
    #     transformed_original = attgan_model.G(transformed_x_real, c_trg, mode='enc-dec')
    #     transformed_perturbed = attgan_model.G(transformed_perturbed_image, c_trg, mode='enc-dec')


    #     defense_l1_loss = F.l1_loss(transformed_original, transformed_perturbed)
    #     defense_l2_loss = F.mse_loss(transformed_original, transformed_perturbed)

    #     defense_lpips = self.lpips_loss(transformed_original, transformed_perturbed).mean()

    #     # Scale and apply based on L1 Error = [0, 131,072] / L2 Error = [0, 512] / LPIPS = [0, 1]
    #     reward_defense = ((defense_l1_loss / 10) + (defense_l2_loss / 5) + defense_lpips) * 5

    #     # 2. Noise Invisibility Reward
    #     # Use the similarity between the noisy image and the original image as a reward (PSNR, SSIM, LPIPS).
    #     x_real_np = x_real.squeeze().cpu().numpy()
    #     perturbed_image_np = perturbed_image.squeeze().cpu().numpy()

    #     if np.array_equal(x_real_np, perturbed_image_np):
    #         invisibility_psnr = 100.0 # Assign max PSNR if the two images are identical
    #     else:
    #         invisibility_psnr = psnr(x_real.squeeze().cpu().numpy(), perturbed_image.squeeze().cpu().numpy(), data_range=1.0)

    #     invisibility_ssim = ssim(x_real.squeeze().cpu().numpy(), perturbed_image.squeeze().cpu().numpy(), data_range=1.0, win_size=3, channel_axis=0, multichannel=True) # Specify channel axis

    #     invisibility_lpips = self.lpips_loss(perturbed_image, x_real).mean()

    #     # Scale and apply based on PSNR = [0, 100] / SSIM = [-1, 1] / LPIPS = [0, 1]
    #     reward_invisibility = (0.01 * invisibility_psnr) + invisibility_ssim + (1 - invisibility_lpips)

    #     # Final Reward Combination (Since the two rewards are in a trade-off relationship, adjust weights while keeping the sum fixed at 1.0)
    #     w_defense = self.reward_weight
    #     w_invisibility = 1.0 - w_defense
    #     total_reward = w_defense * reward_defense + w_invisibility * reward_invisibility

    #     return total_reward, defense_l1_loss, defense_l2_loss, defense_lpips, invisibility_ssim, invisibility_psnr, invisibility_lpips


    def calculate_reward(self, original_gen_image, perturbed_gen_image, x_real, perturbed_image, c_trg):
        ### a. Effectiveness
        # Randomly apply "image transformations" to original_gen_image and perturbed_gen_image each time, and calculate the difference between them.
        transformed_x_real, transformed_perturbed_image = apply_random_transform(x_real, perturbed_image)
        transformed_original, _ = self.G(transformed_x_real, c_trg)
        transformed_perturbed, _ = self.G(transformed_perturbed_image, c_trg)
        defense_l1_loss = F.l1_loss(transformed_original, transformed_perturbed)
        defense_l2_loss = F.mse_loss(transformed_original, transformed_perturbed)
        defense_lpips = self.lpips_loss(transformed_original, transformed_perturbed).mean()

        ### b. Imperceptibility
        # Use the similarity between the noisy image and the original image as a reward (PSNR, SSIM, LPIPS).
        x_real_np = x_real.squeeze().cpu().numpy()
        perturbed_image_np = perturbed_image.squeeze().cpu().numpy()
        if np.array_equal(x_real_np, perturbed_image_np):
            invisibility_psnr = 100.0 # If the two images are identical, assign the maximum PSNR value.
        else:
            invisibility_psnr = psnr(x_real.squeeze().cpu().numpy(), perturbed_image.squeeze().cpu().numpy(), data_range=1.0)
        invisibility_ssim = ssim(x_real.squeeze().cpu().numpy(), perturbed_image.squeeze().cpu().numpy(), data_range=1.0, win_size=3, channel_axis=0, multichannel=True) # Specify channel axis.
        invisibility_lpips = self.lpips_loss(perturbed_image, x_real).mean()
 
        # parameter setting
        tau_eff = 0.05   # effectiveness threshold
        tau_ssim, m_ssim = 0.96, 0.02   # SSIM margin
        m_eff = 0.03                    # L2 margin 
        lam = 0.7                       # SSIM penalty
        eta = 0.6                       # effectiveness weight (0~1)

        def clip01(x): 
            return max(0.0, min(1.0, x))

        # 1) effectiveness hinge (0→1 saturate)
        g_l2   = clip01((float(defense_l2_loss) - float(tau_eff)) / m_eff)

        # 2) imperceptibility hinge (0→1 saturate)
        g_ssim = clip01((float(invisibility_ssim) - tau_ssim) / m_ssim)

        # 3) SSIM penalty (0→1 saturate)
        pen_ssim = clip01((tau_ssim - float(invisibility_ssim)) / m_ssim)

        # effectiveness more weighting
        base = g_l2 * (eta + (1.0 - eta) * g_ssim)
        total_reward = base - lam * pen_ssim
        # total_reward = max(-1.0, min(1.0, total_reward))  # optionally clip the reward to [-1, 1]
        
        print(f"[total Reward] L2={defense_l2_loss:.4f}, lpips={invisibility_lpips:.4f}, ssim={invisibility_ssim}, total_reward={total_reward}")
        return total_reward, defense_l1_loss, defense_l2_loss, defense_lpips, invisibility_ssim, invisibility_psnr, invisibility_lpips




    """
        Extracts features from images using VGG-19/ResNet50/MesoNet and returns a state vector.
        
        Args:
            x_real: Original image, shape=[1, 3, 256, 256], range=[-1, 1]
            perturbed_image: Image with added noise, shape=[1, 3, 256, 256], range=[-1, 1]
            original_gen_image: Image generated from the original image, shape=[1, 3, 256, 256], range=[-1, 1]
            perturbed_gen_image: Transformed generated image, shape=[1, 3, 256, 256], range=[-1, 1]

        Returns:
            A state vector combining the features of the four images.
        
        1. Using VGG-19 model
        Output dimension: 256x256 input image -> [1, 512, 8, 8] feature map -> [1, 32768] flattened
        Final output when combining 2 images: [1, 65536]

        2. Using ResNet50 model
        Output dimension: 256x256 input image -> [1, 2048, 8, 8] feature map -> [1, 131072] flattened
        Final output when combining 2 images: [1, 262144]

        3. Using MesoNet model
        Final output when combining 2 images: [1, 64]

        4. Using GhostFaceNets model
        Output dimension: 256x256 input image -> [1, 512] embedding vector
        Final output when combining 2 images: [1, 1024]
    """
    def get_state(self, perturbed_image, perturbed_gen_image, force_extract=False):
        # Check if we should extract features based on frequency
        should_extract = force_extract or (self.step_counter % self.feature_extractor_frequency == 0)
        
        # If we don't need to extract and have cached state, return cached
        if not should_extract and self.cached_state is not None:
            return self.cached_state, True  # Return (state, is_cached=True)
        
        # Otherwise, extract features and cache them
        # Convert input image range from [-1, 1] to [0, 1]
        perturbed_image_norm = (perturbed_image + 1) / 2
        perturbed_gen_image_norm = (perturbed_gen_image + 1) / 2
        
        # Normalize with ImageNet mean and standard deviation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
        perturbed_image_norm = normalize(perturbed_image_norm)
        perturbed_gen_image_norm = normalize(perturbed_gen_image_norm)

        # Extract features from images
        with torch.no_grad():
            perturbed_features = self.feature_extractor(perturbed_image_norm)
            perturbed_gen_features = self.feature_extractor(perturbed_gen_image_norm)
        
        # Flatten the features if they are not already 1D (e.g., from VGG/ResNet)
        if perturbed_features.dim() > 2:
            perturbed_features = perturbed_features.view(perturbed_features.size(0), -1)
            perturbed_gen_features = perturbed_gen_features.view(perturbed_gen_features.size(0), -1)

        # Concatenate the features to form the state vector
        state = torch.cat((perturbed_features, perturbed_gen_features), dim=1)
        
        # Cache the extracted features
        self.cached_state = state
        
        return state, False  # Return (state, is_cached=False)




    """Performs the RLAB attack (Rainbow DQN Agent)"""
    def train_attack(self):
        torch.cuda.empty_cache()  # Free unused GPU memory
        torch.autograd.set_detect_anomaly(True) # Enable anomaly detection
        gc.collect()

        # List of 13 attributes used in AttGAN training (declare as a class member or global variable)
        self.attgan_attrs = [
            'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
            'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
        ]

        total_perturbation_map = np.zeros((256, 256)) # Initialize NumPy array to accumulate perturbation values
        total_remain_map = np.zeros((256, 256)) # Initialize NumPy array to accumulate remaining perturbation values after image transformation

        # Load model checkpoint
        checkpoint_path = os.path.join(self.model_save_dir, f'final_rainbow_dqn.pth')            
        self.load_rainbow_dqn_checkpoint(checkpoint_path)


        # Load the trained generator
        self.restore_model(self.test_iters)

        # Set data loader
        data_loader = self.dataset_loader

        # Initialize a dictionary to store results for each image transformation method
        results = {
            "Original(No Transform)": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                        "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "JPEG Compression": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "OpenCV Denoise": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                        "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "Median Smoothing": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                        "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "Resize and Pad": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "Image Transform": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0, 
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))}
        }

        # Initialize Metrics
        total_invisible_psnr, total_invisible_ssim, total_invisible_lpips = 0.0, 0.0, 0.0
        episode = 0
        frame_idx = 0 # frame_idx for beta annealing (PER)

        # Add variables for recording and visualizing action selections
        action_history = []
        image_indices = []
        attr_indices = []
        step_indices = []
        reward_per_episode = [] # To get the reward trend plot


        # c_org.shape is now [1, 13] -> because it always has all 13 attribute values
        for test_img_idx, (x_real, c_org, filename) in enumerate(data_loader):
            print('\n'*3)
            print(f"Image index: {test_img_idx+1}-th image")
            print(f"Processing image filename={filename}")

            # Prepare input images and target domain labels
            x_real = x_real.to(self.device)
            
            # c_org must be of shape (batch_size, 13)
            c_org = c_org.to(self.device) 

            # Create an instance of the class containing the attack function that inserts perturbations
            attack_func = attgan_attacks.AttackFunction(config=self.config, model=attgan_model.G, device=self.device)

            # Lists to save the final results as images
            noattack_result_list = [x_real]
            jpeg_result_list = [x_real]
            opencv_result_list = [x_real]
            median_result_list = [x_real]
            padding_result_list = [x_real]
            transforms_result_list = [x_real]

            # self.selected_attrs is a list of 5 attribute names as before.
            # e.g., ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
            for idx, attr_name  in enumerate(self.selected_attrs):
                print("=" * 100)
                # Generate c_trg one by one in each loop, instead of creating a c_trg_list beforehand.
                print(f"Target Attribute ({idx + 1}/{len(self.selected_attrs)}): {attr_name}")

                # 1. Copy the original 13-dimensional attribute vector to prepare for creating the target vector
                c_trg = c_org.clone()

                # 2. Find the index of the current target attribute in the full list of 13 attributes
                try:
                    attr_index = self.attgan_attrs.index(attr_name)
                except ValueError:
                    print(f"Warning: '{attr_name}' is not a default attribute of AttGAN. Skipping this attribute.")
                    continue

                # 3. Invert the value of that attribute (0 -> 1, 1 -> 0)
                # CelebA attributes have values of 0 or 1.
                c_trg[:, attr_index] = 1 - c_trg[:, attr_index]

                # Now c_trg has the shape (batch_size, 13), with only the target attribute changed.

                # Initialize the "perturbed image" by adding noise from a uniform distribution
                perturbed_image = x_real.clone().detach_() + torch.tensor(np.random.uniform(-self.noise_level, self.noise_level, x_real.shape).astype('float32')).to(self.device)

                # Initialize the "perturbed image" by adding noise from a normal (Gaussian) distribution
                # perturbed_image = x_real.clone().detach_() + torch.tensor(np.random.normal(0, 0.01, x_real.shape).astype('float32')).to(self.device)

                # perturbed_image = x_real.clone().detach_() + (torch.randn(x_real.shape) * 0.01).to(self.device)

                # Generate the original StarGAN image (attack target)
                with torch.no_grad():
                    original_gen_image = attgan_model.G(x_real, c_trg, mode='enc-dec')
                    perturbed_gen_image = attgan_model.G(perturbed_image, c_trg, mode='enc-dec') # Initialize perturbed generated image

                n_step_buffer_test_attack = deque(maxlen=self.n_step) # N-step buffer for the test_attack episode

                total_reward_this_episode = 0 # To get the reward trend plot

                for step in range(self.max_steps_per_episode): # Repeat for max_steps_per_episode
                    frame_idx += 1 # Increment frame_idx (for PER beta annealing)

                    # 1. Calculate State
                    state, _ = self.get_state(perturbed_image, perturbed_gen_image)  # Ignore cache status in training

                    # 2. Select Action (Rainbow DQN Agent)
                    action = self.rl_agent.select_action(state) # NoisyNet does not use Epsilon-greedy

                    # Record the action
                    action_history.append(action.item())
                    image_indices.append(test_img_idx)
                    attr_indices.append(idx)
                    step_indices.append(step)

                    # 3. Apply Perturbation
                    # Action types: "Insert noise once in the spatial domain using FGSM", "Insert noise once in the low-frequency band of the frequency domain", "Insert noise once in the mid-frequency band of the frequency domain", "Insert noise once in the high-frequency band of the frequency domain"
                    if action == 0:
                        # Insert noise in the spatial domain using FGSM
                        perturbed_image, _ = attack_func.PGD(perturbed_image, original_gen_image, c_trg)
                        # print("Action selected this step: PGD-style Space Domain Noise")
                    elif action == 1:
                        # Insert noise in the low-frequency band of the frequency domain
                        perturbed_image, _ = attack_func.perturb_frequency_domain(perturbed_image, original_gen_image, c_trg, freq_band='LOW')
                        # print("Action selected this step: Low-frequency Domain Noise")
                    elif action == 2:
                        # Insert noise in the mid-frequency band of the frequency domain
                        perturbed_image, _ = attack_func.perturb_frequency_domain(perturbed_image, original_gen_image, c_trg, freq_band='MID')
                        # print("Action selected this step: Mid-frequency Domain Noise")
                    elif action == 3:
                        # Insert noise in the high-frequency band of the frequency domain
                        perturbed_image, _ = attack_func.perturb_frequency_domain(perturbed_image, original_gen_image, c_trg, freq_band='HIGH')
                        # print("Action selected this step: High-frequency Domain Noise")
                    else:
                        raise ValueError("Invalid action index")


                    # 4. Generate face-transformed deepfake image with StarGAN (perturbed image)
                    with torch.no_grad():
                        perturbed_gen_image = attgan_model.G(perturbed_image, c_trg, mode='enc-dec')


                    # 5. Calculate Reward
                    reward, defense_l1_loss, defense_l2_loss, defense_lpips, invisibility_ssim, invisibility_psnr, invisibility_lpips = self.calculate_reward(original_gen_image, perturbed_gen_image, x_real, perturbed_image, c_trg)

                    # neptune logging
                    if self.run is not None:
                        self.run["train/reward"].append(float(reward))
                        self.run["train/L1"].append(float(defense_l1_loss))
                        self.run["train/L2"].append(float(defense_l2_loss))
                        self.run["train/LPIPS"].append(float(invisibility_lpips))
                        self.run["train/SSIM"].append(float(invisibility_ssim))

                    # To get the reward trend plot
                    if isinstance(reward, torch.Tensor):
                        total_reward_this_episode += reward.item()
                    else:
                        total_reward_this_episode += reward      


                    reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device) # Convert reward to a tensor

                    # print(f"[DEBUG] Reward calculated in this step {step+1}: {reward.item()}, Deepfake Defense L1 Loss: {defense_l1_loss.item()}, Deepfake Defense L2 Loss: {defense_l2_loss:.5f}, Deepfake Defense LPIPS: {defense_lpips.item()}")

                    # print(f"[DEBUG] Invisibility PSNR: {invisibility_psnr:.5f}, Invisibility SSIM: {invisibility_ssim:.5f}, Invisibility LPIPS: {invisibility_lpips.item()}")


                    # 6. Calculate Next State (same as current state, or the state of the next step)
                    next_state, _ = self.get_state(perturbed_image, perturbed_gen_image)  # Ignore cache status in training


                    # N-step buffer for test_attack episode
                    # Check if the done value is a boolean (here it's not an episode, so False)
                    n_step_buffer_test_attack.append((state, action, reward_tensor, next_state, torch.tensor([False]))) # done flag (tensor) for N-step transition

                    # 7. Create and store N-step transition
                    if len(n_step_buffer_test_attack) == self.n_step: # When an N-step transition is complete, store it in the Replay Buffer
                        state_n_step, action_n_step, reward_n_step, next_state_n_step, done_n_step = self._get_n_step_transition(n_step_buffer_test_attack) # Create N-step transition
                        self.memory.push(state_n_step, action_n_step, reward_n_step.unsqueeze(0), next_state_n_step, done_n_step) # Store N-step transition in Replay Buffer

                    # 8. Update Rainbow DQN (when there are 5 or more items in Replay Memory)
                    if len(self.memory) >= 5:
                        # print(f"[DEBUG] Update Rainbow DQN with {self.batch_size} samples")
                        batch, weights, indices = self.memory.sample(self.batch_size, frame_idx) # PER sample batch, pass frame_idx (for beta annealing)
                        loss_val, priorities = self.rl_agent.update_model(batch, weights, indices, self.batch_size, frame_idx)
                        self.memory.update_priorities(indices, priorities) # PER priority update

                    # NoisyNet Reset Noise
                    self.rl_agent.reset_noise() # Reset noise at every step
                
                reward_per_episode.append(total_reward_this_episode)  # To get the reward trend plot - save after one episode ends

                # Function defined in util.py. Prints various debug logs.
                # print_debug(x_real, perturbed_image, original_gen_image, perturbed_gen_image)

                # Accumulate to understand the average amount of noise inserted into one image.
                analyzed_perturbation_array = analyze_perturbation(perturbed_image - x_real)
                total_perturbation_map += analyzed_perturbation_array

                # [Test 1] No transform (original)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(perturbed_image - x_real)
                    results["Original(No Transform)"]["total_remain_map"] += remain_perturb_array

                    perturbed_gen_image_orig = attgan_model.G(perturbed_image, c_trg, mode='enc-dec') # AttGAN version

                    noattack_result_list.append(perturbed_image)
                    noattack_result_list.append(original_gen_image)
                    noattack_result_list.append(perturbed_gen_image_orig)

                    results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_orig, "Original(No Transform)", results)

                # [Test 2] JPEG Compression
                x_adv_jpeg = compress_jpeg(perturbed_image, quality=75)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_jpeg - x_real)
                    results["JPEG Compression"]["total_remain_map"] += remain_perturb_array


                    perturbed_gen_image_jpeg = attgan_model.G(x_adv_jpeg, c_trg, mode='enc-dec') # AttGAN version

                    jpeg_result_list.append(x_adv_jpeg)
                    jpeg_result_list.append(original_gen_image)
                    jpeg_result_list.append(perturbed_gen_image_jpeg)

                    results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_jpeg, "JPEG Compression", results)

                # [Test 3] OpenCV Denoise
                x_adv_denoise_opencv = denoise_opencv(perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_denoise_opencv - x_real)
                    results["OpenCV Denoise"]["total_remain_map"] += remain_perturb_array


                    perturbed_gen_image_opencv = attgan_model.G(x_adv_denoise_opencv, c_trg, mode='enc-dec') # AttGAN version

                    opencv_result_list.append(x_adv_denoise_opencv)
                    opencv_result_list.append(original_gen_image)
                    opencv_result_list.append(perturbed_gen_image_opencv)

                    results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_opencv, "OpenCV Denoise", results)

                # [Test 4] Median Smoothing
                x_adv_median = denoise_scikit(perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_median - x_real)
                    results["Median Smoothing"]["total_remain_map"] += remain_perturb_array

                    perturbed_gen_image_median = attgan_model.G(x_adv_median, c_trg, mode='enc-dec') # AttGAN version

                    median_result_list.append(x_adv_median)
                    median_result_list.append(original_gen_image)
                    median_result_list.append(perturbed_gen_image_median)

                    results = calculate_and_save_metrics(original_gen_image, perturbed_gen_image_median, "Median Smoothing", results)

                # [Test 5] Random Resize and Padding
                x_real_padding, x_adv_padding = random_resize_padding(x_real, perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_padding - x_real_padding)
                    results["Resize and Pad"]["total_remain_map"] += remain_perturb_array


                    original_gen_image_padding = attgan_model.G(x_real_padding, c_trg, mode='enc-dec') # AttGAN version
                    perturbed_gen_image_padding = attgan_model.G(x_adv_padding, c_trg, mode='enc-dec') # AttGAN version

                    padding_result_list.append(x_adv_padding)
                    padding_result_list.append(original_gen_image_padding)
                    padding_result_list.append(perturbed_gen_image_padding)

                    results = calculate_and_save_metrics(original_gen_image_padding, perturbed_gen_image_padding, "Resize and Pad", results)

                # [Test 6] Random Image Transforms -> shear, shift, zoom, rotation
                x_real_transforms, x_adv_transforms = random_image_transforms(x_real, perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_transforms - x_real_transforms)
                    results["Image Transform"]["total_remain_map"] += remain_perturb_array

                    original_gen_image_transforms = attgan_model.G(x_real_transforms, c_trg, mode='enc-dec') # AttGAN version
                    perturbed_gen_image_transforms = attgan_model.G(x_adv_transforms, c_trg, mode='enc-dec') # AttGAN version

                    transforms_result_list.append(x_adv_transforms)
                    transforms_result_list.append(original_gen_image_transforms)
                    transforms_result_list.append(perturbed_gen_image_transforms)

                    results = calculate_and_save_metrics(original_gen_image_transforms, perturbed_gen_image_transforms, "Image Transform", results)


                with torch.no_grad():
                    # Calculate and accumulate PSNR, SSIM, LPIPS for invisibility
                    x_real_np = x_real.squeeze(0).permute(1, 2, 0).cpu().numpy() # (C, H, W) -> (H, W, C)
                    perturbed_image_np = perturbed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    invisible_lpips_value = self.lpips_loss(x_real, perturbed_image).mean()
                    invisible_psnr_value = psnr(x_real_np, perturbed_image_np, data_range=1.0)
                    invisible_ssim_value = ssim(x_real_np, perturbed_image_np, data_range=1.0, win_size=3, channel_axis=2)

                    total_invisible_lpips += invisible_lpips_value
                    total_invisible_psnr += invisible_psnr_value
                    total_invisible_ssim += invisible_ssim_value

                    episode += 1 # Here, 1 episode = 1 "face attribute transformation result" for 1 image


                # Update Target Network
                if episode % self.target_update_interval == 0:
                    # print(f"[DEBUG] Update Target Network at episode {episode}")
                    self.rl_agent.update_target_net()
                    self.rl_agent.reset_noise() # Reset Target Network Noise (NoisyNet)


            # Save the list of 5 "face attribute transformation" result images as a single image
            all_result_lists = [noattack_result_list, jpeg_result_list, opencv_result_list, median_result_list, padding_result_list, transforms_result_list]
            row_images = []
            for result_list in all_result_lists:
                row_concat = torch.cat(result_list, dim=3) # Concatenate images horizontally
                row_images.append(row_concat)


            # Create a blank image (white) for vertical spacing
            spacing = 10 # Adjust spacing size
            blank_image = torch.ones_like(row_images[0][:, :, :spacing, :]) # Fill with white, shape based on the first row_image
            blank_image = blank_image * 1.0 # White

            # Add the first row_image without spacing
            vertical_concat_list = [row_images[0]]
            # Concatenate the rest of the row_images vertically with blank images
            for i in range(1, len(row_images)):
                vertical_concat_list.append(blank_image) # Add spacing
                vertical_concat_list.append(row_images[i])

            x_concat = torch.cat(vertical_concat_list, dim=2) # Final vertical concatenation of images
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(test_img_idx + 1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)


            # Save the Rainbow DQN agent at every episode
            checkpoint_path = os.path.join(self.model_save_dir, f'final_rainbow_dqn.pth')            
            try: 
                torch.save({
                'rainbow_dqn_state_dict': self.rl_agent.dqn.state_dict(),
                'optimizer_state_dict': self.rl_agent.optimizer.state_dict(),
                }, checkpoint_path)
                print(f"[*] Saved Rainbow DQN agent weights and optimizer weights (episode {episode}) -> {checkpoint_path}")
            except Exception as e:
                print(f"[!] Error saving Rainbow DQN weights and optimizer weights (episode {episode}): {e}")


            if test_img_idx >= (self.training_image_num - 1): # Process only self.training_image_num number of images
                break

        plot_reward_trend(reward_per_episode, window_size=25, save_path=os.path.join(self.result_dir, "reward_trend.png")) # Save reward trend plot
        save_reward_moving_average_txt(reward_per_episode, window_size=25, save_path=os.path.join(self.result_dir, "reward_moving_avg.txt")) # Save moving average to a text file
        score = print_comprehensive_metrics(results, episode, total_invisible_psnr, total_invisible_ssim, total_invisible_lpips)
        visualize_actions(action_history, image_indices, attr_indices, step_indices)

        # Part to save the trained model for inference
        # print_final_metrics(episode, total_perturbation_map, total_remain_map, total_l1_error, total_l2_error, attack_success, no_gan_psnr, no_gan_ssim, no_gan_lpips, gan_psnr, gan_ssim, gan_lpips)
        checkpoint_path = os.path.join(self.model_save_dir, f'final_rainbow_dqn.pth')
        try: 
            torch.save({
                'rainbow_dqn_state_dict': self.rl_agent.dqn.state_dict(),
                'optimizer_state_dict': self.rl_agent.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"[*] Saved Rainbow DQN agent weights and optimizer weights to: {checkpoint_path}")
        except Exception as e:
            print(f"[!] Error occurred while saving Rainbow DQN agent weights and optimizer weights: {e}")
        print(f"[INFO] Finished saving Rainbow DQN agent weights and optimizer weights: {checkpoint_path}")
        if self.run is not None:
            self.run.stop()
    
    """Helper function to create an N-step transition"""
    def _get_n_step_transition(self, n_step_buffer):
        state, action = n_step_buffer[0][:2] # Start State, Action
        n_step_reward = 0
        next_state = n_step_buffer[-1][3] # Initialization, the final next_state must be updated even if done=True
        done = False # Initialization

        for i in range(self.n_step): # Iterate through the N-step buffer
            reward, s, d = n_step_buffer[i][2:] # Extract reward, next_state, done
            n_step_reward += reward * (self.gamma ** i) # Accumulate N-step discounted reward
            # Ensure d is a boolean value before using in conditional
            d_bool = bool(d) # Convert d to a boolean
            next_state, done = (s, d_bool) if d_bool else (next_state, done) # If done=True is encountered, update next_state and done
            if d_bool: # If done=True is encountered in the middle, break (episode ends)
                break
        return state, action, n_step_reward, next_state, done # Return N-step transition



"""
Rainbow DQN Agent (Prioritized Experience Replay, Dueling DQN, Noisy Network, Categorical DQN, Double DQN, N-step Learning) 
-> Uses only N-step Loss, applies N-step Transition sampling
"""
class RainbowDQNAgent:
    def __init__(self, state_dim, action_dim, agent_lr, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_interval,
                v_min, v_max, atom_size, beta_start, beta_frames, prior_eps, n_step, initial_ratio_steps):
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(device) # Support (Categorical DQN)
        self.delta_z = float(v_max - v_min) / (atom_size - 1) # delta_z (Categorical DQN)

        self.epsilon_start = epsilon_start # Epsilon (not used by NoisyNet)
        self.epsilon_end = epsilon_end # Epsilon (not used by NoisyNet)
        self.epsilon_decay = epsilon_decay # Epsilon Decay (not used by NoisyNet)

        self.prior_eps = prior_eps # Prioritized Replay Buffer Prior Epsilon
        self.lr = agent_lr
        self.gamma = gamma
        self.beta_start = beta_start # Beta (PER)
        self.beta_frames = beta_frames # Beta frames (PER)
        self.n_step = n_step # N-step Learning
        self.target_update_interval = target_update_interval
        self.frame_idx = 0 # frame index for beta annealing
        self.v_min = v_min
        self.v_max = v_max

        # Added: Variables for initial ratio-based action selection
        self.steps_done = 0
        self.initial_ratio_steps = initial_ratio_steps

        # Rainbow DQN Network (DuelingNet, NoisyNet, Categorical DQN)
        self.dqn = RainbowDQNNet(state_dim, action_dim, atom_size, self.support) # policy net -> dqn, changed to RainbowDQNNet
        self.dqn_target = RainbowDQNNet(state_dim, action_dim, atom_size, self.support) # target net -> dqn_target, changed to RainbowDQNNet
        self.dqn_target.load_state_dict(self.dqn.state_dict()) # Initialize Target Network
        self.optimizer = Adam(self.dqn.parameters(), lr=self.lr, eps=0.01/32) # Optimizer is Adam, added eps (for stability)


    """
    NoisyNet does not use Epsilon-greedy, uses Value-based action selection
    """
    def select_action(self, state):
        # [For testing] Force selection of only one action
        # return torch.tensor([[0]], device=device)

        # In the initial steps, select actions based on a 4:1:1:4 ratio
        self.steps_done += 1
        if self.steps_done <= self.initial_ratio_steps:
            # Generate a random number between 0 and 9
            rand_num = random.randint(0, 9)
            
            # Implement 4:1:1:4 ratio (actions 0, 1, 2, 3 are selected with 40%, 10%, 10%, 40% probability respectively)
            if rand_num <= 3:  # 0-3 (40%)
                return torch.tensor([[0]], device=device)  # PGD method
            elif rand_num == 4:  # 4 (10%)
                return torch.tensor([[1]], device=device)  # Low-frequency
            elif rand_num == 5:  # 5 (10%)
                return torch.tensor([[2]], device=device)  # Mid-frequency
            else:  # 6-9 (40%)
                return torch.tensor([[3]], device=device)  # High-frequency
        
        # After the initial steps, select actions using the standard NoisyNet method
        with torch.no_grad():
            self.dqn.reset_noise()
            q_values = self.dqn(state)
            return q_values.argmax(dim=1).view(1, 1)

    """Receives weights, indices, and batch data required for PER -> N-step Transition"""
    def update_model(self, batch, weights, indices, batch_size, frame_idx):
        weights = weights.to(device) # Importance sampling weights (PER)
        indices = indices.to(device) # Sample indices (PER)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward) # N-step reward
        next_state_batch = torch.cat(batch.next_state) # N-step next state
        done_mask = torch.cat(batch.done).float() # N-step done mask

        # Calculate Categorical DQN Loss (with Double DQN & N-step Learning) -> Uses only N-step Loss
        loss, elementwise_loss = self._compute_rainbow_dqn_loss(batch, weights, batch_size, frame_idx)

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0) # Apply Gradient Clipping (for stability)
        self.optimizer.step()

        # PER priority update, elementwise_loss + prior_eps
        prios = elementwise_loss.sum(dim=1).detach().cpu().numpy() + self.prior_eps # Calculate priority, add prior_eps (for non-zero priority), sum over atom_size dimension
        return loss.item(), prios # Return loss value and priority values


    """Add frame_idx, Calculate Categorical DQN Loss (with Double DQN & N-step Loss & N-step Transition)"""
    def _compute_rainbow_dqn_loss(self, batch, weights, batch_size, frame_idx):
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward) # N-step reward
        next_state_batch = torch.cat(batch.next_state) # N-step next state
        done_mask = torch.cat(batch.done).float() # N-step done mask

        # [Unify device] Move reward_batch, done_mask to the device
        reward_batch = reward_batch.to(device)
        done_mask = done_mask.to(device)

        # Calculate Distribution
        current_dist = self.dqn.get_distribution(state_batch) # Current state distribution
        log_p = torch.log(current_dist[range(batch_size), action_batch.squeeze(1)]) # log probability (for cross-entropy loss calculation)

        with torch.no_grad():
            self.dqn_target.reset_noise() # Reset Target Network Noise (NoisyNet)
            next_action = self.select_action(next_state_batch) # Double DQN: action selection is from policy_net
            next_dist = self.dqn_target.get_distribution(next_state_batch) # target distribution is from target_net
            next_dist = next_dist[range(batch_size), next_action.squeeze(1)] # distribution of the selected action

            # Apply N-step Learning (when calculating Target Distribution) - use N-step reward, next_state, done_mask
            n_step = self.n_step
            gamma = self.gamma
            target_dist = torch.zeros((batch_size, self.atom_size), device=device) # Adjust target_dist shape (batch_size, atom_size)

            t_z = reward_batch.reshape(-1, 1) + (1.0 - done_mask.reshape(-1, 1)) * (gamma**n_step) * self.support.unsqueeze(0) # Apply N-step reward, done_mask, reshape(-1, 1) for broadcasting
            t_z = t_z.clamp(min=self.v_min, max=self.v_max) # Clamp to the support range

            b = (t_z - self.v_min) / self.delta_z # Calculate bin index
            l = b.floor().long().clamp(min=0, max=self.atom_size - 1) # lower bin index
            u = b.ceil().long().clamp(min=0, max=self.atom_size - 1) # upper bin index

            offset = (torch.arange(batch_size) * self.atom_size).long().to(device).unsqueeze(1) # Calculate offset index, unsqueeze(1) for broadcasting

            proj_dist = torch.zeros(next_dist.size(), device=device) # Initialize projection distribution (for batch)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)) # Use l, u directly as indices, view(-1) for flatten
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)) # Use l, u directly as indices, view(-1) for flatten

        # Cross-entropy loss with IS weights (PER) - Uses only N-step Loss
        elementwise_loss = -proj_dist * log_p # elementwise loss (cross-entropy)
        loss = torch.mean(weights * elementwise_loss.sum(dim=1)) # Apply Importance Sampling weight (PER), calculate mean loss

        return loss, elementwise_loss # Return loss, elementwise_loss

    """Target Network Update (Hard Update)"""
    def update_target_net(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    """Policy Network Noise Reset (at every step)"""
    def reset_noise(self):
        self.dqn.reset_noise()
        # Target Network Noise Reset is called before loss calculation via reset_noise()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device outside SolverRainbow
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) # Define Transition outside SolverRainbow
