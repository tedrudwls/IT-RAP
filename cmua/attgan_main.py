import os
import argparse
from attgan_solver import SolverRainbow

from attgan_data_loader import get_loader
from torch.backends import cudnn

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # neptune logging is handled inside SolverRainbow class
    run = None

    # Create directories if not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader
    dataset_loader = None

    if config.dataset == 'CelebA':
        dataset_loader = get_loader(config.images_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers, config.start_index)
    elif config.dataset == 'MAADFace':
        dataset_loader = get_loader(config.images_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, config.batch_size,
                                'MAADFace', config.mode, config.num_workers, config.start_index)

    solver = SolverRainbow(dataset_loader, config)

    if config.mode == 'train':
        solver.train_attack()

    elif config.mode == 'inference':
        # checkpoint_path = os.path.join(config.model_save_dir, f'rainbow_dqn_final_{config.test_iters}.pth')
        checkpoint_path = os.path.join(config.model_save_dir, f'final_rainbow_dqn.pth') # rainbow_dqn_agent.ckpt
        solver.load_rainbow_dqn_checkpoint(checkpoint_path)
        # Load AttGAN model (required)
        # Use fixed AttGAN checkpoint path instead of test_iters
        attgan_checkpoint_path = '/scratch/x3092a02/stargan2/attgan/256_shortcut1_inject1_none_hq/checkpoint/weights.199.pth'
        solver.restore_attgan_model(attgan_checkpoint_path)
        # Perform inference
        solver.inference_rainbow_dqn(dataset_loader, result_dir=config.result_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Existing parameter settings...
    # Model configuration
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    
    # Training configuration
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both', 'MAADFace'])
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    parser.add_argument('--training_image_num', type=int, default=5, help='Number of images used to train Rainbow DQN')
    parser.add_argument('--inference_image_num', type=int, default=5, help='Number of images used to inference Rainbow DQN')
    # Starting index to resume training from the point of interruption when training with the MAAD-FACE dataset
    parser.add_argument('--start_index', type=int, default=0, help='Data index to start training from')
    parser.add_argument('--reward_weight', type=float, default=0.5, help='Reward weight (Deepfake defense: reward_weight, Imperceptibility: 1 - reward_weight)')


    # Test configuration
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous settings
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference']) # Changed mode to train

    # Directory settings
    parser.add_argument('--images_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan_celeba_256/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/result_test') # Changed result_dir

    parser.add_argument('--epsilon_start', type=float, default=0.95, help='epsilon start value for exploration') # Currently not in use
    parser.add_argument('--epsilon_end', type=float, default=0.04, help='epsilon end value for exploration') # Currently not in use
    parser.add_argument('--epsilon_decay', type=int, default=1000, help='epsilon decay steps') # Currently not in use

    # Rainbow DQN Hyperparameters (PER, Categorical DQN, N-step Learning)
    parser.add_argument('--batch_size', type=int, default=1, help='How many images to process at once')
    parser.add_argument('--agent_lr', type=float, default=0.0001, help='learning rate for Agent')
    parser.add_argument('--gamma', type=float, default=0.96, help='discount factor for RL')
    parser.add_argument('--target_update_interval', type=int, default=5, help='target network update interval')
    parser.add_argument('--memory_capacity', type=int, default=512, help='replay memory capacity')
    parser.add_argument('--max_steps_per_episode', type=int, default=20, help='max steps per episode')
    parser.add_argument('--action_dim', type=int, default=4, help='max action dimension')
    parser.add_argument('--noise_level', type=float, default=0.005, help='noise level for RLAB perturbation')
    parser.add_argument('--feature_extractor_name', type=str, default="ghostfacenets", help='Image feature extraction for State (mesonet, resnet50, vgg19, ghostfacenets)')
    parser.add_argument('--feature_extractor_frequency', type=int, default=1, help='Feature extractor call frequency (1=every step, 2=every 2 steps, etc.)')
    
    # Neptune Logging Configuration
    parser.add_argument('--neptune_api_token', type=str, default=None, help='Neptune.ai API token for logging')
    parser.add_argument('--neptune_project', type=str, default=None, help='Neptune.ai project name (e.g., "rudwls281689/deepfake")')


    parser.add_argument('--alpha', type=float, default=0.8, help='PER alpha parameter')
    parser.add_argument('--beta_start', type=float, default=0.35, help='PER beta start parameter')
    parser.add_argument('--beta_frames', type=int, default=4000, help='PER beta frames parameter')
    parser.add_argument('--prior_eps', type=float, default=1e-6, help='PER prior epsilon parameter')
    parser.add_argument('--v_min', type=int, default=-5, help='Categorical DQN v_min value') 
    parser.add_argument('--v_max', type=int, default=5, help='Categorical DQN v_max value') 
    parser.add_argument('--atom_size', type=int, default=11, help='Categorical DQN atom size')
    parser.add_argument('--n_step', type=int, default=5, help='N-step Learning step size') 

    parser.add_argument('--pgd_iter', type=int, default=1, help='Action 0, number of PGD iterations')
    parser.add_argument('--dct_iter', type=int, default=1, help='Action 1~3, number of frequency noise insertion iterations')
    parser.add_argument('--dct_coefficent', type=int, default=3, help='DCT noise coefficient')
    parser.add_argument('--dct_clamp', type=int, default=2, help='DCT noise value clamp')

    config = parser.parse_args()

    print(config)
    main(config)
