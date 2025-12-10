import neptune
import os
import argparse
from torch.backends import cudnn

from stargan_solver import SolverRainbow
from stargan_data_loader import get_loader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True
    
    # neptune 설정
    run = neptune.init_run(
        project="input_your_project_name",
        api_token="input_your_apitoken",
    )

    # Create directories if not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # ✅ CMUA train일 때는 data loader mode를 'train'으로 설정
    data_loader_mode = config.mode
    if config.mode == 'inference' and config.attack_method == 'cmua' and config.cmua_mode == 'train':
        data_loader_mode = 'train'
        print(f"[INFO] CMUA train mode: using training split (same as IT-RAP train)")

    # Data loader
    dataset_loader = None

    if config.dataset == 'CelebA':
        dataset_loader = get_loader(config.images_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', data_loader_mode, config.num_workers, config.start_index)  # ✅ 수정
    elif config.dataset == 'MAADFace':
        dataset_loader = get_loader(config.images_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, config.batch_size,
                                'MAADFace', data_loader_mode, config.num_workers, config.start_index)  # ✅ 수정

    solver = SolverRainbow(dataset_loader, config, run=run)

    if config.mode == 'train':
        if config.attack_method == 'itrap':
            solver.train_attack()
        elif config.attack_method == 'cmua':
            print("[ERROR] CMUA training is not supported in 'train' mode.")
            print("[INFO] Use --mode inference --cmua_mode train instead.")
            return
        else:
            print(f"[ERROR] Unknown attack method: {config.attack_method}")
            return

    elif config.mode == 'inference':
        if config.attack_method == 'itrap':
            # IT-RAP inference (Rainbow DQN)
            checkpoint_path = os.path.join(config.model_save_dir, f'final_rainbow_dqn.pth')
            solver.load_rainbow_dqn_checkpoint(checkpoint_path)
            # Load StarGAN model (required)
            solver.restore_model(config.test_iters)
            # Perform inference
            solver.inference_rainbow_dqn(dataset_loader, result_dir=config.result_dir)
        
        elif config.attack_method == 'cmua':
            # Load StarGAN model (required for both modes)
            solver.restore_model(config.test_iters)
            
            if config.cmua_mode == 'train':
                # CMUA Train: Generate universal perturbation
                # dataset_loader는 이미 train split 사용 중
                solver.train_cmua(dataset_loader, result_dir=config.result_dir)
            
            elif config.cmua_mode == 'inference':
                # CMUA Inference: Apply saved perturbation
                # dataset_loader는 test split 사용
                solver.inference_cmua(dataset_loader, result_dir=config.result_dir)
            
            else:
                print(f"[ERROR] Unknown cmua_mode: {config.cmua_mode}")
                print("[INFO] Use --cmua_mode train or --cmua_mode inference")
                return
        
        else:
            print(f"[ERROR] Unknown attack method: {config.attack_method}")
            return



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
    
    # Attack method selection (NEW)
    parser.add_argument('--attack_method', type=str, default='itrap', choices=['itrap', 'cmua'], 
                        help='Attack method: itrap (IT-RAP with Rainbow DQN) or cmua (CMUA universal perturbation)')

    # Directory settings
    parser.add_argument('--images_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='checkpoints/models')
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
    parser.add_argument('--feature_extractor_name', type=str, default="edgeface", help='Image feature extraction for State (mesonet, resnet50, vgg19, ghostfacenets, edgeface)')
    parser.add_argument('--feature_extractor_frequency', type=int, default=1, help='Feature extractor call frequency (1=every step, 2=every 2 steps, etc.)')


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
    # ✅ CMUA specific parameters 추가
    parser.add_argument('--cmua_mode', type=str, default='inference', choices=['train', 'inference'], help='CMUA mode: train (generate perturbation) or inference (apply saved perturbation)')
    parser.add_argument('--cmua_train_images', type=int, default=100, help='Number of training images for CMUA (논문은 128)')
    parser.add_argument('--cmua_inference_images', type=int, default=100, help='Number of inference images for CMUA')
    parser.add_argument('--cmua_perturbation_path', type=str, default='cmua_universal_perturbation.pt', help='Path to save/load universal perturbation')
    # CMUA 기존 파라미터들 
    parser.add_argument('--cmua_iterations', type=int, default=20, help='Number of PGD iterations for CMUA')
    parser.add_argument('--cmua_step_size', type=float, default=0.01,help='Step size for PGD in CMUA')
    parser.add_argument('--cmua_epsilon', type=float, default=0.05, help='Epsilon for CMUA (논문 권장: 0.05)')
    parser.add_argument('--cmua_momentum', type=float, default=0.9, help='Momentum for CMUA gradient updates')
    parser.add_argument('--cmua_batch_size', type=int, default=64, help='Batch size for CMUA training (논문: 64)')
    config = parser.parse_args()

    print(config)
    main(config)
