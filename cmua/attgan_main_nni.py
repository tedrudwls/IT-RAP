import os
import argparse
import torch
import nni

from attgan_solver import SolverRainbow
from attgan_data_loader import get_loader
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    cudnn.benchmark = True

    # NNI에서 step_size parameter 받기
    params = nni.get_next_parameter()
    
    if params:
        config.cmua_step_size = params['cmua_step_size']
        trial_id = nni.get_trial_id()
        print("\n" + "="*70)
        print(f"NNI Trial ID: {trial_id}")
        print(f"Testing Step Size: {config.cmua_step_size:.6f}")
        print("="*70 + "\n")
    else:
        # Standalone 실행 (NNI 없이)
        print("\n" + "="*70)
        print(f"Standalone Mode")
        print(f"Step Size: {config.cmua_step_size:.6f}")
        print("="*70 + "\n")

    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    # Data loader
    if config.dataset == 'CelebA':
        dataset_loader = get_loader(
            config.images_dir, config.attr_path, config.selected_attrs,
            config.celeba_crop_size, config.image_size, config.batch_size,
            'CelebA', config.mode, config.num_workers, config.start_index
        )
    elif config.dataset == 'MAADFace':
        dataset_loader = get_loader(
            config.images_dir, config.attr_path, config.selected_attrs,
            config.celeba_crop_size, config.image_size, config.batch_size,
            'MAADFace', config.mode, config.num_workers, config.start_index
        )

    solver = SolverRainbow(dataset_loader, config)

    # CMUA Training
    print(f"\n[Training] CMUA with step_size={config.cmua_step_size:.6f}")
    print(f"  Training images: {config.cmua_train_images}")
    print(f"  Evaluation images: {config.cmua_eval_images}\n")
    
    # AttGAN 모델 로드
    attgan_checkpoint_path = config.attgan_checkpoint
    solver.restore_attgan_model(attgan_checkpoint_path)
    
    # CMUA 학습 (train_cmua_with_step_size 필요 - 아래에서 추가 예정)
    universal_perturbation = solver.train_cmua_with_step_size(config.cmua_step_size)
    
    # Success rate 평가
    success_rate = solver.evaluate_cmua_success_rate(
        universal_perturbation, 
        config.cmua_eval_images
    )
    
    print(f"\n{'='*70}")
    print(f"Step Size: {config.cmua_step_size:.6f}")
    print(f"Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # NNI에 결과 리포트
    if params:
        nni.report_final_result(success_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--c_dim', type=int, default=13, help='AttGAN uses 13 attributes')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--selected_attrs', '--list', nargs='+',
                        default=['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                                'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open',
                                'Mustache', 'No_Beard', 'Pale_Skin', 'Young'])
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--images_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    
    # CMUA specific
    parser.add_argument('--cmua_step_size', type=float, default=0.01, 
                       help='Step size for CMUA (will be overridden by NNI)')
    parser.add_argument('--cmua_iterations', type=int, default=10)
    parser.add_argument('--cmua_momentum', type=float, default=0.5)
    parser.add_argument('--cmua_epsilon', type=float, default=0.05)
    parser.add_argument('--cmua_batch_size', type=int, default=64)
    parser.add_argument('--cmua_train_images', type=int, default=128)
    parser.add_argument('--cmua_eval_images', type=int, default=100)
    
    # AttGAN checkpoint
    parser.add_argument('--attgan_checkpoint', type=str,
                       default='/scratch/x3092a02/stargan2/attgan/256_shortcut1_inject1_none_hq/checkpoint/weights.199.pth')
    
    # Directories
    parser.add_argument('--log_dir', type=str, default='attgan/logs')
    parser.add_argument('--model_save_dir', type=str, default='attgan_celeba_256/models')
    parser.add_argument('--sample_dir', type=str, default='attgan/samples')
    parser.add_argument('--result_dir', type=str, default='attgan/results')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--start_index', type=int, default=0)
    
    # Training (SolverRainbow에서 필요할 수 있는 것들)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--training_image_num', type=int, default=5)
    parser.add_argument('--inference_image_num', type=int, default=5)
    parser.add_argument('--reward_weight', type=float, default=0.5)
    parser.add_argument('--test_iters', type=int, default=200000)
    
    # Rainbow DQN (필요할 수 있음)
    parser.add_argument('--agent_lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--target_update_interval', type=int, default=5)
    parser.add_argument('--memory_capacity', type=int, default=512)
    parser.add_argument('--max_steps_per_episode', type=int, default=20)
    parser.add_argument('--action_dim', type=int, default=4)
    parser.add_argument('--noise_level', type=float, default=0.005)
    parser.add_argument('--feature_extractor_name', type=str, default="ghostfacenets")
    parser.add_argument('--feature_extractor_frequency', type=int, default=1)
    parser.add_argument('--epsilon_start', type=float, default=0.95)
    parser.add_argument('--epsilon_end', type=float, default=0.04)
    parser.add_argument('--epsilon_decay', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--beta_start', type=float, default=0.35)
    parser.add_argument('--beta_frames', type=int, default=4000)
    parser.add_argument('--prior_eps', type=float, default=1e-6)
    parser.add_argument('--v_min', type=int, default=-5)
    parser.add_argument('--v_max', type=int, default=5)
    parser.add_argument('--atom_size', type=int, default=11)
    parser.add_argument('--n_step', type=int, default=5)
    parser.add_argument('--pgd_iter', type=int, default=1)
    parser.add_argument('--dct_iter', type=int, default=1)
    parser.add_argument('--dct_coefficent', type=int, default=3)
    parser.add_argument('--dct_clamp', type=int, default=2)
    
    # Neptune (optional)
    parser.add_argument('--neptune_api_token', type=str, default=None)
    parser.add_argument('--neptune_project', type=str, default=None)

    config = parser.parse_args()
    main(config)
