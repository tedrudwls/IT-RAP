"""
CMUA Step Size Optimization using NNI TPE
NNI Trial 코드
"""

import os
import sys
import argparse
import torch
import nni
from torch.backends import cudnn

# ✅ 현재 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stargan_data_loader import get_loader
from stargan_solver import SolverRainbow


def str2bool(v):
    return v.lower() in ('true', '1')


def get_nni_config():
    """NNI Trial용 Configuration"""
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', 
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
                        help='selected attributes for the CelebA dataset')
    
    # Test configuration
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    
    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='cmua_nni', choices=['train', 'test', 'cmua_nni'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    
    # Directories
    parser.add_argument('--images_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan_celeba_256/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='result_cmua_nni')
    
    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    
    # CMUA Attack configuration (NNI Search용)
    parser.add_argument('--attack_method', type=str, default='cmua')
    parser.add_argument('--cmua_mode', type=str, default='train')
    parser.add_argument('--cmua_iterations', type=int, default=10, 
                        help='PGD iterations (논문 공식: 10)')
    parser.add_argument('--cmua_epsilon', type=float, default=0.05,
                        help='L-infinity bound (논문 공식: 0.05)')
    parser.add_argument('--cmua_momentum', type=float, default=0.5,
                        help='MI-FGSM momentum (논문 공식: 0.5)')
    parser.add_argument('--cmua_step_size', type=float, default=0.01,
                        help='PGD step size (NNI가 자동으로 설정)')
    parser.add_argument('--cmua_train_images', type=int, default=32,
                        help='Number of training images for search (논문 search: 16-32, 빠른 테스트: 16)')
    parser.add_argument('--cmua_eval_images', type=int, default=20,
                        help='Number of evaluation images for search (빠른 평가용)')
    parser.add_argument('--cmua_batch_size', type=int, default=16,
                        help='Batch size for CMUA training (논문 search: 16)')
    parser.add_argument('--cmua_perturbation_path', type=str, 
                        default='cmua_universal_perturbation_nni.pt')
    
    # ✅ SolverRainbow에 필요한 모든 속성들 (stargan_main.py 기준)
    parser.add_argument('--training_image_num', type=int, default=5,
                        help='Number of images used to train Rainbow DQN')
    parser.add_argument('--inference_image_num', type=int, default=5,
                        help='Number of images used to inference Rainbow DQN')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Data index to start training from')
    parser.add_argument('--reward_weight', type=float, default=0.5,
                        help='Reward weight (Deepfake defense vs Imperceptibility)')
    parser.add_argument('--run_name', type=str, default='cmua_nni_search',
                        help='Experiment run name')
    
    # Rainbow DQN 관련 (IT-RAP용, CMUA에서는 사용 안 함)
    parser.add_argument('--agent_lr', type=float, default=0.0001,
                        help='learning rate for Agent')
    parser.add_argument('--gamma', type=float, default=0.96,
                        help='discount factor for RL')
    parser.add_argument('--target_update_interval', type=int, default=5,
                        help='target network update interval')
    parser.add_argument('--memory_capacity', type=int, default=512,
                        help='replay memory capacity')
    parser.add_argument('--max_steps_per_episode', type=int, default=20,
                        help='max steps per episode')
    parser.add_argument('--action_dim', type=int, default=4,
                        help='max action dimension')
    parser.add_argument('--noise_level', type=float, default=0.005,
                        help='noise level for RLAB perturbation')
    parser.add_argument('--feature_extractor_name', type=str, default="edgeface",
                        help='Image feature extraction for State')
    parser.add_argument('--feature_extractor_frequency', type=int, default=1,
                        help='Feature extractor call frequency')
    
    # PER parameters
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='PER alpha parameter')
    parser.add_argument('--beta_start', type=float, default=0.35,
                        help='PER beta start parameter')
    parser.add_argument('--beta_frames', type=int, default=4000,
                        help='PER beta frames parameter')
    parser.add_argument('--prior_eps', type=float, default=1e-6,
                        help='PER prior epsilon parameter')
    
    # Categorical DQN parameters
    parser.add_argument('--v_min', type=int, default=-5,
                        help='Categorical DQN v_min value')
    parser.add_argument('--v_max', type=int, default=5,
                        help='Categorical DQN v_max value')
    parser.add_argument('--atom_size', type=int, default=11,
                        help='Categorical DQN atom size')
    parser.add_argument('--n_step', type=int, default=5,
                        help='N-step Learning step size')
    
    # Action parameters
    parser.add_argument('--pgd_iter', type=int, default=1,
                        help='Action 0, number of PGD iterations')
    parser.add_argument('--dct_iter', type=int, default=1,
                        help='Action 1~3, number of frequency noise insertion iterations')
    parser.add_argument('--dct_coefficent', type=int, default=3,
                        help='DCT noise coefficient')
    parser.add_argument('--dct_clamp', type=int, default=2,
                        help='DCT noise value clamp')
    
    # Epsilon exploration parameters (IT-RAP용, 사용 안 함)
    parser.add_argument('--epsilon_start', type=float, default=0.95,
                        help='epsilon start value for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.04,
                        help='epsilon end value for exploration')
    parser.add_argument('--epsilon_decay', type=int, default=1000,
                        help='epsilon decay steps')
    
    config = parser.parse_args()
    return config


def main(config):
    """
    NNI Trial Main Function
    
    1. NNI로부터 step_size 파라미터 받기
    2. 해당 step_size로 CMUA 학습
    3. Success rate 평가
    4. NNI에 결과 리포트
    """
    # NNI에서 하이퍼파라미터 받기
    params = nni.get_next_parameter()
    
    if params:
        # NNI로부터 step_size 받음
        config.cmua_step_size = params['cmua_step_size']
        trial_id = nni.get_trial_id()
        print("\n" + "="*70)
        print(f"NNI Trial ID: {trial_id}")
        print(f"Testing Step Size: {config.cmua_step_size:.6f}")
        print("="*70 + "\n")
    else:
        # Standalone 실행 시 (디버깅용)
        print("\n" + "="*70)
        print("Warning: Running in standalone mode (not via NNI)")
        print(f"Using default step size: {config.cmua_step_size:.6f}")
        print("="*70 + "\n")
    
    # For fast debugging
    cudnn.benchmark = True
    
    # Create directories if not exist
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # Data loader (only CelebA for CMUA)
    celeba_loader = get_loader(
        config.images_dir, 
        config.attr_path,
        config.selected_attrs, 
        config.celeba_crop_size,
        config.image_size, 
        config.batch_size,
        'CelebA', 
        config.mode, 
        config.num_workers
    )
    
    # Solver (SolverRainbow uses dataset_loader, config, run)
    solver = SolverRainbow(celeba_loader, config, run=None)
    
    try:
        # 1. CMUA 학습 (Small batch, 적은 이미지로 빠른 학습)
        print(f"\n{'='*70}")
        print(f"[1/2] Training CMUA Universal Perturbation")
        print(f"{'='*70}")
        print(f"  Step Size: {config.cmua_step_size:.6f}")
        print(f"  Training Images: {config.cmua_train_images}")
        print(f"  Batch Size: {config.cmua_batch_size}")
        print(f"  Iterations: {config.cmua_iterations}")
        print(f"  Epsilon: {config.cmua_epsilon}")
        print(f"  Momentum: {config.cmua_momentum}")
        print(f"{'='*70}\n")
        
        # ✅ train_cmua_with_step_size 호출
        universal_perturbation = solver.train_cmua_with_step_size(config.cmua_step_size)
        
        # 2. Success Rate 평가 (적은 이미지로 빠른 평가)
        print(f"\n{'='*70}")
        print(f"[2/2] Evaluating Success Rate")
        print(f"{'='*70}")
        print(f"  Evaluation Images: {config.cmua_eval_images}")
        print(f"{'='*70}\n")
        
        success_rate = solver.evaluate_cmua_success_rate(
            universal_perturbation,
            num_eval_images=config.cmua_eval_images
        )
        
        # 3. NNI에 최종 결과 리포트
        nni.report_final_result(success_rate)
        
        print(f"\n{'='*70}")
        print(f"Trial Complete - Results")
        print(f"{'='*70}")
        print(f"  Step Size: {config.cmua_step_size:.6f}")
        print(f"  Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
        print(f"{'='*70}\n")
        
        return success_rate
        
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"Trial Failed with Error")
        print(f"{'!'*70}")
        print(f"Error: {e}")
        print(f"{'!'*70}\n")
        
        import traceback
        traceback.print_exc()
        
        # 실패 시 0.0 리포트
        nni.report_final_result(0.0)
        
        return 0.0


if __name__ == '__main__':
    config = get_nni_config()
    
    print("\n" + "="*70)
    print("CMUA Step Size Optimization - NNI Trial")
    print("Paper: CMUA-Watermark (AAAI 2022)")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Image Size: {config.image_size}")
    print(f"  Selected Attributes: {config.selected_attrs}")
    print(f"  Model Path: {config.model_save_dir}")
    print(f"  Result Path: {config.result_dir}")
    print("="*70 + "\n")
    
    main(config)
