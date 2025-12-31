#!/usr/bin/env python
"""
AttGAN CMUA Step Size Optimization using NNI TPE

실행명령어(attgan_main_nni.py 파일 필요홤)
python -u run_attgan_nni_optimization.py \
  --trials 1000 \
  --train-images 128 \
  --eval-images 100 \
  --duration 48h \
  --port 8080 \
  --debug \
  2>&1 | tee attgan_nni_final.log
"""

from nni.experiment import Experiment
from nni.experiment.config.training_services import LocalConfig

def run_attgan_cmua_optimization(n_trials=1000, max_duration='48h', port=8080, 
                                  debug=False, train_images=128, eval_images=100):
    """
    AttGAN CMUA Step Size TPE 최적화
    
    Args:
        n_trials: Trial 개수 (논문: 1000)
        max_duration: 최대 실행 시간
        port: Web UI 포트
        debug: 디버그 모드
        train_images: Training images 수 (논문: 128)
        eval_images: Evaluation images 수 (논문: 100+)
    """
    
    # Experiment 생성
    experiment = Experiment('local')
    
    # 기본 설정
    experiment.config.experiment_name = f'AttGAN_CMUA_StepSize_TPE_{n_trials}trials'
    experiment.config.trial_concurrency = 1
    experiment.config.max_trial_number = n_trials
    experiment.config.max_experiment_duration = max_duration
    
    # Search Space (논문: [0, 10])
    experiment.config.search_space = {
        'cmua_step_size': {
            '_type': 'uniform',
            '_value': [0.0, 10.0]
        }
    }
    
    # Trial 설정
    experiment.config.trial_command = (
        f'export QT_QPA_PLATFORM=offscreen && PYTHONPATH=. python attgan_main_nni.py '
        f'--cmua_train_images {train_images} --cmua_eval_images {eval_images}'
    )
    
    experiment.config.trial_code_directory = '.'
    
    # Tuner 설정 (TPE)
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize'  # Success rate 최대화
    }
    
    # Training Service 설정
    local_config = LocalConfig()
    local_config.use_active_gpu = True
    local_config.max_trial_number_per_gpu = 1
    local_config.gpu_indices = [0]
    
    experiment.config.training_service = local_config
    
    # 실험 시작
    print("\n" + "="*70)
    print(f"Starting AttGAN CMUA Step Size Optimization")
    print("="*70)
    print(f"  Experiment Name: {experiment.config.experiment_name}")
    print(f"  Max Trials: {n_trials}")
    print(f"  Max Duration: {max_duration}")
    print(f"  Search Space: [0.0, 10.0]")
    print(f"  Tuner: TPE")
    print(f"  Port: {port}")
    print(f"  Training Images: {train_images}")
    print(f"  Evaluation Images: {eval_images}")
    print("="*70)
    print(f"\nWeb UI will be available at: http://localhost:{port}")
    print("="*70 + "\n")
    
    print("="*70)
    print("Experiment Started Successfully!")
    print("="*70)
    print(f"\n📊 Monitor at: http://localhost:{port}")
    print(f"\n⏸️  Press Ctrl+C to stop the experiment")
    print("="*70 + "\n")
    
    if debug:
        print("[DEBUG MODE] Experiment will run until completion or Ctrl+C")
        print("[DEBUG MODE] Check trial logs at: ~/nni-experiments/\n")
    
    try:
        experiment.start(port, debug=True)
        print(f"\n✅ Experiment started! Experiment ID: {experiment.id}")
        print(f"   Waiting for {n_trials} trials to complete...")
        print(f"   This will take approximately: {max_duration}")
        print(f"\n   Press Ctrl+C to stop early\n")
        
        import time
        check_count = 0
        while True:
            time.sleep(5)
            check_count += 1
            
            try:
                trials = experiment.list_trial_jobs()
                if trials:
                    completed = sum(1 for t in trials if hasattr(t, 'status') and t.status in ['SUCCEEDED', 'FAILED'])
                    running = sum(1 for t in trials if hasattr(t, 'status') and t.status == 'RUNNING')
                    
                    if debug and check_count % 6 == 0:
                        print(f"\n[Status] Completed: {completed}/{n_trials}, Running: {running}, Total: {len(trials)}")
                        for i, t in enumerate(trials[:5]):
                            trial_id = getattr(t, 'trial_job_id', getattr(t, 'id', f'Trial{i}'))
                            status = getattr(t, 'status', 'UNKNOWN')
                            print(f"  {trial_id}: {status}")
                    else:
                        print(f"[Progress] {completed}/{n_trials} trials completed (Running: {running})", end='\r')
                    
                    if completed >= n_trials:
                        print(f"\n✅ All {n_trials} trials completed!")
                        break
                else:
                    print(f"[Wait] Waiting for trials to start... ({check_count*5}s)", end='\r')
            except Exception as e:
                if debug:
                    print(f"\n[Debug] Status check error: {e}")
                pass
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Experiment error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    experiment.stop()
    
    print("\n" + "="*70)
    print("Experiment Results")
    print("="*70)
    
    try:
        trials = experiment.list_trial_jobs()
    except Exception as e:
        print(f"\n⚠️  Could not retrieve job statistics: {e}")
        print(f"ℹ️  Experiment may have been stopped early or trials failed to run.")
        print(f"To check experiment logs:")
        print(f"  ls -la ~/nni-experiments/")
        print(f"To resume or view results, use analyze_nni_results.py:")
        print(f"  python analyze_nni_results.py --exp-dir ~/nni-experiments/{experiment.id}")
        trials = None
    
    if trials:
        valid_trials = [t for t in trials 
                       if hasattr(t, 'final_metric') and t.final_metric is not None]
        
        if valid_trials:
            best_trial = max(valid_trials, 
                           key=lambda t: float(t.final_metric) if t.final_metric else 0)
            
            print(f"\n🏆 Best Trial:")
            trial_id = getattr(best_trial, 'trial_job_id', getattr(best_trial, 'id', 'Unknown'))
            print(f"   Trial ID: {trial_id}")
            
            params = getattr(best_trial, 'hyperparameters', 
                           getattr(best_trial, 'hyper_parameters', None))
            
            if params:
                print(f"   Parameters: {params}")
                
                step_size = params.get('cmua_step_size') if isinstance(params, dict) else None
                if step_size:
                    print(f"   Best Step Size: {step_size}")
            
            if hasattr(best_trial, 'final_metric') and best_trial.final_metric:
                success_rate = float(best_trial.final_metric)
                print(f"   Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
            
            print("\n📌 Use this step_size for final training:")
            if params and isinstance(params, dict) and 'cmua_step_size' in params:
                print(f"   python attgan_main.py --mode train --attack_method cmua \\")
                print(f"   --cmua_step_size {params['cmua_step_size']} \\")
                print(f"   --cmua_iterations 10 --cmua_momentum 0.5 --cmua_epsilon 0.05 \\")
                print(f"   --cmua_batch_size 64 --cmua_train_images 128")
        else:
            print("\n⚠️  No valid trials with metrics found.")
    else:
        print("\n⚠️  No trials found.")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AttGAN CMUA Step Size Optimization using NNI TPE')
    parser.add_argument('--trials', type=int, default=1000, 
                       help='Number of trials (논문: 1000)')
    parser.add_argument('--duration', type=str, default='48h',
                       help='Max experiment duration (e.g., 48h, 100h)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Web UI port (default: 8080)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--train-images', type=int, default=128,
                       help='Number of training images (논문: 128)')
    parser.add_argument('--eval-images', type=int, default=100,
                       help='Number of evaluation images (논문: 100+)')
    
    args = parser.parse_args()
    
    run_attgan_cmua_optimization(
        n_trials=args.trials,
        max_duration=args.duration,
        port=args.port,
        debug=args.debug,
        train_images=args.train_images,
        eval_images=args.eval_images
    )
