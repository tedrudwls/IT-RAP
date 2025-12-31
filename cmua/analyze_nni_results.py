#!/usr/bin/env python
"""
NNI 실험 결과 분석 스크립트
실험 종료 후에도 결과 확인 가능
"""

import sqlite3
import json
import os
import argparse
from pathlib import Path

def analyze_nni_experiment(exp_dir):
    """NNI 실험 결과 분석"""
    
    db_path = os.path.join(exp_dir, 'db/nni.sqlite')
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 모든 trial 결과 가져오기
    query = """
    SELECT 
        t.trialJobId,
        t.data as params,
        m.data as metric
    FROM TrialJobEvent t
    LEFT JOIN MetricData m ON t.trialJobId = m.trialJobId
    WHERE t.event = 'RUNNING' AND m.type = 'FINAL'
    ORDER BY CAST(m.data AS REAL) DESC
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    if not results:
        print("⚠️  No results found")
        return
    
    print("="*80)
    print(f"NNI Experiment Results: {exp_dir}")
    print("="*80)
    
    # 결과 파싱
    trials_data = []
    for trial_id, params_str, metric_str in results:
        try:
            params = json.loads(params_str)
            step_size = params['parameters']['cmua_step_size']
            success_rate = float(metric_str.strip('"'))
            trials_data.append({
                'trial_id': trial_id,
                'step_size': step_size,
                'success_rate': success_rate
            })
        except:
            continue
    
    # 통계
    print(f"\n📊 Statistics:")
    print(f"   Total trials: {len(trials_data)}")
    
    if trials_data:
        step_sizes = [t['step_size'] for t in trials_data]
        success_rates = [t['success_rate'] for t in trials_data]
        
        print(f"   Step size range: [{min(step_sizes):.4f}, {max(step_sizes):.4f}]")
        print(f"   Success rate range: [{min(success_rates):.4f}, {max(success_rates):.4f}]")
        print(f"   Avg success rate: {sum(success_rates)/len(success_rates):.4f}")
    
    # Top 10 trials
    print(f"\n🏆 Top 10 Trials (by success rate):")
    print(f"{'Rank':<6} {'Trial ID':<10} {'Step Size':<15} {'Success Rate':<15}")
    print("-"*60)
    
    for i, trial in enumerate(trials_data[:10], 1):
        print(f"{i:<6} {trial['trial_id']:<10} {trial['step_size']:<15.6f} {trial['success_rate']:<15.4f}")
    
    # 100% success rate trials
    perfect_trials = [t for t in trials_data if t['success_rate'] >= 0.9999]
    
    if perfect_trials:
        print(f"\n✅ Trials with ~100% Success Rate: {len(perfect_trials)}")
        
        # 가장 작은 step_size (best)
        best_trial = min(perfect_trials, key=lambda x: x['step_size'])
        print(f"\n🎯 Best Step Size (smallest with 100% SR):")
        print(f"   Trial ID: {best_trial['trial_id']}")
        print(f"   Step Size: {best_trial['step_size']:.6f}")
        print(f"   Success Rate: {best_trial['success_rate']:.4f}")
        
        # Step size 분포
        perfect_step_sizes = sorted([t['step_size'] for t in perfect_trials])
        print(f"\n📈 100% Success Rate Step Size Range:")
        print(f"   Min: {min(perfect_step_sizes):.6f}")
        print(f"   Max: {max(perfect_step_sizes):.6f}")
        print(f"   Median: {perfect_step_sizes[len(perfect_step_sizes)//2]:.6f}")
    
    # 결과를 파일로 저장
    output_file = os.path.join(os.path.dirname(exp_dir), 'nni_results_summary.txt')
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"NNI Experiment Results Summary\n")
        f.write(f"Experiment: {exp_dir}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Trials: {len(trials_data)}\n")
        if trials_data:
            f.write(f"Step Size Range: [{min(step_sizes):.6f}, {max(step_sizes):.6f}]\n")
            f.write(f"Success Rate Range: [{min(success_rates):.4f}, {max(success_rates):.4f}]\n\n")
        
        f.write("All Trials (sorted by success rate):\n")
        f.write(f"{'Trial ID':<15} {'Step Size':<20} {'Success Rate':<20}\n")
        f.write("-"*60 + "\n")
        for trial in trials_data:
            f.write(f"{trial['trial_id']:<15} {trial['step_size']:<20.6f} {trial['success_rate']:<20.4f}\n")
        
        if perfect_trials:
            f.write(f"\n\nBest Step Size: {best_trial['step_size']:.6f}\n")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    conn.close()
    
    return best_trial['step_size'] if perfect_trials else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze NNI experiment results')
    parser.add_argument('--exp-dir', type=str, default='~/nni-experiments/_latest',
                       help='NNI experiment directory')
    
    args = parser.parse_args()
    
    exp_dir = os.path.expanduser(args.exp_dir)
    
    if not os.path.exists(exp_dir):
        print(f"❌ Experiment directory not found: {exp_dir}")
        print("\nAvailable experiments:")
        nni_base = os.path.expanduser('~/nni-experiments')
        if os.path.exists(nni_base):
            for exp in sorted(os.listdir(nni_base), reverse=True)[:10]:
                exp_path = os.path.join(nni_base, exp)
                if os.path.isdir(exp_path):
                    print(f"  {exp}")
    else:
        best_step_size = analyze_nni_experiment(exp_dir)
        
        if best_step_size:
            print("\n" + "="*80)
            print("📌 Next Steps:")
            print("="*80)
            print(f"\nUse the best step_size for final CMUA training:\n")
            print(f"python stargan_main.py --mode inference --attack_method cmua --cmua_mode train \\")
            print(f"  --cmua_step_size {best_step_size:.6f} \\")
            print(f"  --cmua_iterations 10 --cmua_momentum 0.5 --cmua_epsilon 0.05 \\")
            print(f"  --cmua_batch_size 64 --cmua_train_images 128 \\")
            print(f"  --model_save_dir ./stargan_celeba_128/models \\")
            print(f"  --celeba_image_dir /data/celeba \\")
            print(f"  --attr_path /data/celeba/list_attr_celeba.txt \\")
            print(f"  --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young")
