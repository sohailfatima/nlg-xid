#!/usr/bin/env python3
"""
Run ablation studies on trained models.
This script automates running multiple experiments with different settings.
"""

import argparse
import subprocess
import os
import json
from itertools import product


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {description}")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies on trained models')
    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (.pkl file)')
    parser.add_argument('--test_path', required=True,
                        help='Path to test data file')
    parser.add_argument('--config', default='configs/defaults.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', default='ablation_outputs',
                        help='Directory to save ablation results')
    
    # Ablation parameters
    parser.add_argument('--explain_methods', nargs='+', 
                        choices=['rules', 'llm', 'hybrid', 'shap_only'],
                        default=['rules', 'llm'], 
                        help='Explanation methods to test')
    parser.add_argument('--llm_providers', nargs='+',
                        choices=['stub', 'ollama', 'huggingface','openrouter'],
                        default=['stub'], 
                        help='LLM providers to test (only used when explain includes llm/hybrid)')
    parser.add_argument('--llm_inputs', nargs='+',
                        choices=['label', 'label+features', 'label+shap', 'full'],
                        default=['full'], 
                        help='LLM input configurations to test')
    parser.add_argument('--llm_models', nargs='+', default=[None],
                        help='LLM models to test (None for default)')
    parser.add_argument('--enable_judge', action='store_true',
                        help='Enable LLM-as-judge evaluation for all experiments')
    parser.add_argument('--api_key', default=None,
                        help='API key for LLM provider (if required)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare base command
    base_cmd = [
        'python', 'pipeline/eval_model.py',
        '--model_path', args.model_path,
        '--test_path', args.test_path,
        '--config', args.config,
        '--output_dir', args.output_dir,
        '--api_key', args.api_key
    ]
    
    if args.enable_judge:
        base_cmd.append('--judge')
    
    # Track experiment results
    results = []
    experiment_count = 0
    
    # Generate all combinations for ablation
    for explain_method in args.explain_methods:
        if explain_method in ['llm', 'hybrid']:
            # For LLM-based methods, test different providers and configurations
            for provider, llm_input, llm_model in product(args.llm_providers, args.llm_inputs, args.llm_models):
                experiment_count += 1
                exp_name = f"exp_{experiment_count}_{explain_method}_{provider}_{llm_input}"
                if llm_model:
                    exp_name += f"_{llm_model.replace('/', '_').replace('-', '_')}"
                
                cmd = base_cmd + [
                    '--explain', explain_method,
                    '--llm_provider', provider,
                    '--ablation_llm_inputs', llm_input,
                    '--experiment_name', exp_name,
                ]
                
                if llm_model:
                    cmd.extend(['--llm_model', llm_model, '--api_key', args.api_key])
                
                description = f"Experiment {experiment_count}: {explain_method} with {provider} provider, {llm_input} inputs"
                if llm_model:
                    description += f", model {llm_model}"
                
                success = run_command(cmd, description)
                results.append({
                    'experiment_id': experiment_count,
                    'experiment_name': exp_name,
                    'explain_method': explain_method,
                    'llm_provider': provider if explain_method in ['llm', 'hybrid'] else None,
                    'llm_inputs': llm_input if explain_method in ['llm', 'hybrid'] else None,
                    'llm_model': llm_model if explain_method in ['llm', 'hybrid'] else None,
                    'success': success
                })
        else:
            # For non-LLM methods (rules, shap_only), just run once
            experiment_count += 1
            exp_name = f"exp_{experiment_count}_{explain_method}"
            
            cmd = base_cmd + [
                '--explain', explain_method,
                '--experiment_name', exp_name
            ]
            
            description = f"Experiment {experiment_count}: {explain_method}"
            success = run_command(cmd, description)
            
            results.append({
                'experiment_id': experiment_count,
                'experiment_name': exp_name,
                'explain_method': explain_method,
                'llm_provider': None,
                'llm_inputs': None,
                'llm_model': None,
                'success': success
            })
    
    # Save experiment summary
    summary = {
        'ablation_config': {
            'model_path': args.model_path,
            'test_path': args.test_path,
            'output_dir': args.output_dir,
            'explain_methods': args.explain_methods,
            'llm_providers': args.llm_providers,
            'llm_inputs': args.llm_inputs,  
            'llm_models': args.llm_models,
            'judge_enabled': args.enable_judge
        },
        'experiments': results,
        'summary': {
            'total_experiments': experiment_count,
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success'])
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'ablation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETED")
    print('='*60)
    print(f"Total experiments: {experiment_count}")
    print(f"Successful: {summary['summary']['successful']}")
    print(f"Failed: {summary['summary']['failed']}")
    print(f"Results saved in: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    
    if summary['summary']['failed'] > 0:
        print("\nFailed experiments:")
        for result in results:
            if not result['success']:
                print(f"  - {result['experiment_name']}: {result['explain_method']}")


if __name__ == '__main__':
    main()