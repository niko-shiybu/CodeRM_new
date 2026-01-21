"""
Generate a summary table for Table 2 results.
"""

import json
import os
import argparse
from collections import defaultdict


def load_results(results_dir, benchmark, sol_model):
    """Load all results for a given benchmark and solution model."""
    results = {}
    
    ut_models = ['llama3-8b', 'llama3-70b', 'coderm-8b']
    
    for ut_model in ut_models:
        result_file = os.path.join(results_dir, f"{benchmark}_{sol_model}_{ut_model}.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[ut_model] = json.load(f)
        else:
            print(f"Warning: Results not found for {ut_model}")
    
    return results


def print_table(results, mode='individual'):
    """Print a formatted table of results."""
    print(f"\n{'='*80}")
    print(f"Quality of {'Individual' if mode == 'individual' else 'Multiple'} Unit Tests")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Acc (↑)':<12} {'F1 (↑)':<12} {'FAR (↓)':<12} {'FRR (↓)':<12} {'Line Coverage (↑)':<18}")
    print("-" * 80)
    
    ut_models = ['llama3-8b', 'llama3-70b', 'coderm-8b']
    model_names = {
        'llama3-8b': 'Llama3.1-8B',
        'llama3-70b': 'Llama3.1-70B',
        'coderm-8b': 'CodeRM-8B (Ours)'
    }
    
    for ut_model in ut_models:
        if ut_model in results:
            data = results[ut_model].get(mode, {})
            model_name = model_names.get(ut_model, ut_model)
            
            acc = data.get('accuracy', 0.0)
            f1 = data.get('f1', 0.0)
            far = data.get('far', 0.0)
            frr = data.get('frr', 0.0)
            coverage = data.get('line_coverage', 0.0)
            
            print(f"{model_name:<20} {acc:<12.2f} {f1:<12.2f} {far:<12.2f} {frr:<12.2f} {coverage:<18.2f}")
        else:
            print(f"{model_names.get(ut_model, ut_model):<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<18}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Generate Table 2 summary")
    parser.add_argument('--results_dir', type=str, default='output/table2_results',
                       help='Directory containing result JSON files')
    parser.add_argument('--benchmark', type=str, default='humaneval+',
                       help='Benchmark name')
    parser.add_argument('--sol_model', type=str, default='llama3-8b',
                       help='Solution model name')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_dir, args.benchmark, args.sol_model)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    # Print individual unit test results
    print_table(results, mode='individual')
    
    # Print multiple unit test results
    print_table(results, mode='multiple')
    
    # Save to markdown file
    output_file = os.path.join(args.results_dir, f"table2_summary_{args.benchmark}.md")
    with open(output_file, 'w') as f:
        f.write(f"# Table 2: Quality of Unit Tests\n\n")
        f.write(f"Benchmark: {args.benchmark}\n")
        f.write(f"Policy Model: {args.sol_model}\n\n")
        
        f.write("## Quality of Individual Unit Tests\n\n")
        f.write("| Model | Acc (↑) | F1 (↑) | FAR (↓) | FRR (↓) | Line Coverage (↑) |\n")
        f.write("|-------|---------|--------|---------|---------|-------------------|\n")
        
        ut_models = ['llama3-8b', 'llama3-70b', 'coderm-8b']
        model_names = {
            'llama3-8b': 'Llama3.1-8B',
            'llama3-70b': 'Llama3.1-70B',
            'coderm-8b': 'CodeRM-8B (Ours)'
        }
        
        for ut_model in ut_models:
            if ut_model in results:
                data = results[ut_model].get('individual', {})
                model_name = model_names.get(ut_model, ut_model)
                acc = data.get('accuracy', 0.0)
                f1 = data.get('f1', 0.0)
                far = data.get('far', 0.0)
                frr = data.get('frr', 0.0)
                coverage = data.get('line_coverage', 0.0)
                f.write(f"| {model_name} | {acc:.2f} | {f1:.2f} | {far:.2f} | {frr:.2f} | {coverage:.2f} |\n")
        
        f.write("\n## Quality of Multiple Unit Tests\n\n")
        f.write("| Model | Acc (↑) | F1 (↑) | FAR (↓) | FRR (↓) | Line Coverage (↑) |\n")
        f.write("|-------|---------|--------|---------|---------|-------------------|\n")
        
        for ut_model in ut_models:
            if ut_model in results:
                data = results[ut_model].get('multiple', {})
                model_name = model_names.get(ut_model, ut_model)
                acc = data.get('accuracy', 0.0)
                f1 = data.get('f1', 0.0)
                far = data.get('far', 0.0)
                frr = data.get('frr', 0.0)
                coverage = data.get('line_coverage', 0.0)
                f.write(f"| {model_name} | {acc:.2f} | {f1:.2f} | {far:.2f} | {frr:.2f} | {coverage:.2f} |\n")
    
    print(f"\nSummary saved to: {output_file}")


if __name__ == '__main__':
    main()
