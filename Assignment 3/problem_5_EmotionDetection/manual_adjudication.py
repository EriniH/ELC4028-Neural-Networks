import json
import os
import pandas as pd
import sys

# Model configurations
MODELS = [
    ('Gemini 2.5 Flash', 'gemini', 'gemini-2.5-flash', 'gemini_output'),
    ('GPT-4o-mini', 'GPT', 'gpt-4o-mini', 'gpt_output'),
    ('Fanar-C-2-27B', 'Fanar', 'Fanar-C-2-27B', 'fanar_output'),
    ('ALLaM-7B', 'ALLaM', 'ALLaM-7B-Instruct-preview', 'allam_output'),
    ('Jais-2-8B-Chat', 'Jais', 'Jais-2-8B-Chat', 'jais_output'),
]

DECISIONS_FILE = 'adjudication_decisions.json'

def load_mismatches():
    all_mismatches = []
    for model_name, model_dir, file_prefix, output_col in MODELS:
        for strategy in ['zero_shot', 'few_shot']:
            path = os.path.join(model_dir, f'{file_prefix}_{strategy}_results.jsonl')
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line.strip())
                    if not record.get('is_match', True):
                        all_mismatches.append({
                            'id': len(all_mismatches),
                            'model': model_name,
                            'strategy': strategy.replace('_', '-'),
                            'text': record.get('text', ''),
                            'gold_label': record.get('arabic_label', ''),
                            'english_label': record.get('english_label', ''),
                            'prediction': record.get(output_col, ''),
                            'output_col': output_col,
                        })
    return all_mismatches

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def recalculate_metrics(mismatches, decisions):
    print("\n--- Recalculating Metrics ---")
    accepted_ids = {int(k) for k, v in decisions.items() if v == 'accept'}
    
    results_rows = []
    for model_name, model_dir, file_prefix, output_col in MODELS:
        for strategy in ['zero_shot', 'few_shot']:
            path = os.path.join(model_dir, f'{file_prefix}_{strategy}_results.jsonl')
            if not os.path.exists(path):
                continue
                
            with open(path, 'r', encoding='utf-8') as f:
                records = [json.loads(line.strip()) for line in f]
                
            original_correct = sum(1 for r in records if r.get('is_match', False))
            
            accepted_for_this = sum(
                1 for m in mismatches
                if m['model'] == model_name
                and m['strategy'] == strategy.replace('_', '-')
                and m['id'] in accepted_ids
            )
            
            adjusted_correct = original_correct + accepted_for_this
            total = len(records)
            
            original_acc = original_correct / total * 100 if total > 0 else 0
            adjusted_acc = adjusted_correct / total * 100 if total > 0 else 0
            
            results_rows.append({
                'Model': model_name,
                'Strategy': strategy.replace('_', '-'),
                'Total': total,
                'Original Correct': original_correct,
                'Accepted Mismatches': accepted_for_this,
                'Adjusted Correct': adjusted_correct,
                'Original Accuracy (%)': round(original_acc, 1),
                'Adjusted Accuracy (%)': round(adjusted_acc, 1),
                'Improvement (%)': round(adjusted_acc - original_acc, 1),
            })
            
    df = pd.DataFrame(results_rows)
    df.to_csv('adjusted_results_summary.csv', index=False, encoding='utf-8-sig')
    print("✅ Saved detailed results to adjusted_results_summary.csv")
    
    print("\nLaTeX Ready Summary Table:")
    print("-" * 110)
    print(f"{'Model':<18} | ZS Orig | ZS Adj | ZS Δ  | FS Orig | FS Adj | FS Δ")
    print("-" * 110)
    for model_name, _, _, _ in MODELS:
        model_data = df[df['Model'] == model_name]
        zs = model_data[model_data['Strategy'] == 'zero-shot']
        fs = model_data[model_data['Strategy'] == 'few-shot']
        
        if not zs.empty and not fs.empty:
            zs = zs.iloc[0]
            fs = fs.iloc[0]
            print(f"{model_name:<18} | {zs['Original Accuracy (%)']:>6.1f}% | {zs['Adjusted Accuracy (%)']:>5.1f}% | +{zs['Improvement (%)']:>4.1f}% | "
                  f"{fs['Original Accuracy (%)']:>6.1f}% | {fs['Adjusted Accuracy (%)']:>5.1f}% | +{fs['Improvement (%)']:>4.1f}%")
    print("-" * 110)
    print("\nAll done!")

def main():
    mismatches = load_mismatches()
    if not mismatches:
        print("No mismatches found!")
        return

    decisions = {}
    if os.path.exists(DECISIONS_FILE):
        with open(DECISIONS_FILE, 'r', encoding='utf-8') as f:
            decisions = json.load(f)
            
    # Start at the first unreviewed item
    current_idx = 0
    for i in range(len(mismatches)):
        if str(mismatches[i]['id']) not in decisions:
            current_idx = i
            break
            
    while current_idx < len(mismatches):
        item = mismatches[current_idx]
        status = decisions.get(str(item['id']), 'unreviewed')
        
        clear_screen()
        accepted = sum(1 for v in decisions.values() if v == 'accept')
        rejected = sum(1 for v in decisions.values() if v == 'reject')
        remaining = len(mismatches) - len(decisions)
        
        print(f"--- Review Progress: {len(decisions)}/{len(mismatches)} ---")
        print(f"✅ Accepted: {accepted} | ❌ Rejected: {rejected} | ⏳ Remaining: {remaining}")
        print("="*80)
        print(f"Status:   [{status.upper()}]")
        print(f"Model:    {item['model']} ({item['strategy']})")
        print(f"Text:     {item['text']}")
        print("-" * 80)
        print(f"Gold:     {item['gold_label']} ({item['english_label']})")
        print(f"Pred:     {item['prediction']}")
        print("="*80)
        
        print("\nOptions:")
        print("  [a] Accept (Valid synonym/answer)")
        print("  [r] Reject (Incorrect)")
        print("  [s] Skip / Next unreviewed")
        print("  [p] Previous")
        print("  [q] Quit and recalculate metrics")
        
        choice = input("\nEnter choice (a/r/s/p/q): ").strip().lower()
        
        if choice == 'a':
            decisions[str(item['id'])] = 'accept'
            # Save state
            with open(DECISIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(decisions, f, ensure_ascii=False, indent=2)
            # Find next unreviewed
            current_idx += 1
            while current_idx < len(mismatches) and str(mismatches[current_idx]['id']) in decisions:
                current_idx += 1
        elif choice == 'r':
            decisions[str(item['id'])] = 'reject'
            # Save state
            with open(DECISIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(decisions, f, ensure_ascii=False, indent=2)
            # Find next unreviewed
            current_idx += 1
            while current_idx < len(mismatches) and str(mismatches[current_idx]['id']) in decisions:
                current_idx += 1
        elif choice == 's':
            current_idx += 1
            while current_idx < len(mismatches) and str(mismatches[current_idx]['id']) in decisions:
                current_idx += 1
        elif choice == 'p':
            current_idx = max(0, current_idx - 1)
        elif choice == 'q':
            break
            
    # End of review loop
    clear_screen()
    print(f"Review session ended. {len(decisions)}/{len(mismatches)} reviewed.")
    recalculate_metrics(mismatches, decisions)

if __name__ == "__main__":
    main()
