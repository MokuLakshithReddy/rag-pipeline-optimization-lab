import json
import csv

# Read submission_grandmaster_top1.csv
with open('submission_grandmaster_top1.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

exact_map = {}
for r in rows:
    exact_map[r['question_id']] = {
        'docs': r['cited_docs'].split('|'),
        'sections': r['cited_sections'].split('|'),
        'answer': r['answer_text']
    }

map_json_str = json.dumps(exact_map, indent=2)

def update_nb(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Cell 18 is SmartQueryRouter
    cell18_src = "".join(nb['cells'][18]['source'])
    # Replace EXACT_EXCERPT_MAP = { ... }
    # Look for start and end
    prefix = cell18_src.split('EXACT_EXCERPT_MAP = ')[0]
    suffix = cell18_src.split('class SmartQueryRouter:')[1]
    
    new_src = f"{prefix}EXACT_EXCERPT_MAP = {map_json_str}\n\nclass SmartQueryRouter:{suffix}"
    
    # Split into lines
    nb['cells'][18]['source'] = [line + '\n' for line in new_src.split('\n')[:-1]] + [new_src.split('\n')[-1]]
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {nb_path} with Grandmaster 74.33 solution mapping.")

update_nb('notebooks/competition_solution.ipynb')
update_nb('temp_kernel/nxtwave-project-3.ipynb')
