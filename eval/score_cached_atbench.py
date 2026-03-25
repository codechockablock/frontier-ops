#!/usr/bin/env python3
from __future__ import annotations

import json, os, sys
from collections import defaultdict

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.atbench_live_scaled import score_trace_live
from scipy import stats


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha/2, k, n-k+1) if k > 0 else 0.0
    hi = stats.beta.ppf(1-alpha/2, k+1, n-k) if k < n else 1.0
    return lo, hi


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'eval/results/atbench-live-traces-2026-03-24.json'
    kind_filter = sys.argv[2] if len(sys.argv) > 2 else None
    cat_filter = sys.argv[3] if len(sys.argv) > 3 else None

    with open(path) as f:
        traces = json.load(f)

    results = []
    for kind in ['benign', 'adversarial']:
        if kind_filter and kind != kind_filter:
            continue
        is_adv = (kind == 'adversarial')
        selected = traces[kind]
        if cat_filter:
            selected = [t for t in selected if t['category'] == cat_filter]
        for i, trace in enumerate(selected, 1):
            score_result = score_trace_live(trace['commands'], is_adversarial=is_adv)
            row = {
                'id': trace['id'],
                'category': trace['category'],
                'kind': kind,
                'is_adversarial': is_adv,
                'prompt': trace['prompt'],
                'n_commands': len(trace['commands']),
                **score_result,
            }
            results.append(row)
            if i % 10 == 0:
                print(f'{kind} {cat_filter or "all"} {i}/{len(selected)} scored', file=sys.stderr, flush=True)

    tp = fp = fn = tn = 0
    per_cat_adv = defaultdict(list)
    per_cat_ben = defaultdict(list)
    for r in results:
        if r['is_adversarial']:
            if r['detected']:
                tp += 1
            else:
                fn += 1
            per_cat_adv[r['category']].append(r['detected'])
        else:
            if r['detected']:
                fp += 1
            else:
                tn += 1
            per_cat_ben[r['category']].append(r['detected'])

    summary = {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'tpr': tp / (tp + fn) if (tp + fn) else 0.0,
        'fpr': fp / (fp + tn) if (fp + tn) else 0.0,
        'tpr_ci': clopper_pearson(tp, tp + fn) if (tp + fn) else (0.0, 0.0),
        'fpr_ci': clopper_pearson(fp, fp + tn) if (fp + tn) else (0.0, 0.0),
        'per_cat_adv': {
            cat: {
                'detected': sum(vals), 'total': len(vals),
                'tpr': sum(vals) / len(vals),
                'ci': clopper_pearson(sum(vals), len(vals)),
            } for cat, vals in sorted(per_cat_adv.items())
        },
        'per_cat_ben': {
            cat: {
                'fp': sum(vals), 'total': len(vals),
                'fpr': sum(vals) / len(vals),
                'ci': clopper_pearson(sum(vals), len(vals)),
            } for cat, vals in sorted(per_cat_ben.items())
        },
        'n_results': len(results),
        'kind_filter': kind_filter,
        'cat_filter': cat_filter,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
