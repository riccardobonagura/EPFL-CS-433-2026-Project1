"""Staged experiment orchestration; all model algorithms live in the augmented module."""

import copy

import numpy as np

from feature_processing import get_config
from .common import best, chart, new_log, report_footer, report_header, report_rows, run_candidate, select, write_csv
from .representations import base_recipes, prepare, save_representation


def logistic_params(config, penalty, gamma):
    return {'model': 'logistic', 'lambda_': float(penalty), 'gamma': float(gamma),
            'batch_size': config['batch_size'], 'max_iters': config['max_updates'],
            'early_stopping': True, 'patience': config['patience'], 'tol': config['tol'],
            'eval_every': config['eval_every'], 'seed': config['seed'],
            'lr_schedule': 'constant', 'lr_decay': 0.0}


def attempt_one(context, config):
    log = new_log(context, config['attempt'], config)
    winners = []
    for name, recipe in base_recipes().items():
        print('Preparing ' + name + '...', flush=True)
        bundle = prepare(context, recipe)
        save_representation(bundle['state'], log['output'] / 'representation_states' / name[:2])
        coarse = [run_candidate(log, bundle, recipe, logistic_params(config, penalty, config['initial_gamma']), 'lambda')
                  for penalty in config['lambda_values']]
        chosen_lambda = best(coarse)['lambda']
        for gamma in config['gamma_values']:
            if gamma != config['initial_gamma']:
                run_candidate(log, bundle, recipe, logistic_params(config, chosen_lambda, gamma), 'gamma')
        winners.append(best([row for row in log['rows'] if row['representation'] == name]))
        del bundle
    winner = best(winners)
    optional = [{'status': 'skipped', 'reason': 'Engineering is off: previous A4-to-A5 gain was only about 0.00011'}]
    if config['test_engineering']:
        recipe = copy.deepcopy(log['recipes'][winner['run_id']])
        recipe['name'] += ' + engineering'
        recipe['config']['engineer'] = True
        bundle = prepare(context, recipe)
        if bundle['state']['base'].engineered_feature_config.get('recipes'):
            row = run_candidate(log, bundle, recipe, log['params'][winner['run_id']], 'optional_engineering')
            optional = [row]
            if row['status'] == 'ok' and row['balanced_accuracy'] - winner['balanced_accuracy'] >= config['meaningful_gain']:
                winner = row
        else:
            optional = [{'status': 'skipped', 'reason': 'No validated engineered block exists for the winning base profile'}]
        del bundle
    payload = select(log, winner)
    write_csv(log['output'] / 'representation_results.csv', winners)
    write_csv(log['output'] / 'lambda_results.csv', [r for r in log['rows'] if r['stage'] == 'lambda'])
    write_csv(log['output'] / 'gamma_results.csv', [r for r in log['rows'] if r['stage'] == 'gamma'])
    write_csv(log['output'] / 'optional_engineering_ablation.csv', optional)
    chart(log['output'] / 'figures/representation_vs_balanced_accuracy.png', 'Tuned logistic representations',
          [r['representation'][:2] for r in winners],
          {'Default threshold BA': [r['default_balanced_accuracy'] for r in winners],
           'Selected threshold BA': [r['balanced_accuracy'] for r in winners]}, 'Representation', 'Balanced accuracy', unit_interval=True)
    chart(log['output'] / 'figures/lambda_vs_balanced_accuracy.png', 'Coarse L2 searches at gamma 0.03',
          np.log10(config['lambda_values']),
          {r['representation'][:2]: [row['balanced_accuracy'] if row['status'] == 'ok' else np.nan
                                   for row in log['rows'] if row['representation'] == r['representation'] and row['stage'] == 'lambda']
           for r in winners}, 'log10(lambda)', 'Selected validation balanced accuracy', unit_interval=True)
    r0, r1, r2 = winners
    lines = report_header('Attempt 1: tuned logistic representation comparison', log) + report_rows(winners)
    lines += ['', '## Answers', '',
              f'- **Does A0 remain strongest?** The selected benchmark is **{winner["representation"]}**, '
              f'BA={winner["balanced_accuracy"]:.3%}, F1={winner["f1"]:.3%}. '
              + ('A0 remains the strongest tested base.' if winner['representation'].startswith('R0') else 'A0 was beaten in this exploratory search.'),
              f'- **Different optimization?** R0 selected lambda={r0["lambda"]}, gamma={r0["gamma"]}; '
              f'curated selected lambda={r2["lambda"]}, gamma={r2["gamma"]}. All tested budgets/stops are in all_runs.csv.',
              '- **Optimization versus calibration?** The two metric columns separate default and tuned thresholds.',
              '  Staged retuning changes both fitting and calibration; it cannot uniquely decompose earlier gaps.',
              f'- **Missingness tradeoff?** R1 minus R0: BA {r1["balanced_accuracy"] - r0["balanced_accuracy"]:+.5f}, '
              f'F1 {r1["f1"] - r0["f1"]:+.5f}, recall {r1["recall"] - r0["recall"]:+.5f}.',
              f'- **Curated information sacrifice?** Curated minus minimal BA: {r2["balanced_accuracy"] - r0["balanced_accuracy"]:+.5f}.',
              '  Curated deliberately excludes semantically uncertain numeric/medium categories and canonical alternatives;',
              '  this is a conservatism-versus-information experiment, not proof any individual excluded source is useless.',
              '- **Engineering?** ' + ('A single explicit comparison was requested; see optional_engineering_ablation.csv.'
                                      if config['test_engineering'] else 'Skipped by default; the earlier gain did not justify automatically expanding the search.'),
              f'- **Benchmark for Attempts 2/3:** {winner["representation"]}, with its independently selected model settings.',
              '', 'The lambda grid is evaluated at one fixed gamma, then only gamma is varied around the best lambda.',
              'There is no full Cartesian grid. Successful/failed configurations and actual/best update counts are retained.']
    (log['output'] / 'experiment_report.md').write_text('\n'.join(lines + report_footer(log)), encoding='utf-8')
    return payload


def attempt_two(context, config, benchmark):
    log = new_log(context, config['attempt'], config)
    base_params = benchmark['model_params']
    penalties = [base_params['lambda_'] / 10, base_params['lambda_'], base_params['lambda_'] * 10]
    compact = {'name': 'TE1 curated compact', 'kind': 'standard', 'config': get_config('compact_target_encoded', target_alpha=config['alpha'])}
    recovery = copy.deepcopy(benchmark['recipe'])
    recovery.update(name='TE2 selective recovery', kind='recovery', recovery=config, alpha=config['alpha'])
    winners = []
    for recipe in (compact, recovery):
        print('Preparing ' + recipe['name'] + '...', flush=True)
        bundle = prepare(context, recipe)
        save_representation(bundle['state'], log['output'] / 'representation_states' / recipe['name'][:3])
        for penalty in penalties:
            params = dict(base_params, lambda_=float(penalty))
            run_candidate(log, bundle, recipe, params, 'lambda')
        winners.append(best([row for row in log['rows'] if row['representation'] == recipe['name']]))
        del bundle
    leading = best(winners)
    competitive = leading['balanced_accuracy'] >= benchmark['best_run']['balanced_accuracy'] - config['competitive_margin']
    if competitive:
        chosen_recipe = log['recipes'][leading['run_id']]
        for alpha in config['competitive_alpha_values']:
            recipe = copy.deepcopy(chosen_recipe)
            if recipe['kind'] == 'recovery':
                recipe['alpha'] = alpha
            else:
                recipe['config']['target_alpha'] = alpha
            bundle = prepare(context, recipe)
            row = run_candidate(log, bundle, recipe, log['params'][leading['run_id']], 'alpha')
            row['target_alpha'] = alpha
            del bundle
    for row in log['rows']:
        row.setdefault('target_alpha', config['alpha'])
    winner = best(log['rows'])
    payload = select(log, winner)
    representation_winners = [best([r for r in log['rows'] if r['representation'] == recipe['name']]) for recipe in (compact, recovery)]
    write_csv(log['output'] / 'all_runs.csv', log['rows'])
    write_csv(log['output'] / 'validation_metrics.csv', log['rows'])
    write_csv(log['output'] / 'representation_results.csv', representation_winners)
    write_csv(log['output'] / 'lambda_results.csv', [r for r in log['rows'] if r['stage'] == 'lambda'])
    write_csv(log['output'] / 'target_encoding_alpha_results.csv', log['rows'])
    comparisons = [benchmark['best_run']] + representation_winners
    write_csv(log['output'] / 'memory_comparison.csv', comparisons)
    chart(log['output'] / 'figures/target_encoding_comparison.png', 'Target encoding versus actual tuned logistic benchmark',
          ['BASE', 'TE1', 'TE2'], {'Selected BA': [r['balanced_accuracy'] for r in comparisons],
                                  'Selected F1': [r['f1'] for r in comparisons]}, 'Representation', 'Validation metric', unit_interval=True)
    chart(log['output'] / 'figures/feature_count_vs_performance.png', 'Representation size and validation performance',
          [r['feature_count'] for r in comparisons], {'Selected BA': [r['balanced_accuracy'] for r in comparisons]},
          'Output feature count excluding intercept', 'Balanced accuracy', unit_interval=True, connect=False)
    alpha_rows = [r for r in log['rows'] if r['representation'] == winner['representation'] and r['lambda'] == winner['lambda'] and r['status'] == 'ok']
    alpha_rows = sorted(alpha_rows, key=lambda row: row['target_alpha'])
    chart(log['output'] / 'figures/alpha_vs_balanced_accuracy.png', 'Only competitive variants receive a smoothing check',
          [r['target_alpha'] for r in alpha_rows], {'Selected BA': [r['balanced_accuracy'] for r in alpha_rows]},
          'Target encoding alpha', 'Balanced accuracy', unit_interval=True)
    baseline = benchmark['best_run']
    gain, added = winner['balanced_accuracy'] - baseline['balanced_accuracy'], winner['feature_count'] - baseline['feature_count']
    materially_better = gain >= config['meaningful_gain']
    compact_equivalent = abs(gain) <= config['competitive_margin'] and winner['feature_count'] <= 0.8 * baseline['feature_count']
    lines = report_header('Attempt 2: selective information recovery with OOF target encoding', log)
    lines += report_rows(comparisons) + ['', '## Answers', '',
        f'- **Information recovery / tuned minimal benchmark:** selected TE minus actual Attempt 1 benchmark BA is {gain:+.5f}.',
        f'  Material-gain criterion (predeclared {config["meaningful_gain"]:.3f} absolute BA): {materially_better}.',
        f'  Similar performance with >=20% fewer dimensions: {compact_equivalent}. Tiny gains are not declared success.',
        f'- **Selective versus curated target encoding:** TE2 BA {representation_winners[1]["balanced_accuracy"]:.5f} '
        f'versus TE1 {representation_winners[0]["balanced_accuracy"]:.5f}.',
        '- **Medium groups tested:** ' + (representation_winners[1]['recovered_sources'] or 'none met the training-only support rules') + '.',
        '  The explicit priority list is filtered by curated unused-medium status, support and quarantine.',
        '  Only the block is tested; individual group usefulness is not identified by this comparison.',
        f'- **Dimensionality / memory:** chosen TE {winner["feature_count"]} columns, {winner["matrix_memory_mb"]:.1f} MiB; '
        f'benchmark {baseline["feature_count"]} columns, {baseline["matrix_memory_mb"]:.1f} MiB.',
        f'  Added dimensions relative to benchmark: {added}; gain per added column is '
        + (f'{gain / added:+.6g} BA.' if added > 0 else 'not applicable (no positive dimensional expansion).'),
        '- **Smoothing:** ' + ('The competitive variant received alpha 5/20/100 checks at one selected lambda.' if competitive
                               else 'Neither variant was competitive enough to justify an expanded alpha search; only alpha=20 was tested.'),
        f'  Selected alpha: {winner["target_alpha"]}; at its selected lambda, observed alpha-check BA range is '
        f'{max(row["balanced_accuracy"] for row in alpha_rows) - min(row["balanced_accuracy"] for row in alpha_rows):.6g}.',
        '- **Complexity justified?** ' + ('This exploratory evidence merits confirmation.' if materially_better or compact_equivalent
                                         else 'No demonstrated material benefit or compact equivalent yet; preserve the simpler benchmark.'),
        '', 'TE1 uses the unengineered compact profile, not an assumed winner from cumulative A6.',
        'TE2 adds at most eight predeclared, training-supported nominal sources to the winning base. For a minimal',
        'base, raw numeric-code representations may coexist with added TE columns; this explicitly tests extra',
        'nonlinear categorical information and is not described as restoring previously absent raw columns.',
        'Every training TE value is OOF, including fold-specific vocabulary/rare groups/prior. Full-training',
        'encoders are used only for validation/test; targets never enter validation preprocessing.']
    lines += ['', 'The new representations also receive a lambda neighborhood around the benchmark. A gain can',
              'therefore reflect regularization retuning as well as encoding; this is not an isolated causal',
              'feature-block ablation. A boundary lambda winner suggests a future matched-base control or',
              'neighbor check, not proof of an encoding benefit. The present search stays deliberately bounded.']
    (log['output'] / 'experiment_report.md').write_text('\n'.join(lines + report_footer(log)), encoding='utf-8')
    return payload


def attempt_three(context, config, benchmark):
    log = new_log(context, config['attempt'], config)
    recipe = copy.deepcopy(benchmark['recipe'])
    recipe['name'] = 'S0 ' + recipe['name']
    bundle = prepare(context, recipe)
    save_representation(bundle['state'], log['output'] / 'representation_states/S0')
    def parameters(gamma, batch, schedule='constant'):
        return {'model': 'mse', 'gamma': float(gamma), 'batch_size': batch,
                'max_iters': config['max_updates'], 'early_stopping': True,
                'patience': config['patience'], 'tol': config['tol'], 'eval_every': config['eval_every'],
                'seed': config['seed'], 'lr_schedule': schedule,
                'lr_decay': config['lr_decay'] if schedule == 'inverse' else 0.0}
    learning = [run_candidate(log, bundle, recipe, parameters(gamma, config['initial_batch_size']), 'learning_rate')
                for gamma in config['gamma_values']]
    chosen_gamma = best(learning)['gamma']
    for batch in config['batch_sizes']:
        if batch == config['initial_batch_size']:
            continue
        row = run_candidate(log, bundle, recipe, parameters(chosen_gamma, batch), 'batch_size')
        if batch == 1 and row['status'] != 'ok':
            bound = 1.0 / max(float(np.einsum('ij,ij->i', bundle['train'], bundle['train']).max()), 1.0)
            run_candidate(log, bundle, recipe, parameters(min(chosen_gamma, bound), 1), 'batch_size_safe_bound')
    chosen = best(log['rows'])
    run_candidate(log, bundle, recipe, parameters(chosen['gamma'], chosen['batch_size'], config['decaying_schedule']), 'schedule')
    chosen = best(log['rows'])
    prefix_params = log['params'][chosen['run_id']]
    for budget in config['checkpoint_budgets']:
        if budget != prefix_params['max_iters']:
            run_candidate(log, bundle, recipe, dict(prefix_params, max_iters=budget), 'classification_checkpoint')
    winner = best(log['rows'])
    payload = select(log, winner)
    write_csv(log['output'] / 'batch_size_results.csv', [r for r in log['rows'] if r['stage'].startswith('batch_size') or r['run_id'] == best(learning)['run_id']])
    write_csv(log['output'] / 'learning_rate_results.csv', [r for r in log['rows'] if r['stage'] in {'learning_rate', 'schedule'}])
    trajectory = [r for r in log['rows'] if r['status'] == 'ok' and r['gamma'] == prefix_params['gamma'] and r['batch_size'] == prefix_params['batch_size'] and r['lr_schedule'] == prefix_params['lr_schedule']]
    trajectory.sort(key=lambda row: row['max_updates'])
    write_csv(log['output'] / 'convergence_results.csv', trajectory)
    write_csv(log['output'] / 'optional_representation_ablation.csv', [{'status': 'skipped', 'reason': 'Use Attempt 1 winning recipe; no automatic representation/engineering search'}])
    chart(log['output'] / 'figures/updates_vs_mse.png', 'Independent budget prefixes: selected validation-MSE checkpoints',
          [r['max_updates'] for r in trajectory], {'Training half-MSE': [r['train_loss'] for r in trajectory],
                                                 'Best validation half-MSE': [r['best_validation_loss'] for r in trajectory]},
          'Requested gradient-update budget', 'Half-MSE data loss')
    chart(log['output'] / 'figures/updates_vs_balanced_accuracy.png', 'Classification at independently replayed budget prefixes',
          [r['max_updates'] for r in trajectory], {'Default threshold BA': [r['default_balanced_accuracy'] for r in trajectory],
                                                 'Selected threshold BA': [r['balanced_accuracy'] for r in trajectory]},
          'Requested update budget (not necessarily returned checkpoint)', 'Balanced accuracy', unit_interval=True)
    batch_rows = [best([r for r in log['rows'] if r['batch_size'] == size and r['status'] == 'ok'])
                  for size in config['batch_sizes'] if any(r['batch_size'] == size and r['status'] == 'ok' for r in log['rows'])]
    chart(log['output'] / 'figures/batch_size_vs_performance.png', 'Best observed settings per tested batch size',
          [r['batch_size'] for r in batch_rows], {'Selected BA': [r['balanced_accuracy'] for r in batch_rows]},
          'Batch size', 'Balanced accuracy', unit_interval=True)
    chart(log['output'] / 'figures/runtime_vs_performance.png', 'Successful MSE candidates: cost and classification performance',
          [r['training_runtime_seconds'] for r in log['rows'] if r['status'] == 'ok'],
          {'Selected BA': [r['balanced_accuracy'] for r in log['rows'] if r['status'] == 'ok']},
          'Training wall-clock seconds', 'Balanced accuracy', unit_interval=True, connect=False)
    baseline = benchmark['best_run']
    lines = report_header('Attempt 3: early-stopped MSE-SGD on the winning logistic representation', log)
    lines += report_rows([baseline, winner]) + ['', '## Answers', '',
        f'- **Match tuned logistic?** MSE minus logistic selected BA: {winner["balanced_accuracy"] - baseline["balanced_accuracy"]:+.5f}; '
        f'F1 difference {winner["f1"] - baseline["f1"]:+.5f}. Same underlying recipe: {benchmark["recipe"]["name"]}.',
        f'- **Best observed batch/gamma/schedule:** {winner["batch_size"]} / {winner["gamma"]:.5g} / {winner["lr_schedule"]}.',
        '- **Gamma sensitivity and batch sizes:** full staged results, including nonfinite attempts and conservative',
        '  training-norm fallback for failed size-one runs, are saved. There is no Cartesian sweep.',
        '- **MSE versus classification trajectory:** see convergence_results.csv and the prefix figures. Each prefix',
        '  replays from zero with the same local RNG and schedule, retaining that prefix fit\'s best validation-MSE',
        '  weights. Classification then selects among these lightweight saved checkpoints. These are not fabricated',
        '  dense per-update classification histories or warm-started RNG-reset chunks.',
        f'- **Early stopping:** always enabled during exploratory fitting; chosen run used {winner["actual_updates"]} updates,',
        f'  returning checkpoint {winner["best_iteration"]} with stop reason {winner["stop_reason"]}.',
        f'- **Threshold contribution:** default BA {winner["default_balanced_accuracy"]:.5f} -> selected {winner["balanced_accuracy"]:.5f}.',
        f'- **Runtime comparison:** selected MSE call {winner["training_runtime_seconds"]:.3f}s; tuned logistic call',
        f'  {baseline["training_runtime_seconds"]:.3f}s. Different budgets/early stopping can drive this comparison;',
        '  the separate all-training final-refit timers should also be consulted.',
        '', 'Size-one SGD budgets can be much less than one data pass; inspect training_data_passes before',
        'interpreting a low-cost underfit score as a converged model advantage. The final full-data fit uses',
        'a fixed, classification-selected stopping budget learned without test data, since no validation rows',
        'remain after all training rows are included. There is no automatic engineered-feature block.']
    (log['output'] / 'experiment_report.md').write_text('\n'.join(lines + report_footer(log)), encoding='utf-8')
    from .reporting import refine_mse_report
    refine_mse_report(log['output'])
    del bundle
    return payload
