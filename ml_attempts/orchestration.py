"""Entry points for each attempt and the complete five-attempt workflow."""

import argparse
import importlib.util
import json
from pathlib import Path

from feature_processing.io_utils import PROJECT_DIR
from .data import load_context, validate_sources
from .searches import attempt_one, attempt_two, attempt_three, attempt_four, attempt_five


ATTEMPTS = ['01_logistic_representation', '02_logistic_target_encoded', '03_mse_sgd', '04_logistic_newton',
            '05_feature_recovery']


def load_configs():
    """Import each attempt's config.py and return {attempt name: CONFIG}."""
    configs = {}
    for name in ATTEMPTS:
        path = Path(__file__).resolve().parent / name / 'config.py'
        spec = importlib.util.spec_from_file_location('experiment_config_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        configs[name] = module.CONFIG
    return configs


def selection_path(context, attempt):
    """Path of an attempt's saved selection.json."""
    return context['root'] / 'ml_attempts' / attempt / 'outputs/selection.json'


def read_selection(context, attempt):
    """Load an attempt's selection if it exists and belongs to the current split."""
    path = selection_path(context, attempt)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['split'] != context['metadata']:
        return None
    return payload


def explore(context, configs, names):
    """Run the hyperparameter searches of the requested attempts (Attempt 1 first if needed)."""
    selections = {}
    benchmark = read_selection(context, ATTEMPTS[0])
    if ATTEMPTS[0] in names or benchmark is None or context.get('force_benchmark', False):
        benchmark = attempt_one(context, configs[ATTEMPTS[0]])
    selections[ATTEMPTS[0]] = benchmark
    if ATTEMPTS[1] in names:
        selections[ATTEMPTS[1]] = attempt_two(context, configs[ATTEMPTS[1]], benchmark)
    if ATTEMPTS[2] in names:
        selections[ATTEMPTS[2]] = attempt_three(context, configs[ATTEMPTS[2]], benchmark)
    if ATTEMPTS[3] in names:
        # The Attempt 2 GD selection is optional context for the solver comparison only.
        gd = {'te2': selections.get(ATTEMPTS[1]) or read_selection(context, ATTEMPTS[1])}
        selections[ATTEMPTS[3]] = attempt_four(context, configs[ATTEMPTS[3]], benchmark, configs[ATTEMPTS[1]], gd)
    if ATTEMPTS[4] in names:
        # The Attempt 4 selection is optional context for the report comparison only.
        newton = selections.get(ATTEMPTS[3]) or read_selection(context, ATTEMPTS[3])
        selections[ATTEMPTS[4]] = attempt_five(context, configs[ATTEMPTS[4]], benchmark, newton)
    validate_sources(context)
    return selections


def execute(context, configs, names, phase='all'):
    """Run the requested phases: search, 3-fold confirmation, then all-row refit and export."""
    selections = explore(context, configs, names) if phase in ('all', 'explore') else {}
    if phase in ('all', 'finish', 'confirm', 'final'):
        from .confirmation import confirm_finalist, final_refit, master_comparison
        for name in names:
            payload = selections.get(name) or read_selection(context, name)
            if payload is None:
                raise ValueError('Run exploration first for ' + name)
            if name == ATTEMPTS[2]:
                from .reporting import refine_mse_report
                refine_mse_report(context['root'] / 'ml_attempts' / name / 'outputs')
            if phase in ('all', 'confirm') or phase == 'finish' and not payload.get('confirmation_complete'):
                payload = confirm_finalist(context, payload)
            selections[name] = payload
        # Complete confirmation for every requested finalist before any test
        # file is opened for local inference/export.
        master_comparison(context)
        if phase in ('all', 'finish', 'final'):
            if any(not selections[name].get('confirmation_complete') for name in names):
                raise ValueError('Final refitting requires completed finalist confirmation')
            for name in names:
                payload = selections[name]
                if not payload.get('confirmation_complete'):
                    raise ValueError('Final refitting requires completed finalist confirmation')
                payload = final_refit(context, payload)
                selections[name] = payload
        master_comparison(context)
    validate_sources(context)
    return selections


def cli(name=None, argv=None):
    """Command-line entry point shared by run_all.py and each attempt's run.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'explore', 'finish', 'confirm', 'final'], default='all')
    parser.add_argument('--max-rows', type=int, default=0, help='0 uses all rows; sample runs share one deterministic subset')
    parser.add_argument('--max-logistic-updates', type=int)
    parser.add_argument('--max-mse-updates', type=int)
    args = parser.parse_args(argv)
    if args.max_rows < 0:
        parser.error('--max-rows must be nonnegative')
    configs = load_configs()
    if args.max_logistic_updates is not None:
        if args.max_logistic_updates < 1:
            parser.error('--max-logistic-updates must be positive')
        configs[ATTEMPTS[0]]['max_updates'] = args.max_logistic_updates
    if args.max_mse_updates is not None:
        if args.max_mse_updates < 1:
            parser.error('--max-mse-updates must be positive')
        config = configs[ATTEMPTS[2]]
        config['max_updates'] = args.max_mse_updates
        config['checkpoint_budgets'] = sorted(set([max(1, args.max_mse_updates // 4), args.max_mse_updates]))
    print('Loading dataset/ and reusing/reconstructing the deterministic shared split...', flush=True)
    context = load_context(PROJECT_DIR, max_rows=args.max_rows)
    context['force_benchmark'] = args.max_logistic_updates is not None
    print(f'Shared rows: {len(context["train"]):,} train / {len(context["validation"]):,} validation.', flush=True)
    execute(context, configs, ATTEMPTS if name is None else [name], args.phase)
    print('Requested experiment phase complete.', flush=True)
