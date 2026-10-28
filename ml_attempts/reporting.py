"""Numerical answers drawn from saved experiment evidence, without new fitting."""

import csv
from pathlib import Path


def refine_mse_report(output):
    """Append the step-size and convergence discussion to the MSE-SGD report."""
    output = Path(output)
    with (output / 'all_runs.csv').open(newline='', encoding='utf-8') as stream:
        rows = [row for row in csv.DictReader(stream) if row['status'] == 'ok']
    with (output / 'convergence_results.csv').open(newline='', encoding='utf-8') as stream:
        trajectory = list(csv.DictReader(stream))
    learning = [row for row in rows if row['stage'] == 'learning_rate']
    lines = ['## Numerical interpretation of MSE diagnostics', '']
    if learning:
        values = [float(row['balanced_accuracy']) for row in learning]
        lines += [f'At the initial fixed batch/budget, the tested gammas span selected BA '
                  f'**{min(values):.3%}..{max(values):.3%}** (range {(max(values) - min(values)) * 100:.3f} percentage points).', '']
    declines = []
    for earlier, later in zip(trajectory, trajectory[1:]):
        if (float(later['train_loss']) < float(earlier['train_loss']) and
                (float(later['default_balanced_accuracy']) < float(earlier['default_balanced_accuracy']) or
                 float(later['balanced_accuracy']) < float(earlier['balanced_accuracy']))):
            declines.append((earlier, later))
    if declines:
        earlier, later = declines[0]
        lines += [f'The first sampled prefix pair showing lower training MSE with worse BA at the default or selected threshold is '
                  f'budget **{earlier["max_updates"]} -> {later["max_updates"]}**: loss '
                  f'{float(earlier["train_loss"]):.6g} -> {float(later["train_loss"]):.6g}, default BA '
                  f'{float(earlier["default_balanced_accuracy"]):.3%} -> {float(later["default_balanced_accuracy"]):.3%}. '
                  f'Selected-threshold BA changes {float(earlier["balanced_accuracy"]):.3%} -> '
                  f'{float(later["balanced_accuracy"]):.3%}.', '']
    else:
        lines += ['No adjacent sampled prefixes exhibited a lower-MSE/BA-decline pair; this run does not '
                  'locate the historical collapse point at the sampled budgets. The prefix coverage is deliberately limited.', '']
    lines += ['All candidates use validation-MSE early stopping. There is no controlled early-stopping-off fit, '
              'so its standalone causal benefit is not estimated. Classification primarily selects among the '
              'saved prefix checkpoints rather than automatically choosing the lowest-MSE configuration.', '']
    path = output / 'experiment_report.md'
    report = path.read_text(encoding='utf-8')
    marker = '## Numerical interpretation of MSE diagnostics'
    if marker not in report:
        report = report.replace('## Selection limits and completion', '\n'.join(lines) + '\n## Selection limits and completion')
        path.write_text(report, encoding='utf-8')
