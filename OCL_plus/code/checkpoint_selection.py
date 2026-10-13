"""Pure, deterministic SOURCE-only selectors. No model, image, or target I/O."""
import math

RULE = 'source_hard_margin_v1'
SELECTION_STATEMENT = 'Checkpoint selection uses source discriminative retrieval metrics, not raw positive cosine.'


def source_only(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in str(key).lower() for word in ('satellite', 'target')):
                raise ValueError('Target/satellite fields are forbidden in source checkpoint selection')
            source_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            source_only(item)


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Selector requires finite metrics')
    return number


def micro_selection(real, closure, epoch):
    source_only([real, closure])
    if int(epoch) != epoch or epoch < 0:
        raise ValueError('Epoch must be a nonnegative integer')
    def average(key):
        return .5*(finite(real[key])+finite(closure[key]))
    return dict(rule=RULE, source_only=True, epoch=int(epoch),
                selection_margin=average('hard_margin'), selection_mean_margin=average('mean_margin'),
                selection_r1=average('r1'), selection_mrr=average('mrr'),
                old_positive_cosine_score=average('positive_mean'), real=dict(real), closure=dict(closure))


def micro_key(candidate):
    source_only(candidate)
    if candidate.get('rule') != RULE or candidate.get('source_only') is not True:
        raise ValueError('Missing explicit source selector provenance')
    return tuple(finite(candidate[k]) for k in ('selection_margin', 'selection_mean_margin', 'selection_r1', 'selection_mrr')) + (-int(candidate['epoch']),)


def select_checkpoint_micro(candidates):
    if not candidates:
        raise ValueError('No checkpoint candidates')
    return max(candidates, key=micro_key)


def select_checkpoint_source_retrieval(candidates, *, preregistered_tasks, ranking_metric='mrr', min_val_places=20):
    """Future formal source selection: mean R1 -> mean AP/MRR -> hard margin -> earlier.

    Each candidate explicitly declares source_only=True, validation_place_count,
    epoch and tasks={task_name: {r1, mrr (or map), hard_margin}}.
    Caller must provide the preregistered task set, not target-derived metrics.
    """
    if ranking_metric not in ('mrr', 'map') or not preregistered_tasks or len(set(preregistered_tasks)) != len(preregistered_tasks):
        raise ValueError('Specify distinct preregistered source tasks and mrr/map')
    if min_val_places < 2 or not candidates:
        raise ValueError('Invalid validation size or no candidates')
    def key(c):
        source_only(c)
        if c.get('source_only') is not True or c['validation_place_count'] < min_val_places:
            raise ValueError('Requires a sufficiently large source-only validation set')
        if set(c['tasks']) != set(preregistered_tasks):
            raise ValueError('Candidate tasks differ from preregistered source tasks')
        def mean(metric):
            return sum(finite(c['tasks'][t][metric]) for t in preregistered_tasks)/len(preregistered_tasks)
        return mean('r1'), mean(ranking_metric), mean('hard_margin'), -int(c['epoch'])
    return max(candidates, key=key)
