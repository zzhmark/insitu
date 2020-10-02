import numpy as np
from typing import Any

import pandas as pd
from sklearn.preprocessing import normalize

from utils import Step
from .preprocessing import extract, register
from .GMM import global_gmm, local_gmm
from .scoring import global_gmm_compare, local_gmm_compare


def batch_apply(step: Step, **kwargs) -> Any:
    """
    :param step: enum for steps
    :param kwargs: key word arguments
    :return: depending on step
    """
    images = {}
    masks = {}

    if step == Step.EXTRACT:
        for key, image in kwargs['images'].items():
            images[key], masks[key] = \
                extract(image, kwargs['kernel'], kwargs['thr'])
        return images, masks

    if step == Step.REGISTER:
        for key, image, label in zip(kwargs['keys'],
                                     kwargs['images'],
                                     kwargs['masks']):
            images[key], masks[key] = \
                register(image, label, kwargs['size'])
        return images, masks

    images = {}
    models = {}
    labels = {}

    if step == Step.GLOBAL_GMM:
        for key, image, mask in zip(kwargs['keys'],
                                    kwargs['images'],
                                    kwargs['masks']):
            images[key], masks[key], labels[key], models[key] = \
                global_gmm(image, mask, kwargs['nok'], kwargs['patch'])
        return images, masks, labels, models

    if step == Step.LOCAL_GMM:
        for key, image, label, model in zip(kwargs['keys'],
                                            kwargs['images'],
                                            kwargs['labels'],
                                            kwargs['models']):
            images[key], labels[key], models[key] = \
                local_gmm(image, label, model, kwargs['nok'])
        return images, labels, models

    n = len(kwargs['keys'])

    if step == Step.HYBRID:
        # Global scoring.
        global_score_table = np.zeros((n, n))
        for i, mask1, label1 in zip(range(n),
                                    kwargs['masks'].values(),
                                    kwargs['global_labels'].values()):
            for j, mask2, label2 in zip(range(n),
                                        kwargs['masks'].values(),
                                        kwargs['global_labels'].values()):
                if i > j:
                    global_score_table[i, j] = global_score_table[j, i]
                else:
                    global_score_table[i, j] = global_gmm_compare(mask1, mask2, label1, label2)
        global_score_table = pd.DataFrame(global_score_table,
                                          columns=kwargs['keys'],
                                          index=kwargs['keys'])
        # Local scoring.
        local_score_table = np.zeros((n, n))
        for i, label1, model1 in zip(range(n),
                                     kwargs['local_labels'].values(),
                                     kwargs['local_models'].values()):
            for j, label2, model2 in zip(range(n),
                                         kwargs['local_labels'].values(),
                                         kwargs['local_models'].values()):
                if i > j:
                    local_score_table[i, j] = local_score_table[j, i]
                elif kwargs['scores'].iloc[i, j] < kwargs['cutoff']:
                    local_score_table[i, j] = 0
                else:
                    local_score_table[i, j] = \
                        local_gmm_compare(label1, label2, model1['mean'], model2['mean'])
        local_score_table = normalize(local_score_table, norm='max', axis=0)
        local_score_table = pd.DataFrame(local_score_table,
                                         columns=kwargs['keys'],
                                         index=kwargs['keys'])
        # Hybrid scoring.
        hybrid_score_table = global_score_table * local_score_table
        return global_score_table, local_score_table, hybrid_score_table
