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
    # Hybrid scoring.
    if step == Step.HYBRID:
        table = []
        for t in range(4):
            table.append(kwargs['global_scores'][t] * kwargs['local_scores'][t])
        return table

    images = dict.fromkeys(kwargs['keys'])
    masks = dict.fromkeys(kwargs['keys'])

    # Extraction.
    if step == Step.EXTRACT:
        for key, image in zip(kwargs['keys'],
                              kwargs['images']):
            images[key], masks[key] = \
                extract(image, kwargs['kernel'], kwargs['thr'])
        return images, masks

    # Registration.
    if step == Step.REGISTER:
        for key, image, label in zip(kwargs['keys'],
                                     kwargs['images'],
                                     kwargs['masks']):
            images[key], masks[key] = \
                register(image, label, kwargs['size'])
        return images, masks

    n = len(kwargs['keys'])
    images = dict.fromkeys(kwargs['keys'])
    models = dict.fromkeys(kwargs['keys'])
    labels = dict.fromkeys(kwargs['keys'])
    table1 = np.zeros((n, n))
    table2 = np.zeros((n, n))
    table3 = np.zeros((n, n))
    table4 = np.zeros((n, n))
    mask1: np.ndarray
    mask2: np.ndarray
    label1: np.ndarray
    label2: np.ndarray

    # Global GMM.
    if step == Step.GLOBAL_GMM:
        # Image processing.
        for key, image, mask in zip(kwargs['keys'],
                                    kwargs['images'],
                                    kwargs['masks']):
            images[key], masks[key], labels[key], models[key] = \
                global_gmm(image, mask, kwargs['nok'], kwargs['patch'])
        # Calculating global scores.
        for i, mask1, label1 in zip(range(n),
                                    masks.values(),
                                    labels.values()):
            for j, mask2, label2 in zip(range(n),
                                        masks.values(),
                                        labels.values()):
                if i == j:
                    table1[i, j] = table2[i, j] = table3[i, j] = table4[i, j] = 1
                elif i > j:
                    table1[i, j] = table1[j, i]
                    table2[i, j] = table2[j, i]
                    table3[i, j] = table3[j, i]
                    table4[i, j] = table4[j, i]
                else:
                    table1[i, j], table2[i, j], table3[i, j], table4[i, j] = \
                        global_gmm_compare(mask1, mask2, label1, label2)
        table1 = pd.DataFrame(table1, columns=kwargs['keys'], index=kwargs['keys'])
        table2 = pd.DataFrame(table2, columns=kwargs['keys'], index=kwargs['keys'])
        table3 = pd.DataFrame(table3, columns=kwargs['keys'], index=kwargs['keys'])
        table4 = pd.DataFrame(table4, columns=kwargs['keys'], index=kwargs['keys'])
        return images, masks, labels, models, [table1, table2, table3, table4]

    # Local GMM.
    if step == Step.LOCAL_GMM:
        # Image processing.
        for key, image, label, model in zip(kwargs['keys'],
                                            kwargs['images'],
                                            kwargs['labels'],
                                            kwargs['models']):
            images[key], labels[key], models[key] = \
                local_gmm(image, label, model, kwargs['nok'])
        # Calculating local scores.
        for i, label1, model1 in zip(range(n),
                                     labels.values(),
                                     models.values()):
            for j, label2, model2 in zip(range(n),
                                         labels.values(),
                                         models.values()):
                if i > j:
                    table1[i, j] = table1[j, i]
                    table2[i, j] = table2[j, i]
                    table3[i, j] = table3[j, i]
                    table4[i, j] = table4[j, i]
                elif max(table.iloc[i, j] for table in kwargs['scores']) < kwargs['cutoff']:
                    table1[i, j] = 0
                    table2[i, j] = 0
                    table3[i, j] = 0
                    table4[i, j] = 0
                else:
                    table1[i, j], table2[i, j], table3[i, j], table4[i, j] = \
                        local_gmm_compare(label1, label2, model1['mean'], model2['mean'])
                    if i == j:
                        table2[i, j] = table3[i, j] = table4[i, j] = table1[i, j]
        table1 = normalize(table1, norm='max', axis=0)
        table2 = normalize(table2, norm='max', axis=0)
        table3 = normalize(table3, norm='max', axis=0)
        table4 = normalize(table4, norm='max', axis=0)
        table1 = pd.DataFrame(table1, columns=kwargs['keys'], index=kwargs['keys'])
        table2 = pd.DataFrame(table2, columns=kwargs['keys'], index=kwargs['keys'])
        table3 = pd.DataFrame(table3, columns=kwargs['keys'], index=kwargs['keys'])
        table4 = pd.DataFrame(table4, columns=kwargs['keys'], index=kwargs['keys'])
        return images, labels, models, [table1, table2, table3, table4]
