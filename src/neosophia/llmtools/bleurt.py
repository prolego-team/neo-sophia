"""
Wrapper around Bleurt.

Based on instructions here:
https://huggingface.co/lucadiliello/BLEURT-20-D12
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer

from typing import List, Tuple

import torch
import bleurt_pytorch as bp
from bleurt_pytorch.bleurt.tokenization_bleurt_sp import BleurtSPTokenizer


# recommended
BLEURT_20 = 'lucadiliello/BLEURT-20'

# smaller - ~680 MB on disk
BLEURT_20_D12 = 'lucadiliello/BLEURT-20-D12'  # smaller


def load_bleurt(
        model_name: str,
        cache_dir_path: str
        ) -> Tuple[
            bp.BleurtForSequenceClassification,
            BleurtSPTokenizer
        ]:
    """load a Pytorch Bleurt model"""

    # os.environ['TRANSFORMERS_CACHE'] = cache_dir_path

    # this was in the example, not sure what the point is
    # config = bp.BleurtConfig.from_pretrained(model_name)

    model = bp.BleurtForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir_path
    )
    tokenizer = bp.BleurtTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir_path
    )

    return model, tokenizer


def compare(
        model: bp.BleurtForSequenceClassification,
        tokenizer: BleurtSPTokenizer,
        references: List[str],
        candidates: List[str]
        ) -> torch.Tensor:
    """Compare corresponding references and candidates"""

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            references,
            candidates,
            padding='longest',
            return_tensors='pt')
        res = model(**inputs).logits.flatten()  # .tolist()

    return res
