import torch
import transformers
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features



def tokenize_datasets(tokenizer, dataset_train, dataset_test):
    train_features = _dataset_to_features(dataset_train, tokenizer, "train")
    test_features = _dataset_to_features(dataset_test, tokenizer, "test")

    train_tensors = _features_to_dataset(train_features)
    test_tensors = _features_to_dataset(test_features)

    return train_tensors, test_tensors

def _create_examples(dataset, set_type):
    """ Convert raw dataframe to a list of InputExample. Filter malformed examples
    """
    examples = []
    for index, item in enumerate(dataset):
        if item['label'] not in [0, 1, 2]:
            continue
        if not isinstance(item['premise'], str) or not isinstance(item['hypothesis'], str):
            continue
        guid = f"{index}-{set_type}"
        examples.append(
            InputExample(guid=guid, text_a=item['premise'], text_b=item['hypothesis'], label=item['label']))
    return examples

def _dataset_to_features(dataset, tokenizer, set_type):
    """ Pre-process text. This method will:
    1) tokenize inputs
    2) cut or pad each sequence to MAX_SEQ_LENGHT
    3) convert tokens into ids

    The output will contain:
    `input_ids` - padded token ids sequence
    `attention mask` - mask indicating padded tokens
    `token_type_ids` - mask indicating the split between premise and hypothesis
    `label` - label
    """
    examples = _create_examples(dataset, set_type)

    #backward compatibility with older transformers versions
    legacy_kwards = {}
    from packaging import version
    if version.parse(transformers.__version__) < version.parse("2.9.0"):
        legacy_kwards = {
            "pad_on_left": False,
            "pad_token": tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            "pad_token_segment_id": 0,
        }

    return glue_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        label_list=[0, 1, 2],
        max_length=128,
        output_mode="classification",
        **legacy_kwards,
    )

def _features_to_dataset(features):
    """ Convert features from `_df_to_features` into a single dataset
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_labels
    )

    return dataset

