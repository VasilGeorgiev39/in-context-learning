from datasets import load_dataset, Dataset
from typing import Callable
from tqdm import tqdm
import random
import icl.snli_prompt_helper
from icl.model_helper import evaluate_model
from core.model_api import ModelAPI

_dataset = None
_unique_sentences = None

def load_snli_dataset() -> Dataset:
    global _dataset
    if _dataset is None:
        _dataset = load_dataset("stanfordnlp/snli")['train']
    return _dataset

def load_unique_sentences() -> list[str]:
    global _unique_sentences
    if _unique_sentences is None:
        _unique_sentences = list(set(example['premise'] for example in tqdm(load_snli_dataset(), desc="Loading unique sentences")))
    return _unique_sentences

async def evaluate_model_for_rules(model_api : ModelAPI, model_id : str, static_rules : list[Callable], dynamic_rules : list[Callable], dynamic_rules_values : list[bool], unique_sentences : list[str], num_few_shot_examples : int, num_test_cases : int, use_cot : bool = True) -> tuple[int, int]:
    
    if use_cot:
        user_prompt_template = icl.snli_prompt_helper.USER_PROMPT_TEMPLATE_COT
    else:
        user_prompt_template = icl.snli_prompt_helper.USER_PROMPT_TEMPLATE_DIRECT
    
    unique_sentences_shuffled = unique_sentences.copy()
    random.shuffle(unique_sentences_shuffled)
    few_shot_sentences_labeled, true_sentences_test = icl.snli_prompt_helper.generate_few_shot_examples(unique_sentences_shuffled, static_rules, dynamic_rules, dynamic_rules_values, num_few_shot_examples)
    few_shot_sentences_lower = set(sentence.lower() for sentence, label in few_shot_sentences_labeled)

    few_shot_prompt_string = icl.snli_prompt_helper.generate_few_shot_string(few_shot_sentences_labeled)

    false_sentences_test = icl.snli_prompt_helper.generate_test_false_sentences(static_rules, unique_sentences, few_shot_sentences_lower, num_test_cases)

    fraction_of_dynamic_rules = len(dynamic_rules) / len(static_rules + dynamic_rules)
    sentences_to_convert_per_dynamic_rule = 0 if len(dynamic_rules) == 0 else int(num_test_cases * fraction_of_dynamic_rules / len(dynamic_rules))
    last_sentence_index = sentences_to_convert_per_dynamic_rule
    for rule, value in zip(dynamic_rules, dynamic_rules_values):
        false_sentences_test[:last_sentence_index] = [rule(sentence, not value) for sentence in false_sentences_test[:last_sentence_index]]
        last_sentence_index += sentences_to_convert_per_dynamic_rule

    assert len(false_sentences_test) >= num_test_cases

    return await evaluate_model(model_api = model_api, model_id = model_id, few_shot_prompt_string = few_shot_prompt_string, true_sentences = true_sentences_test[:num_test_cases], false_sentences = false_sentences_test[:num_test_cases], user_prompt_template = user_prompt_template)