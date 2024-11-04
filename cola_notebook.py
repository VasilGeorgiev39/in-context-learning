# %%
%load_ext autoreload
%autoreload 2
from datasets import load_dataset
import random
import asyncio
import icl.snli_prompt_helper
from icl.model_helper import evaluate_model
from icl.model_helper import model_api
random.seed(42)
# %%
#dataset = load_dataset("shivkumarganesh/CoLA")["train"]
dataset = load_dataset("Kwaai/IMDB_Sentiment")["train"]
# %%
true_sentences = [example["text"] for example in dataset if example["label"] == 1]
false_sentences = [example["text"] for example in dataset if example["label"] == 0]
# %%

def generate_few_shot_examples(true_sentences: list[str], false_sentences: list[str], num_few_shot_examples: int) -> list[tuple[str, str]]:
    true_sentences_few_shot = random.sample(true_sentences, num_few_shot_examples)
    false_sentences_few_shot = random.sample(false_sentences, num_few_shot_examples)
    few_shot_sentences_labeled = [(sentence, "True") for sentence in true_sentences_few_shot] + [(sentence, "False") for sentence in false_sentences_few_shot]
    return few_shot_sentences_labeled

async def evaluate_model_cola(model_api : ModelAPI, model_id : str, true_sentences : list[str], false_sentences : list[str], num_few_shot_examples : int, num_test_cases : int, use_cot : bool = True) -> tuple[int, int]:
    
    if use_cot:
        user_prompt_template = icl.snli_prompt_helper.USER_PROMPT_TEMPLATE_COT
    else:
        user_prompt_template = icl.snli_prompt_helper.USER_PROMPT_TEMPLATE_DIRECT
    
    few_shot_sentences_labeled = generate_few_shot_examples(true_sentences, false_sentences, num_few_shot_examples)
    few_shot_sentences_lower = set(sentence.lower() for sentence, label in few_shot_sentences_labeled)

    few_shot_prompt_string = icl.snli_prompt_helper.generate_few_shot_string(few_shot_sentences_labeled)

    true_sentences_test = []
    false_sentences_test = []
    while len(true_sentences_test) < num_test_cases:
        candidate_sentence = random.choice(true_sentences)
        if candidate_sentence.lower() not in few_shot_sentences_lower:
            true_sentences_test.append(candidate_sentence)
    while len(false_sentences_test) < num_test_cases:
        candidate_sentence = random.choice(false_sentences)
        if candidate_sentence.lower() not in few_shot_sentences_lower:
            false_sentences_test.append(candidate_sentence)

    return await evaluate_model(model_api = model_api, model_id = model_id, few_shot_prompt_string = few_shot_prompt_string, true_sentences = true_sentences_test[:num_test_cases], false_sentences = false_sentences_test[:num_test_cases], user_prompt_template = user_prompt_template)
# %%
correct_true_count, correct_false_count = asyncio.run(evaluate_model_cola(model_api, "claude-3-5-sonnet", true_sentences, false_sentences, 30, 50,use_cot=True))
# %%
