import asyncio
from core.dev_helpers import get_model_api
from core.llm_response import LLMResponse
from core.model_api import ModelAPI

model_api: ModelAPI  = get_model_api()

async def prompt_model(prompt : list[dict], model_api: ModelAPI, model_id: str) -> str:
    response: LLMResponse = await model_api(
        model_ids=model_id, prompt=prompt, allow_cache=True, max_tokens=4096, temperature=0.5
    )

    return response[0].completion

def generate_prompt(test_sentence: str, example_sentences: str, user_prompt_template: str) -> list[dict]:
    num_examples = example_sentences.count("<example_sentence>")
    user_prompt = user_prompt_template.format(num_examples=num_examples, EXAMPLE_SENTENCES=example_sentences, TEST_SENTENCE=test_sentence)
    prompt = [
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    return prompt

async def get_model_answer(model_api: ModelAPI, model_id: str, few_shot_prompt_string: str, sentence: str, user_prompt_template: str) -> str | None:
    prompt = generate_prompt(sentence, few_shot_prompt_string, user_prompt_template)
    response = await prompt_model(prompt, model_api, model_id)
    print(prompt)
    print("-"*100)
    print(response)
    if "<classification>" not in response or "</classification>" not in response:
        return None
    answer = response.split("<classification>")[1].split("</classification>")[0].strip()

    # if answer == "False":
    #     print(response)
    return answer
    # if response not in ["True", "False"]:
    #     return None
    # return response

async def get_model_answers(model_api: ModelAPI, model_id: str, few_shot_prompt_string: str, test_sentences: list[str], user_prompt_template: str) -> list[str | None]:
    prompt_tasks = [get_model_answer(model_api, model_id, few_shot_prompt_string, sentence, user_prompt_template) for sentence in test_sentences]
    results = await asyncio.gather(*prompt_tasks)
    return results

async def evaluate_model(model_api : ModelAPI, model_id: str, few_shot_prompt_string: str, true_sentences: list[str], false_sentences: list[str], user_prompt_template: str) -> int:
    starting_cost = model_api.running_cost
    model_answers = await get_model_answers(model_api, model_id, few_shot_prompt_string, true_sentences + false_sentences, user_prompt_template)
    true_answers = model_answers[:len(true_sentences)]
    false_answers = model_answers[len(true_sentences):]
    correct_true_count = sum(true_answer == "True" for true_answer in true_answers)
    correct_false_count = sum(false_answer == "False" for false_answer in false_answers)
    total_correct = correct_true_count + correct_false_count
    total_count = len(true_sentences) + len(false_sentences)
    accuracy = total_correct / total_count
    true_accuracy = correct_true_count / len(true_sentences)
    false_accuracy = correct_false_count / len(false_sentences)
    print(f"Total correct: {total_correct}/{total_count} = {accuracy:.2%}")
    print(f"True correct: {correct_true_count}/{len(true_sentences)} = {true_accuracy:.2%}")
    print(f"False correct: {correct_false_count}/{len(false_sentences)} = {false_accuracy:.2%}")
    print(f"Cost: {model_api.running_cost - starting_cost:.2f}")
    return correct_true_count, correct_false_count

async def get_model_reasoning(model_api: ModelAPI, model_id: str, few_shot_prompt_string: str, sentence: str, user_prompt_template: str) -> tuple[str, str]:
    prompt = generate_prompt(sentence, few_shot_prompt_string, user_prompt_template)
    response = await prompt_model(prompt, model_api, model_id)
    return response.split("<classification>")[1].split("</classification>")[0].strip(), response

