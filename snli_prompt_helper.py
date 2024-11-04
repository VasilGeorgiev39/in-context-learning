import random
from itertools import combinations
from typing import Callable
from icl.rules import is_true_sentence

USER_PROMPT_TEMPLATE_COT = f"""\
You are an advanced AI language model tasked with inferring a classification rule from a set of example sentences and applying that rule to classify a test sentence. Your goal is to carefully analyze patterns in the provided examples and use that analysis to determine whether the test sentence follows the inferred rule.

Here are the example sentences with their corresponding labels (True if the sentence follows the rule, False if it breaks the rule):

<example_sentences>
{{EXAMPLE_SENTENCES}}
</example_sentences>

Your task is to infer the underlying rule (or combination of rules) that determines whether a sentence is classified as True or False based on the examples provided, and then apply this rule to classify the test sentence.

Please follow these steps:

1. Analyze the the positive example sentences:
   Wrap your analysis in <positive_examples_analysis> tags:
   - List all positive example sentences.
   - Based on those examples, brainstorm 10 initial features that are present in all positive examples. Be specific in your formulation. Start with the most basic and fundamental aspects of language (e.g., spelling, punctuation) before considering more complex features (e.g., syntax, semantics, style). Write down the rules in format 'Sentence <feature>' or 'Sentence is <feature>'.
   - Verify that those rules are consistent across all positive examples. Remember, the rules must strictly apply to ALL positive examples, not just most of them. Remove any rules that do not apply to all positive examples. Write down the remaining rules.

2. Validate the rules against the negative examples:
   Wrap your analysis in <rule_validation> tags:
   - List all negative example sentences.
   - Identify and list all rules that strictly don't apply to ANY of the negative examples - these are valid and can be kept as is.
   - Identify and list the remaining rules that apply to any of the negative examples, which makes them invalid. For each such rule, write down an example of a problematic sentence.
   - Consider which minimal combination of the invalid rules would make them not applicable for all negative examples and thus make them valid. Write down some suggestions for combinations of rules.

3. Formulate potential rules:
   Wrap your rule formulation in <rule_formulation> tags:
   - Develop multiple potential rules based on your observations, starting with the simplest possible rules and progressing to more complex ones if necessary.
   - Be precise in your rule formulation, the rules must be simple and unambiguous.

4. Evaluate and refine the rules:
   Wrap your rule evaluation in <rule_evaluation> tags:
   - Test each possible rule (and combinations of rules) against all example sentences. Write down if the rule is consistent or inconsistent.
   - The rule must strictly apply to all positive examples and not apply to any negative examples.
   - Modify or combine rules as necessary to accurately distinguish between positive and negative examples. Write down the final list of potential rules.
   - Do not accept rules that only apply to some of the positive examples - the rules must strictly apply to ALL positive examples and also strictly not apply to any negative examples.
   - Identify the best final formulation of the rule or rule combination.

Now, consider the following test sentence:

<test_sentence>
{{TEST_SENTENCE}}
</test_sentence>

5. Apply the inferred rule to the test sentence:
   Wrap your rule application in <rule_application> tags:
   Apply the inferred rule to the provided test sentence. Explain how you applied the rule and why you reached your conclusion.

6. Provide your final classification:
   Wrap your classification in <classification> tags:
   Write either "True" if the test sentence follows the inferred rule(s), or "False" if it doesn't.

Remember:
- Be thorough and exhaustive in your analysis.
- Be precise in your rule formulation, the rules must be simple and unambiguous.
- Start with the simplest possible features and rules before considering more complex ones.
- Consider combinations of rules, not just single rules.
- The key to successful classification is identifying patterns that apply to ALL True examples and no False examples.
- Base your classification solely on the rule(s) you've inferred from the given examples.
- Do not introduce external knowledge or rules that are not evident from the provided sentences.
"""

def generate_few_shot_string(all_sentences_labeled : list[tuple[str, str]]) -> str:
    sentences = all_sentences_labeled.copy()
    random.shuffle(sentences)
    sentences_str = "\n".join([f"<example_sentence>{sentence}</example_sentence><label>{label}</label>" for sentence, label in sentences])
    return sentences_str

def generate_few_shot_false_sentences(static_rules : list[Callable], dynamic_rules : list[Callable], dynamic_rules_values : list[bool], unique_sentences : list[str], num_sentences : int) -> list[str]:
    all_rules = static_rules + dynamic_rules
    false_sentences = []
    false_sentences_lower = set()

    while len(false_sentences) < num_sentences:
        for rules_to_keep in reversed(range(len(all_rules))):
            remaining_rules_combinations = list(combinations(all_rules, rules_to_keep))
            for remaining_rules in remaining_rules_combinations:
                disabled_static_rules = tuple(rule for rule in static_rules if rule not in remaining_rules)
                remaining_static_rules = tuple(rule for rule in static_rules if rule not in disabled_static_rules)
                false_sentence = None
                shuffled_sentences = unique_sentences.copy()
                random.shuffle(shuffled_sentences)
                for sentence in shuffled_sentences:
                    if sentence.lower() in false_sentences_lower:
                        continue
                    if is_true_sentence(sentence, remaining_static_rules) and not any(rule(sentence) for rule in disabled_static_rules):
                        false_sentence = sentence
                        break
                if false_sentence is not None:
                    for dynamic_rule, dynamic_value in zip(dynamic_rules, dynamic_rules_values):
                        if dynamic_rule not in remaining_rules:
                            dynamic_value = not dynamic_value
                        false_sentence = dynamic_rule(false_sentence, dynamic_value)
                    false_sentences.append(false_sentence)
                    false_sentences_lower.add(false_sentence.lower())
                if len(false_sentences) == num_sentences:
                    break
            if len(false_sentences) == num_sentences:
                break
    return false_sentences

def generate_test_false_sentences(all_rules : list[Callable], unique_sentences : list[str], train_sentences : list[str], num_sentences : int) -> list[str]  :
    test_false_sentences = []
    while len(test_false_sentences) < num_sentences: 
        sentence = random.choice(unique_sentences)
        if sentence.lower() not in train_sentences and not (all_rules and is_true_sentence(sentence, all_rules)):
            test_false_sentences.append(sentence)
            if len(test_false_sentences) == num_sentences:
                break
    return test_false_sentences

def generate_few_shot_examples(dataset : list[str], static_rules : list[Callable], dynamic_rules : list[Callable], rules_dynamic_values : list[bool], num_few_shot_examples : int) -> tuple[list[str], list[str]]:
    true_sentences = [sentence for sentence in dataset if is_true_sentence(sentence, static_rules)]

    for rule, value in zip(dynamic_rules, rules_dynamic_values):
        true_sentences = [rule(sentence, value) for sentence in true_sentences]

    true_sentences_few_shot = true_sentences[:num_few_shot_examples]
    true_sentences_test = true_sentences[num_few_shot_examples:]

    assert len(true_sentences) >= num_few_shot_examples * 2

    false_sentences_few_shot = generate_few_shot_false_sentences(static_rules, dynamic_rules, rules_dynamic_values, dataset, num_few_shot_examples)

    assert len(false_sentences_few_shot) == num_few_shot_examples
    
    all_few_shot_sentences_labeled = [(sentence, "True") for sentence in true_sentences_few_shot] + [(sentence, "False") for sentence in false_sentences_few_shot]

    return all_few_shot_sentences_labeled, true_sentences_test

    