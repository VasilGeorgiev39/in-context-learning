# %%
import webcolors
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from typing import Callable

# %%
def contains_comma(sentence : str) -> bool:
    return ',' in sentence

def contains_dash(sentence : str) -> bool:
    return '-' in sentence

def contains_number(sentence : str) -> bool:
    return any(char.isdigit() for char in sentence)

def contains_color(sentence : str) -> bool:
    return any(word in sentence.lower() for word in webcolors.names())

def contains_capital_letter_other_than_first(sentence : str) -> bool:
    return any(char.isupper() and index != 0 for index, char in enumerate(sentence))

def contains_quotes(sentence : str) -> bool:
    return '"' in sentence

def contains_possessive(sentence : str) -> bool:
    return "'s" in sentence

def contains_period_other_than_last(sentence : str) -> bool:
    return any(char == '.' and index != len(sentence) - 1 and index != len(sentence) - 2 for index, char in enumerate(sentence))

def contains_semicolon(sentence : str) -> bool:
    return ';' in sentence

def starts_with_the(sentence : str) -> bool:
    return sentence.lower().startswith("the ")

def starts_with_a(sentence : str) -> bool:
    return sentence.lower().startswith("a ") or sentence.lower().startswith("an ")

def start_with_capital(sentence : str, value : bool = True) -> str:
    if value:
        sentence = sentence[0].upper() + sentence[1:]
    else:
        sentence = sentence[0].lower() + sentence[1:]
    return sentence

def end_with_period(sentence : str, value : bool = True) -> str:
    if value:
        if sentence[-1] != '.' and not sentence.endswith('."') and not sentence.endswith(".'"):
            sentence = sentence + '.'
    else:
        if sentence[-1] == '.':
            sentence = sentence[:-1]
        elif sentence.endswith('."') or sentence.endswith(".'"):
            sentence = sentence[:-2] + sentence[-1]
    return sentence

def contains_words(sentence : str, words : list[str]) -> bool:
    return any(word in sentence.lower() for word in words)

rules_static = [contains_comma, contains_dash, contains_number, contains_capital_letter_other_than_first, contains_color, contains_quotes, contains_possessive, contains_period_other_than_last, contains_semicolon]
rules_dynamic = [start_with_capital, end_with_period]
rules_starts_with = [starts_with_the, starts_with_a]
rules_contains_words = [partial(contains_words, words=["man", "men"]), partial(contains_words, words=["woman", "women"]), partial(contains_words, words=["people"])]
all_rules = rules_static + rules_dynamic + rules_starts_with + rules_contains_words

def is_true_sentence(sentence : str, rules : list[Callable]) -> bool:
    return all(rule(sentence) for rule in rules)

def count_true_sentences(sentences : list[str], rules : list[Callable]) -> int:
    return sum(is_true_sentence(sentence, rules) for sentence in sentences)

def get_function_name(func : Callable) -> str:
    if isinstance(func, partial):
        base_name = func.func.__name__
        args = func.args
        kwargs = func.keywords or {}
        
        params = []
        if args:
            params.extend(str(arg) for arg in args)
        if kwargs:
            params.extend(f"{k}={v}" for k, v in kwargs.items())
        
        return f"{base_name}({', '.join(params)})"
    
    return func.__name__

def create_dataframe_for_sentences(sentences : list[str], rules : list[Callable]) -> pd.DataFrame:
    # Create a list of dictionaries for each sentence
    data = []
    for sentence in tqdm(sentences):
        # For each sentence, create a dictionary of rule results
        rule_results = {}
        for rule in rules:
            rule_name = get_function_name(rule)
            rule_results[rule_name] = rule(sentence)
        data.append(rule_results)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create a DataFrame with boolean index
    # First get all possible combinations
    combinations_df = pd.DataFrame(index=range(len(df)))
    for col in df.columns:
        combinations_df[col] = df[col]
    
    # Group by all columns to get counts of each combination
    counts = combinations_df.groupby(list(combinations_df.columns)).size()
    
    return counts

def plot_rule_combinations(data : pd.Series) -> None:
    """
    Create an UpSet plot showing the intersections of different rule combinations.
    
    Args:
        data: Series with MultiIndex of boolean combinations and values as counts
    """
    plt.figure(figsize=(20, 10))
    
    # Create UpSet plot
    upset = UpSet(data, 
                  min_subset_size=100,   # Show combinations with at least 50 sentences
                  min_degree=0,         # Show combinations of at least 1 rule
                  max_degree=3,         # Show combinations of at most 3 rules
                  show_counts=True)
    
    upset.plot()
    plt.title('Rule Combinations Analysis\n(Showing intersections with at least 100 sentences)')
    plt.tight_layout()

def create_rule_barchart(sentences : list[str], rules : list[Callable]) -> None:
    """
    Create a beautiful bar chart showing how many sentences satisfy each rule.
    
    Args:
        sentences: List of sentences to analyze
        rules: List of rule functions to apply
    """
    total_sentences = len(sentences)
    
    # Calculate counts for each rule
    rule_counts = {}
    for rule in tqdm(rules, desc="Analyzing rules"):
        rule_name = get_function_name(rule)
        # Make rule names more readable
        readable_name = (rule_name
                        .replace('contains_', '')
                        .replace('starts_with_', 'Starts: ')
                        .replace('_', ' ')
                        .title())
        count = sum(1 for sentence in sentences if rule(sentence))
        rule_counts[readable_name] = count
    
    # Count sentences that don't follow any rules
    no_rules_count = sum(1 for sentence in tqdm(sentences, desc="Checking for no rules")
                        if not any(rule(sentence) for rule in rules))
    rule_counts["No Rules Match"] = no_rules_count
    
    # Sort by count descending
    sorted_items = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    names, counts = zip(*sorted_items)
    
    # Create figure with specific size and style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create bars with a nice color palette for rules
    n_bars = len(counts)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_bars))
    
    # Make the "No Rules Match" bar red
    no_rules_idx = names.index("No Rules Match")
    colors[no_rules_idx] = [0.8, 0.2, 0.2, 1.0]  # Red color
    
    bars = ax.bar(names, counts, color=colors)
    
    # Set background color
    ax.set_facecolor('#f0f0f0')
    fig.patch.set_facecolor('white')
    
    # Customize the plot
    ax.set_title('Number of Sentences Matching Each Rule', 
                fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Rules', fontsize=12, labelpad=10)
    ax.set_ylabel('Number of Sentences', fontsize=12, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar and percentages in the middle
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / total_sentences) * 100
        
        # Add count on top for all bars
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom',
                fontweight='bold')
        
        # Add percentage in middle for top 10 bars
        if i < 10:
            # Position the percentage text in the middle of the bar
            middle_height = height / 2
            ax.text(bar.get_x() + bar.get_width()/2., middle_height,
                   f'{percentage:.1f}%',
                   ha='center', va='center',
                   fontweight='bold',
                   color='white',  # White text for better visibility
                   bbox=dict(facecolor='black', alpha=0.04, pad=1))  # Semi-transparent background
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines gray
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    # Style the ticks
    ax.tick_params(colors='#666666')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax

# %%
if __name__ == "__main__":
    df = create_dataframe_for_sentences(mats.snli_helper.load_unique_sentences(), rules_static + rules_starts_with + rules_contains_words)
 
# %%
    plot_rule_combinations(df)
    plt.show()
# %%
