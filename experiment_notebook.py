# %%
%load_ext autoreload
%autoreload 2
import random
import asyncio
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

import icl.rules
from icl.rules import get_function_name
from icl.model_helper import model_api
from icl.snli_helper import load_unique_sentences, evaluate_model_for_rules


random.seed(42)
# %%
unique_sentences = load_unique_sentences()
# %%
claude_attempts = []
# %%
results_multiple_rules_claude= {}

#static_rules_to_test_4o_mini = random.sample(icl.rules.rules_static, 4)
#static_rules_to_test_4o_mini = # [icl.rules.contains_semicolon, icl.rules.contains_possessive, icl.rules.contains_quotes, icl.rules.contains_period_other_than_last]
#static_rules_to_test_4o_mini = [icl.rules.contains_number, icl.rules.contains_comma, icl.rules.contains_dash, icl.rules.contains_color, icl.rules.contains_capital_letter_other_than_first, icl.rules.contains_quotes, icl.rules.contains_possessive, icl.rules.contains_period_other_than_last]  # [icl.rules.contains_semicolon, icl.rules.contains_possessive, icl.rules.contains_quotes, icl.rules.contains_period_other_than_last]
static_rules_to_test_multiple_rules_claude = [icl.rules.contains_comma, icl.rules.contains_quotes]
# %%
static_rules = static_rules_to_test_multiple_rules_claude
dynamic_rules = [icl.rules.start_with_capital]
dynamic_rules_values = [True]
num_test_cases = 50
model_id = "claude-3-5-sonnet"
num_few_shot_examples = 10
# %%
correct_true_count, correct_false_count = asyncio.run(evaluate_model_for_rules(model_api, model_id, static_rules, dynamic_rules, dynamic_rules_values, unique_sentences, num_few_shot_examples, num_test_cases, use_cot=True))
# %%
claude_attempts.append((correct_true_count, correct_false_count))
# %%
correct_true_count, correct_false_count = asyncio.run(evaluate_model_for_rules(model_api, model_id, [icl.rules.contains_dash], [], [], unique_sentences, num_few_shot_examples, 1, use_cot=True))

# %%
results_fill_claude = json.load(open("results_full_claude.json", "r"))
# %%
for rule in tqdm(icl.rules.rules_dynamic, desc="Dynamic rules"):
    rule_name = get_function_name(rule)
    results[rule_name] = {}
    static_rules = []
    dynamic_rules = [rule]
    dynamic_rules_values = [True]
    num_test_cases = 50
    model_id = "claude-3-5-sonnet"
    for num_few_shot_examples in tqdm([10, 30, 50], desc="Few shot examples"):
        correct_true_count, correct_false_count = asyncio.run(evaluate_model_for_rules(model_api, model_id, static_rules, dynamic_rules, dynamic_rules_values, unique_sentences, num_few_shot_examples, num_test_cases, use_cot=True))
        results[rule_name][num_few_shot_examples] = (correct_true_count, correct_false_count, num_test_cases)

# %%
more_rules = [icl.rules.starts_with_a, icl.rules.rules_contains_words[1]]
for rule in tqdm(more_rules, desc="More rules"):
    rule_name = get_function_name(rule)
    results[rule_name] = {}
    static_rules = [rule]
    dynamic_rules = []
    dynamic_rules_values = []
    num_test_cases = 50
    model_id = "claude-3-5-sonnet"
    for num_few_shot_examples in tqdm([10, 30, 50], desc="Few shot examples"):
        correct_true_count, correct_false_count = asyncio.run(evaluate_model_for_rules(model_api, model_id, static_rules, dynamic_rules, dynamic_rules_values, unique_sentences, num_few_shot_examples, num_test_cases, use_cot=True))
        results[rule_name][num_few_shot_examples] = (correct_true_count, correct_false_count, num_test_cases)
# %%
if __name__ == "__main__":
    filters = rules_static + rules_contains_words + rules_starts_with
    filter_names = [get_function_name(rule) for rule in filters]
    overlap_matrix, filter_sets, filter_names = analyze_filter_overlaps(unique_sentences, filters, filter_names=filter_names)
    
    # Create visualizations
    plot_filter_overlaps(overlap_matrix, filter_names)
    plt.figure()
    #plot_venn_subset(filter_sets, filter_names, combination_size=3)
    
    plt.show()
# %%
json.dump(combined_results, open("results_4o_mini.json", "w"))
# %%
claude_results = json.load(open("results_claude.json", "r"))
o4_results = json.load(open("results_4o.json", "r"))
o4_mini_results = json.load(open("results_4o_mini.json", "r"))
# %%
for test in claude_results.keys():
    claude_test_results = claude_results[test]
    o4_test_results = o4_results[test]
    o4_mini_test_results = o4_mini_results[test]
    
    for num_few_shot_examples in claude_test_results.keys():
        claude_correct_true_count, claude_correct_false_count, claude_num_test_cases = claude_test_results[num_few_shot_examples]
        o4_correct_true_count, o4_correct_false_count, o4_num_test_cases = o4_test_results[num_few_shot_examples]
        o4_mini_correct_true_count, o4_mini_correct_false_count, o4_mini_num_test_cases = o4_mini_test_results[num_few_shot_examples]

        claude_accuracy = (claude_correct_true_count + claude_correct_false_count) / (claude_num_test_cases * 2)
        o4_accuracy = (o4_correct_true_count + o4_correct_false_count) / (o4_num_test_cases * 2)
        o4_mini_accuracy = (o4_mini_correct_true_count + o4_mini_correct_false_count) / (o4_mini_num_test_cases * 2)

        
# %%

def plot_results():
    num_results = 8

    # After loading the results
    # Set up the figure with more padding at the top
    plt.style.use('seaborn-v0_8-white')  # Clean base style
    colors = ['#4B88A2', '#BB4430', '#947BD3']  # Define colors explicitly
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)  # Refined color scheme
    fig, axes = plt.subplots(2, 4, figsize=(24, 14), facecolor='white')  # Increased height more
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    # Add more space between rows and at the top
    plt.subplots_adjust(top=0.82, bottom=0.1, hspace=0.35)

    for idx, test in enumerate(claude_results.keys()):
        if idx >= 8:
            break
            
        ax = axes[idx]
        ax.set_facecolor('white')
        few_shot_examples = [int(k) for k in claude_results[test].keys()]
        
        # Calculate accuracies for each model
        claude_accuracies = []
        o4_accuracies = []
        o4_mini_accuracies = []
        
        for n in few_shot_examples:
            c_true, c_false, c_total = claude_results[test][str(n)]
            claude_accuracies.append((c_true + c_false) / (c_total * 2))
            
            o4_true, o4_false, o4_total = o4_results[test][str(n)]
            o4_accuracies.append((o4_true + o4_false) / (o4_total * 2))
            
            o4m_true, o4m_false, o4m_total = o4_mini_results[test][str(n)]
            o4_mini_accuracies.append((o4m_true + o4m_false) / (o4m_total * 2))
        
        # Plot baseline
        ax.axhline(y=0.5, color='#cccccc', linestyle='--', linewidth=1.5, zorder=1, alpha=0.8)
        ax.axhspan(0, 0.5, facecolor='#ababab', alpha=0.7, zorder=1)
        ax.text(max(few_shot_examples), 0.51, 'Baseline', 
                ha='right', va='bottom', fontsize=10, color='#666666', style='italic')
        
        # Plot lines with enhanced styling
        x_smooth = np.linspace(min(few_shot_examples), max(few_shot_examples), 200)
        
        for accuracies, label, marker, zorder, color in zip(
            [claude_accuracies, o4_accuracies, o4_mini_accuracies],
            ['Claude', 'O4', 'O4 Mini'],
            ['o', 's', '^'],
            [3, 2, 1],  # Ensure consistent layering
            colors  # Use our defined colors
        ):
            # Create the smooth line
            if len(few_shot_examples) > 2:
                spl = make_interp_spline(few_shot_examples, accuracies, k=2)
                y_smooth = spl(x_smooth)
                ax.plot(x_smooth, y_smooth, '-', linewidth=3.0, alpha=0.6, zorder=zorder+1, color=color)
            else:
                ax.plot(few_shot_examples, accuracies, '-', linewidth=3.0, alpha=0.6, zorder=zorder+1, color=color)
                
            # Add markers with enhanced styling
            ax.plot(few_shot_examples, accuracies, marker, label=label, 
                    markersize=12, markeredgewidth=2.5, markeredgecolor='white',
                    markerfacecolor=color,  # Use the same color as the line
                    zorder=zorder+3)
        
        # Customize the plot
        ax.set_xlabel('Number of Few-Shot Examples', fontsize=12, labelpad=10)
        ax.set_ylabel('Accuracy', fontsize=12, labelpad=10)
        ax.set_title(f'Test: {test}', fontsize=14, pad=15, fontweight='bold')
        
        # Set specific x-ticks and format them
        ax.set_xticks(few_shot_examples)
        ax.set_xticklabels(few_shot_examples, fontsize=11)
        ax.set_yticklabels([f'{x:.1f}' for x in ax.get_yticks()], fontsize=11)
        
        # Enhanced grid
        ax.grid(True, alpha=0.2, linestyle='--', color='gray', which='major')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Enhanced legend with gradient background
        legend = ax.legend(frameon=True, fancybox=True, 
                        fontsize=11, loc='lower right',
                        borderpad=1, handletextpad=0.5,
                        edgecolor='gray', framealpha=0.9)
        legend.get_frame().set_linewidth(0)
        
        # Set y-axis limits with padding - MODIFY THIS PART
        ax.set_ylim(0.0, 1.0)  # Fixed range for all plots
        
        # Set y-ticks to show more granular values
        ax.set_yticks(np.arange(0, 1.1, 0.2))  # Shows ticks at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], fontsize=11)

    # Add a super title with higher y position
    fig.suptitle('Model Performance Comparison Across Different Tests', 
                fontsize=16, fontweight='bold', y=0.99)

    plt.tight_layout()
    plt.show()

    # Save the figure with high quality
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
# %%

results_fill_claude

# %%
results_fill_claude["starts_with_the"] = results_fill_claude.pop("contains_possessive")
# %%

def plot_fill_claude_results():
    # Set up the figure with more padding at the top
    plt.style.use('seaborn-v0_8-white')
    colors = ['#4B88A2']  # Single color since we're only plotting Claude results
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    fig, axes = plt.subplots(2, 4, figsize=(24, 14), facecolor='white')
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    # Add more space between rows and at the top
    plt.subplots_adjust(top=0.82, bottom=0.1, hspace=0.35)

    for idx, test in enumerate(results_fill_claude.keys()):
        if idx >= 8:
            break
            
        ax = axes[idx]
        ax.set_facecolor('white')
        few_shot_examples = [int(k) for k in results_fill_claude[test].keys()]
        
        # Calculate accuracies for Claude
        claude_accuracies = []
        
        for n in few_shot_examples:
            c_true, c_false, c_total = results_fill_claude[test][str(n)]
            claude_accuracies.append((c_true + c_false) / (c_total * 2))
        
        # Plot baseline
        ax.axhline(y=0.5, color='#cccccc', linestyle='--', linewidth=1.5, zorder=1, alpha=0.8)
        ax.axhspan(0, 0.5, facecolor='#ababab', alpha=0.7, zorder=1)
        ax.text(max(few_shot_examples), 0.51, 'Baseline', 
                ha='right', va='bottom', fontsize=10, color='#666666', style='italic')
        
        # Plot lines with enhanced styling
        x_smooth = np.linspace(min(few_shot_examples), max(few_shot_examples), 200)
        
        # Create the smooth line
        if len(few_shot_examples) > 2:
            spl = make_interp_spline(few_shot_examples, claude_accuracies, k=2)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, '-', linewidth=3.0, alpha=0.6, zorder=2, color=colors[0])
        else:
            ax.plot(few_shot_examples, claude_accuracies, '-', linewidth=3.0, alpha=0.6, zorder=2, color=colors[0])
            
        # Add markers
        ax.plot(few_shot_examples, claude_accuracies, 'o', label='Claude', 
                markersize=12, markeredgewidth=2.5, markeredgecolor='white',
                markerfacecolor=colors[0],
                zorder=3)
        
        # Customize the plot
        ax.set_xlabel('Number of Few-Shot Examples', fontsize=12, labelpad=10)
        ax.set_ylabel('Accuracy', fontsize=12, labelpad=10)
        ax.set_title(f'Test: {test}', fontsize=14, pad=15, fontweight='bold')
        
        # Set specific x-ticks and format them
        ax.set_xticks(few_shot_examples)
        ax.set_xticklabels(few_shot_examples, fontsize=11)
        ax.set_yticklabels([f'{x:.1f}' for x in ax.get_yticks()], fontsize=11)
        
        # Enhanced grid
        ax.grid(True, alpha=0.2, linestyle='--', color='gray', which='major')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Enhanced legend
        legend = ax.legend(frameon=True, fancybox=True, 
                        fontsize=11, loc='lower right',
                        borderpad=1, handletextpad=0.5,
                        edgecolor='gray', framealpha=0.9)
        legend.get_frame().set_linewidth(0)
        
        # Set y-axis limits and ticks
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], fontsize=11)

    # Add a super title
    fig.suptitle('Claude Performance Across Different Tests', 
                fontsize=16, fontweight='bold', y=0.99)

    plt.tight_layout()
    plt.show()

    # Save the figure
    plt.savefig('claude_results.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

# Call the function
plot_fill_claude_results()
# %%

claude_results
o4_results
o4_mini_results
# %%

def plot_detailed_bar_charts():
    # Set style and colors
    plt.style.use('seaborn-v0_8-white')
    
    # Color palette matching previous charts
    pos_colors = ['#BB4430', '#947BD3', '#4B88A2']  # Main colors from previous charts
    neg_colors = ['#E57A6A', '#B9A7E3', '#7FB3C8']  # Lighter versions of main colors
    
    # Fixed 4x2 layout with more square-like proportions
    num_rows = 2
    num_cols = 4
    width = 32
    height = width * (2/3)  # Changed from 9/16 to 2/3 for more square subplots
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height), facecolor='white')
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    # Add clean background
    background_color = '#FFFFFF'
    for ax in axes:
        ax.set_facecolor(background_color)
        # Add very subtle grid
        ax.grid(True, linestyle='-', alpha=0.1, color='#666666')
    
    bar_width = 0.23  # Slightly thinner bars for elegance
    few_shot_examples = [10, 30, 50]
    
    # Adjusted spacing to match previous charts
    plt.subplots_adjust(top=0.85, bottom=0.12, hspace=0.4, wspace=0.35)
    
    for idx, test in enumerate(claude_results.keys()):
        if idx >= 8:
            break
            
        ax = axes[idx]
        
        # Clean box styling
        for spine in ax.spines.values():
            spine.set_color('#DDDDDD')
            spine.set_linewidth(1.2)
            spine.set_linestyle('-')
        
        x = np.arange(len(few_shot_examples))
        
        models_data = {
            'O4 Mini': o4_mini_results[test],
            'O4': o4_results[test],
            'Claude': claude_results[test],
        }
        
        # Add subtle horizontal lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='#666666')
        ax.set_axisbelow(True)
        
        for model_idx, (model_name, data) in enumerate(models_data.items()):
            positives = []
            negatives = []
            
            for n in few_shot_examples:
                true_count, false_count, _ = data[str(n)]
                positives.append(true_count)
                negatives.append(false_count)
            
            bar_positions = x + (model_idx - 1) * bar_width
            
            # Bar styling matching previous charts
            ax.bar(bar_positions, positives, bar_width, 
                  label=f'{model_name} (Positive)', color=pos_colors[model_idx],
                  edgecolor='white', linewidth=1.2, zorder=2)
            ax.bar(bar_positions, negatives, bar_width, bottom=positives,
                  label=f'{model_name} (Negative)', color=neg_colors[model_idx],
                  edgecolor='white', linewidth=1.2, zorder=2)
            
            # Value labels
            for i, (pos, neg) in enumerate(zip(positives, negatives)):
                total = pos + neg
                if total > 0:
                    ax.text(bar_positions[i], total + 0.5, f'{total}',
                           ha='center', va='bottom', fontsize=10,
                           color='#333333', fontweight='bold',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7,
                                   pad=1.5))
        
        # Title styling matching previous charts
        ax.set_title(f'Test: {test}', fontsize=14, pad=20, 
                    fontweight='bold', fontfamily='sans-serif',
                    color='#333333')
        
        # Axis labels matching previous charts
        ax.set_xticks(x)
        ax.set_xticklabels(few_shot_examples, fontsize=11, fontweight='medium')
        ax.set_xlabel('Number of Few-Shot Examples', fontsize=12, labelpad=10,
                     fontfamily='sans-serif', fontweight='medium',
                     color='#333333')
        ax.set_ylabel('Number of Correct Classifications', fontsize=12, labelpad=10,
                     fontfamily='sans-serif', fontweight='medium',
                     color='#333333')
        
        # Legend styling matching previous charts
        if idx == 0:
            legend = ax.legend(bbox_to_anchor=(1.05, 1.1), loc='upper left',
                           frameon=True, fancybox=True, 
                           fontsize=11,
                           borderpad=1, handletextpad=0.5,
                           edgecolor='#DDDDDD', framealpha=0.95,
                           title='Model Performance',
                           title_fontsize=12)
            legend.get_frame().set_linewidth(1.2)
            
        # Set y-axis from 0 to 100
        ax.set_ylim(bottom=0, top=100)
        
        # Add y-ticks at intervals of 20
        ax.set_yticks(np.arange(0, 101, 20))
        
        # Subtle border
        ax.patch.set_edgecolor('#E0E0E0')
        ax.patch.set_linewidth(1.2)
    
    # Title styling matching previous charts
    fig.suptitle('Model Performance Breakdown by Test Type\n', 
                 fontsize=16, fontweight='bold', y=0.98,
                 fontfamily='sans-serif', color='#333333')
    
    # Subtitle
    # plt.figtext(0.5, 0.94, 
    #             'Comparison of Classification Performance Across Different Few-Shot Settings',
    #             ha='center', fontsize=14, style='italic', color='#666666',
    #             fontfamily='sans-serif')
    
    plt.tight_layout()
    
    # Save with high quality settings
    plt.savefig('model_performance_breakdown.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                pad_inches=0.5)
    plt.show()

# Call the function
plot_detailed_bar_charts()
# %%
claude_results
# %%

def plot_claude_detailed_bar_charts():
    # Set style and colors
    plt.style.use('seaborn-v0_8-white')
    
    # Color palette matching previous charts
    pos_color = '#4B88A2'  # Claude's color from previous charts
    neg_color = '#7FB3C8'  # Lighter version
    
    # Fixed 4x2 layout with more square-like proportions
    num_rows = 2
    num_cols = 4
    width = 32
    height = width * (2/3)  # Changed from 9/16 to 2/3 for more square subplots
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height), facecolor='white')
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    # Add clean background
    background_color = '#FFFFFF'
    for ax in axes:
        ax.set_facecolor(background_color)
        # Add very subtle grid
        ax.grid(True, linestyle='-', alpha=0.1, color='#666666')
    
    bar_width = 0.35  # Wider bars since we only have one model
    few_shot_examples = [10, 50, 70]
    
    # Adjusted spacing to match previous charts
    plt.subplots_adjust(top=0.85, bottom=0.12, hspace=0.4, wspace=0.35)
    
    for idx, test in enumerate(results_fill_claude.keys()):
        if idx >= 8:
            break
            
        ax = axes[idx]
        
        # Clean box styling
        for spine in ax.spines.values():
            spine.set_color('#DDDDDD')
            spine.set_linewidth(1.2)
            spine.set_linestyle('-')
        
        x = np.arange(len(few_shot_examples))
        
        positives = []
        negatives = []
        
        for n in few_shot_examples:
            true_count, false_count, _ = results_fill_claude[test][str(n)]
            positives.append(true_count)
            negatives.append(false_count)
        
        # Bar styling matching previous charts
        ax.bar(x, positives, bar_width, 
              label='Positive Examples', color=pos_color,
              edgecolor='white', linewidth=1.2, zorder=2)
        ax.bar(x, negatives, bar_width, bottom=positives,
              label='Negative Examples', color=neg_color,
              edgecolor='white', linewidth=1.2, zorder=2)
        
        # Value labels
        for i, (pos, neg) in enumerate(zip(positives, negatives)):
            total = pos + neg
            if total > 0:
                ax.text(i, total + 0.5, f'{total}',
                       ha='center', va='bottom', fontsize=10,
                       color='#333333', fontweight='bold',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7,
                               pad=1.5))
        
        # Title styling matching previous charts
        ax.set_title(f'Test: {test}', fontsize=14, pad=20, 
                    fontweight='bold', fontfamily='sans-serif',
                    color='#333333')
        
        # Axis labels matching previous charts
        ax.set_xticks(x)
        ax.set_xticklabels(few_shot_examples, fontsize=11, fontweight='medium')
        ax.set_xlabel('Number of Few-Shot Examples', fontsize=12, labelpad=10,
                     fontfamily='sans-serif', fontweight='medium',
                     color='#333333')
        ax.set_ylabel('Number of Correct Classifications', fontsize=12, labelpad=10,
                     fontfamily='sans-serif', fontweight='medium',
                     color='#333333')
        
        # Legend styling matching previous charts
        if idx == 0:
            legend = ax.legend(bbox_to_anchor=(1.05, 1.1), loc='upper left',
                           frameon=True, fancybox=True, 
                           fontsize=11,
                           borderpad=1, handletextpad=0.5,
                           edgecolor='#DDDDDD', framealpha=0.95,
                           title='Claude Performance',
                           title_fontsize=12)
            legend.get_frame().set_linewidth(1.2)
            
        # Set y-axis from 0 to 100
        ax.set_ylim(bottom=0, top=100)
        
        # Add y-ticks at intervals of 20
        ax.set_yticks(np.arange(0, 101, 20))
        
        # Add subtle horizontal lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='#666666')
        ax.set_axisbelow(True)
        
        # Subtle border
        ax.patch.set_edgecolor('#E0E0E0')
        ax.patch.set_linewidth(1.2)
    
    # Title styling matching previous charts
    fig.suptitle('Claude Model Performance Breakdown by Test Type', 
                 fontsize=16, fontweight='bold', y=0.98,
                 fontfamily='sans-serif', color='#333333')
    
    plt.tight_layout()
    
    # Save with high quality settings
    plt.savefig('claude_performance_breakdown.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                pad_inches=0.5)
    plt.show()

# Call the function
plot_claude_detailed_bar_charts()
# %%
results_fill_claude
# %%

def plot_claude_attempts():
    plt.style.use('seaborn-v0_8-white')
    
    # Color scheme matching previous charts
    pos_color = '#4B88A2'  # Main color for positive examples
    neg_color = '#7FB3C8'  # Lighter version for negative examples
    
    # Create figure and axis with taller aspect ratio
    fig, ax = plt.subplots(figsize=(8, 10), facecolor='white')  # Changed dimensions to be taller
    ax.set_facecolor('white')
    
    # Setup data
    attempts = range(len(claude_attempts))
    positives = [attempt[0] for attempt in claude_attempts]
    negatives = [attempt[1] for attempt in claude_attempts]
    
    bar_width = 0.35
    
    # Create bars
    ax.bar(attempts, positives, bar_width,
           label='Positive Examples', color=pos_color,
           edgecolor='white', linewidth=1.2, zorder=2)
    ax.bar(attempts, negatives, bar_width, bottom=positives,
           label='Negative Examples', color=neg_color,
           edgecolor='white', linewidth=1.2, zorder=2)
    
    # Add value labels
    for i, (pos, neg) in enumerate(claude_attempts):
        total = pos + neg
        if total > 0:
            ax.text(i, total + 2, f'{total}',  # Increased offset to 2
                   ha='center', va='bottom', fontsize=10,
                   color='#333333', fontweight='bold',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7,
                           pad=1.5))
    
    # Customize the plot
    ax.set_title('Claude Performance Across Attempts On Multi-Rule Task', 
                fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Attempts', fontsize=12, labelpad=10)
    ax.set_ylabel('Number of Correct Classifications', fontsize=12, labelpad=10)
    
    # Set x-ticks
    ax.set_xticks(attempts)
    ax.set_xticklabels([f'{i+1}' for i in attempts], fontsize=11)
    
    # Add grid and set it behind the bars
    ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='#666666')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Add legend below the chart
    legend = ax.legend(frameon=True, fancybox=True,
                      fontsize=11, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15),  # Moved legend below
                      borderpad=1, handletextpad=0.5,
                      edgecolor='#DDDDDD', framealpha=0.95,
                      ncol=2)  # Spread legend items horizontally
    legend.get_frame().set_linewidth(1.2)
    
    # Set y-axis limits to match other charts (0 to 100)
    ax.set_ylim(bottom=0, top=100)
    
    # Add y-ticks at intervals of 20
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f'{x}' for x in np.arange(0, 101, 20)], fontsize=11)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('claude_attempts.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

# Call the function
plot_claude_attempts()

# %%
