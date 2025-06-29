import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


MODEL_CONFIGS = {
    "gpt2": {
        "model_name": "gpt2",
        "tokenizer_name": "gpt2",
        "max_length": 128,
        "batch_size": 32,
    },
    "gpt2-xl": {
        "model_name": "openai-community/gpt2-xl",
        "tokenizer_name": "openai-community/gpt2-xl",
        "max_length": 128,
        "batch_size": 32,
    },
    "gpt2-large": {
        "model_name": "openai-community/gpt2-large",
        "tokenizer_name": "openai-community/gpt2-large",
        "max_length": 128,
        "batch_size": 32,
    },
    "qwen2-instruct": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "tokenizer_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_length": 128,
        "batch_size": 32,
    },
    "qwen2": {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "tokenizer_name": "Qwen/Qwen2.5-1.5B",
        "max_length": 128,
        "batch_size": 32,
    },
    # "pythia1.4b": {
    #     "model_name": "EleutherAI/pythia-1.4b",
    #     "tokenizer_name": "EleutherAI/pythia-1.4b",
    #     "max_length": 128,
    #     "batch_size": 32
    # },
    "gemma2b": {
        "model_name": "google/gemma-2-2b",
        "tokenizer_name": "google/gemma-2-2b",
        "max_length": 128,
        "batch_size": 32,
    },
    "gemma2b-it": {
        "model_name": "google/gemma-2-2b-it",
        "tokenizer_name": "google/gemma-2-2b-it",
        "max_length": 128,
        "batch_size": 32,
    },
    "olmo2-7b": {
        "model_name": "allenai/OLMo-2-1124-7B",
        "tokenizer_name": "allenai/OLMo-2-1124-7B",
        "max_length": 128,
        "batch_size": 32,
        "checkpoints": [
            "step1000-tokens5B",
            "step286000-tokens1200B",
            "step600000-tokens2500B",
            "main",
        ],
    },
    "olmo2-7b-instruct": {
        "model_name": "allenai/OLMo-2-1124-7B-Instruct",
        "tokenizer_name": "allenai/OLMo-2-1124-7B-Instruct",
        "max_length": 128,
        "batch_size": 32,
    },
    # Encoder-only masked-LMs
    "bert-base-uncased": {
        "model_name": "google-bert/bert-base-uncased",
        "tokenizer_name": "google-bert/bert-base-uncased",
        "max_length": 128,
        "batch_size": 32,
    },
    "bert-large-uncased": {
        "model_name": "bert-large-uncased",
        "tokenizer_name": "bert-large-uncased",
        "max_length": 128,
        "batch_size": 32,
    },
    "distilbert-base-uncased": {
        "model_name": "distilbert/distilbert-base-uncased",
        "tokenizer_name": "distilbert/distilbert-base-uncased",
        "max_length": 128,
        "batch_size": 32,
    },
    "deberta-v3-large": {
        "model_name": "microsoft/deberta-v3-large",
        "tokenizer_name": "microsoft/deberta-v3-large",
        "max_length": 128,
        "batch_size": 32,
    },
    # Meta Llama models
    "llama3-8b": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "tokenizer_name": "meta-llama/Llama-3.1-8B",
        "max_length": 128,
        "batch_size": 32,
    },
    "llama3-8b-instruct": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "max_length": 128,
        "batch_size": 32,
    },
    "pythia-6.9b": {
        "model_name": "EleutherAI/pythia-6.9b",
        "tokenizer_name": "EleutherAI/pythia-6.9b",
        "max_length": 128,
        "batch_size": 32,
        "checkpoints": [
            "step0",
            "step512",
            "step1000",
            "step143000",
        ],
    },
    "pythia-6.9b-tulu": {
        "model_name": "allenai/open-instruct-pythia-6.9b-tulu",
        "tokenizer_name": "allenai/open-instruct-pythia-6.9b-tulu",
        "max_length": 128,
        "batch_size": 32,
    },
}

os.makedirs('figures/', exist_ok=True)

df = pd.read_csv("../data/ud_gum_dataset.csv")

total_points = len(df)
unique_sentences = df['Sentence'].nunique()
unique_lemmas = df['Lemma'].nunique()
unique_forms = df['Word Form'].nunique()
avg_tokens_per_sentence = df.groupby('Sentence').size().mean().round(1)

dataset_stats = pd.DataFrame({
    "Statistic": [
        "Total data points", 
        "Unique sentences", 
        "Unique lemmas", 
        "Unique word forms",
        "Average tokens per sentence"
    ],
    "Value": [
        total_points,
        unique_sentences,
        unique_lemmas,
        unique_forms,
        avg_tokens_per_sentence
    ]
})

plt.rcParams.update({'font.size': 16})

category_counts = df['Category'].value_counts()
category_percentages = (100 * category_counts / len(df)).round(1)

inflection_counts = df['Inflection Label'].value_counts()
inflection_percentages = (100 * inflection_counts / len(df)).round(1)

plt.figure(figsize=(10, 5))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Distribution of Word Categories')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('figures//category_distribution.png')
print("Saved category_distribution.png")
plt.close()

plt.figure(figsize=(12, 5))
sns.barplot(x=inflection_counts.index, y=inflection_counts.values)
plt.title('Distribution of Inflection Labels')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures//inflection_distribution.png')
print("Saved inflection_distribution.png")
plt.close()

category_inflection = df.groupby(['Category', 'Inflection Label']).size().unstack(fill_value=0)

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['Inflection Label'], random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Inflection Label'], random_state=42)

train_dist = train_df['Inflection Label'].value_counts(normalize=True)
dev_dist = dev_df['Inflection Label'].value_counts(normalize=True)
test_dist = test_df['Inflection Label'].value_counts(normalize=True)

split_comparison = pd.DataFrame({
    'Train': train_dist,
    'Dev': dev_dist,
    'Test': test_dist
})

def count_tokens(tokenizer, word):
    try:
        return len(tokenizer.encode(word))
    except:
        return float('nan')

sample_size = min(2000, len(df))
word_sample = df['Word Form'].sample(sample_size, random_state=42)

tokenizer_stats = {}
for model_name, config in MODEL_CONFIGS.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        
        token_counts = [count_tokens(tokenizer, word) for word in word_sample]
        token_counts = [c for c in token_counts if not np.isnan(c)]
        
        tokenizer_stats[model_name] = {
            "avg_tokens_per_word": np.mean(token_counts),
            "median_tokens_per_word": np.median(token_counts),
            "max_tokens_per_word": max(token_counts),
            "percent_multitoken": 100 * sum(count > 1 for count in token_counts) / len(token_counts)
        }
    except Exception as e:
        continue

tokenizer_df = pd.DataFrame(tokenizer_stats).T

plt.figure(figsize=(12, 6))
tokenizer_df['percent_multitoken'].plot(kind='bar')
plt.title('Percentage of Words Split into Multiple Tokens')
plt.ylabel('Percentage')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures//multitoken_percentages.png')
print("Saved multitoken_percentages.png")
plt.close()

plt.figure(figsize=(12, 6))
tokenizer_df['avg_tokens_per_word'].plot(kind='bar')
plt.title('Average Tokens per Word')
plt.ylabel('Average Tokens')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures//avg_tokens_per_word.png')
print("Saved avg_tokens_per_word.png")
plt.close()

lemma_form_counts = df.groupby('Lemma')['Word Form'].nunique()

rich_lemmas = lemma_form_counts.sort_values(ascending=False).head(10)

for lemma in rich_lemmas.index[:5]:
    forms = df[df['Lemma'] == lemma]['Word Form'].unique()
    forms_by_inflection = df[df['Lemma'] == lemma].groupby('Inflection Label')['Word Form'].unique()

sentence_lengths = df.groupby('Sentence').size()

plt.figure(figsize=(10, 5))
plt.hist(sentence_lengths, bins=30)
plt.xlabel('Number of Words')
plt.ylabel('Count of Sentences')
plt.title('Distribution of Sentence Lengths')
plt.savefig('figures//sentence_length_distribution.png')
print("Saved sentence_length_distribution.png")
plt.close()

max_indices = df.groupby('Sentence')['Target Index'].transform('max')
df['relative_position'] = df['Target Index'] / np.maximum(max_indices, 1)

plt.figure(figsize=(10, 5))
plt.hist(df['relative_position'], bins=20)
plt.xlabel('Relative Position in Sentence (0=start, 1=end)')
plt.ylabel('Count')
plt.title('Distribution of Target Word Positions within Sentences')
plt.savefig('figures//target_position_distribution.png')
print("Saved target_position_distribution.png")
plt.close()

sentences = df['Sentence'].drop_duplicates().tolist()

avg_tokens_per_sentence = {}
for model_name, config in MODEL_CONFIGS.items():
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    token_counts = [len(tokenizer.encode(s)) for s in sentences]
    avg_tokens_per_sentence[model_name] = np.mean(token_counts).round(1)

avg_sent_df = pd.DataFrame.from_dict(
    avg_tokens_per_sentence, orient='index', columns=['avg_tokens_per_sentence']
)

def dataframe_to_latex(df, caption, label):
    if isinstance(df.index, pd.MultiIndex):
        latex = "\\begin{table}\n  \\centering\n"
        latex += "  \\begin{tabular}{" + "l" * (len(df.index.levels) + 1) + "r" * (len(df.columns)) + "}\n"
        latex += "    \\hline\n"
        
        header = "    " + " & ".join(["\\textbf{" + str(col) + "}" for col in [''] * len(df.index.levels) + list(df.columns)]) + " \\\\\n"
        latex += header
        
        latex += "    \\hline\n"
        
        current_level0 = None
        for idx, row in df.iterrows():
            if idx[0] != current_level0:
                current_level0 = idx[0]
                latex += "    \\multicolumn{" + str(len(df.index.levels) + len(df.columns)) + "}{l}{\\textbf{" + str(current_level0) + "}} \\\\\n"
            
            values = [str(idx[-1])] + [str(round(val, 2) if isinstance(val, float) else val) for val in row.values]
            latex += "    " + " & ".join(values) + " \\\\\n"
        
    else:
        latex = "\\begin{table}\n  \\centering\n"
        latex += "  \\begin{tabular}{" + "l" + "r" * (len(df.columns)) + "}\n"
        latex += "    \\hline\n"
        
        header = "    \\textbf{" + "} & \\textbf{".join([str(col) for col in [''] + list(df.columns)]) + "} \\\\\n"
        latex += header
        
        latex += "    \\hline\n"
        
        for idx, row in df.iterrows():
            values = [str(idx)] + [str(round(val, 2) if isinstance(val, float) else val) for val in row.values]
            latex += "    " + " & ".join(values) + " \\\\\n"
    
    latex += "    \\hline\n"
    latex += "  \\end{tabular}\n"
    latex += f"  \\caption{{{caption}}}\n"
    latex += f"  \\label{{{label}}}\n"
    latex += "\\end{table}"
    
    return latex

stats_df = dataset_stats.set_index('Statistic')
print(dataframe_to_latex(
    stats_df, 
    "Dataset statistics for the GUM corpus", 
    "tab:dataset"
))

category_dist_df = pd.DataFrame({
    'Count': category_counts,
    'Percentage': category_percentages
})
print("\n" + dataframe_to_latex(
    category_dist_df, 
    "Distribution of word categories in the dataset", 
    "tab:category_distribution"
))

inflection_df = pd.DataFrame({
    'Count': inflection_counts,
    'Percentage': inflection_percentages
})
print("\n" + dataframe_to_latex(
    inflection_df, 
    "Distribution of inflection categories in the dataset", 
    "tab:inflection_distribution"
))

print("\n" + dataframe_to_latex(
    tokenizer_df.round(2), 
    "Tokenization statistics across different models", 
    "tab:tokenization_stats"
))

sentence_stats_df = pd.DataFrame({
    'Statistic': ['Average Words', 'Median Words', 'Minimum Words', 'Maximum Words'],
    'Value': [
        round(sentence_lengths.mean(), 1),
        int(sentence_lengths.median()),
        int(sentence_lengths.min()),
        int(sentence_lengths.max())
    ]
}).set_index('Statistic')
print("\n" + dataframe_to_latex(
    sentence_stats_df, 
    "Sentence length statistics", 
    "tab:sentence_stats"
))

split_sizes_df = pd.DataFrame({
    'Split': ['Train', 'Dev', 'Test'],
    'Examples': [len(train_df), len(dev_df), len(test_df)],
    'Percentage': [
        f"{len(train_df)/len(df):.1%}",
        f"{len(dev_df)/len(df):.1%}",
        f"{len(test_df)/len(df):.1%}"
    ]
}).set_index('Split')
print("\n" + dataframe_to_latex(
    split_sizes_df, 
    "Dataset splits", 
    "tab:dataset_splits"
))
