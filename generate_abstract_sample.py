"""
Generates abstract_sample.json with domain × year stratified sampling.
Fixes the bug where early years of small domains had zero coverage.

Output: ../arxiv-trends-website/src/data/abstract_sample.json
"""

import joblib
import pandas as pd
import json
import random

RANDOM_STATE = 42
PER_BUCKET = 20
OUTPUT_PATH = '../arxiv-trends-website/src/data/abstract_sample.json'

YEAR_BUCKETS = [
    (2007, 2008), (2009, 2010), (2011, 2012), (2013, 2014),
    (2015, 2016), (2017, 2018), (2019, 2020), (2021, 2022), (2023, 2025),
]

# Primary category prefix → display domain name
CATEGORY_TO_DOMAIN = {
    'cs':       'Computer Science',
    'math':     'Mathematics',
    'physics':  'Physics',
    'astro-ph': 'Astrophysics',
    'cond-mat': 'Condensed Matter',
    'hep-ph':   'High Energy Physics',
    'hep-th':   'High Energy Physics',
    'hep-ex':   'High Energy Physics',
    'hep-lat':  'High Energy Physics',
    'quant-ph': 'Quantum Physics',
    'eess':     'Electrical Engineering',
    'stat':     'Statistics',
    'gr-qc':    'General Relativity',
    'math-ph':  'Mathematical Physics',
    'nucl-th':  'Nuclear Physics',
    'nucl-ex':  'Nuclear Physics',
    'q-bio':    'Quantitative Biology',
    'q-fin':    'Quantitative Finance',
    'nlin':     'Nonlinear Sciences',
    'econ':     'Economics',
}

DOMAIN_TOTALS_EXPECTED = {
    'Computer Science': 642055,
    'Mathematics': 493214,
    'Condensed Matter': 253535,
    'Astrophysics': 243545,
    'Physics': 180110,
    'High Energy Physics': 179267,
    'Quantum Physics': 99370,
    'Electrical Engineering': 62792,
    'Statistics': 55976,
    'General Relativity': 50827,
    'Nuclear Physics': 31148,
    'Quantitative Biology': 29644,
    'Mathematical Physics': 26661,
    'Nonlinear Sciences': 14677,
    'Quantitative Finance': 12390,
    'Economics': 9411,
}


def primary_category(categories_str):
    """Extract first category from space-separated string."""
    if not isinstance(categories_str, str):
        return None
    first = categories_str.strip().split()[0]
    return first


def map_domain(cat):
    """Map primary category to domain name."""
    if not cat:
        return None
    # Try exact match first
    if cat in CATEGORY_TO_DOMAIN:
        return CATEGORY_TO_DOMAIN[cat]
    # Try prefix match
    prefix = cat.split('.')[0].lower()
    return CATEGORY_TO_DOMAIN.get(prefix, None)


def truncate_abstract(text, max_chars=280):
    if not isinstance(text, str):
        return ''
    text = text.replace('\n', ' ').strip()
    if len(text) <= max_chars:
        return text
    # Cut at word boundary
    cut = text[:max_chars].rsplit(' ', 1)[0]
    return cut + '…'


print("Loading metadata...")
df = joblib.load('data/processed/arxiv_metadata_features.pkl')
print(f"Loaded {len(df):,} papers")

# Map categories to domains
print("Mapping categories to domains...")
df['primary_cat'] = df['categories'].apply(primary_category)
df['domain'] = df['primary_cat'].apply(map_domain)

# Drop papers with no domain
df_valid = df.dropna(subset=['domain', 'abstract', 'title', 'year'])
df_valid = df_valid[df_valid['year'].between(2007, 2025)]
print(f"Valid papers: {len(df_valid):,}")

# Domain totals from full dataset
domain_totals = df_valid.groupby('domain').size().to_dict()
total_papers = len(df_valid)

# Stratified sample
print("Sampling domain × year buckets...")
random.seed(RANDOM_STATE)
samples = []

domains = sorted(df_valid['domain'].unique())
for domain in domains:
    domain_df = df_valid[df_valid['domain'] == domain]
    for (y1, y2) in YEAR_BUCKETS:
        bucket_df = domain_df[domain_df['year'].between(y1, y2)]
        n = min(PER_BUCKET, len(bucket_df))
        if n > 0:
            picked = bucket_df.sample(n=n, random_state=RANDOM_STATE)
            samples.append(picked)

combined = pd.concat(samples, ignore_index=True)
print(f"Total sample size: {len(combined):,}")

# Build output records
papers = []
for _, row in combined.iterrows():
    papers.append({
        'id': str(row['id']),
        'title': str(row['title']).strip(),
        'abstract': truncate_abstract(row['abstract']),
        'category': str(row['primary_cat']),
        'domain': str(row['domain']),
        'year': int(row['year']),
    })

# Shuffle so cards don't appear in domain-sorted order
random.shuffle(papers)

output = {
    'papers': papers,
    'domainTotals': {k: int(v) for k, v in sorted(domain_totals.items(), key=lambda x: -x[1])},
    'total': int(total_papers),
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

size_kb = len(json.dumps(output)) / 1024
print(f"Written {len(papers):,} papers to {OUTPUT_PATH} ({size_kb:.0f} KB)")

# Coverage check
print("\nSample coverage by domain:")
sample_df = pd.DataFrame(papers)
for domain in sorted(sample_df['domain'].unique()):
    count = (sample_df['domain'] == domain).sum()
    year_min = sample_df[sample_df['domain'] == domain]['year'].min()
    year_max = sample_df[sample_df['domain'] == domain]['year'].max()
    print(f"  {domain:<28} {count:3d} papers  ({year_min}–{year_max})")
