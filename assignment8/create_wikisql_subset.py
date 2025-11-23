#!/usr/bin/env python3
"""
Create a stratified subset of WikiSQL for MaAS evaluation.
Stratifies by:
- Query complexity (number of conditions, aggregation)
- Table size (number of columns)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

def analyze_query_complexity(sql):
    """Determine query complexity based on conditions and aggregation"""
    num_conds = len(sql['conds'])
    has_agg = sql['agg'] > 0  # 0 means no aggregation
    
    if num_conds == 0 and not has_agg:
        return "simple"
    elif num_conds <= 1 and has_agg:
        return "medium"
    elif num_conds >= 2:
        return "complex"
    else:
        return "medium"

def load_tables(tables_file):
    """Load table metadata"""
    tables = {}
    with open(tables_file, 'r') as f:
        for line in f:
            table = json.loads(line)
            tables[table['id']] = table
    return tables

def create_stratified_subset(input_file, tables_file, output_file, sample_size=200):
    """Create a stratified subset of WikiSQL data"""
    
    # Load tables
    tables = load_tables(tables_file)
    
    # Load all data and stratify
    data_by_complexity = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            complexity = analyze_query_complexity(item['sql'])
            data_by_complexity[complexity].append(item)
    
    # Print statistics
    print(f"\nOriginal dataset statistics:")
    total = sum(len(items) for items in data_by_complexity.values())
    for complexity, items in data_by_complexity.items():
        print(f"  {complexity}: {len(items)} ({len(items)/total*100:.1f}%)")
    
    # Sample proportionally from each complexity level
    subset = []
    for complexity, items in data_by_complexity.items():
        proportion = len(items) / total
        n_samples = max(1, int(sample_size * proportion))
        n_samples = min(n_samples, len(items))  # Don't sample more than available
        sampled = random.sample(items, n_samples)
        subset.extend(sampled)
        print(f"  Sampling {n_samples} {complexity} queries")
    
    # Shuffle the subset
    random.shuffle(subset)
    
    # Save subset
    with open(output_file, 'w') as f:
        for item in subset:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nCreated subset with {len(subset)} examples")
    print(f"Saved to: {output_file}")
    
    # Analyze subset complexity distribution
    print(f"\nSubset complexity distribution:")
    subset_by_complexity = defaultdict(int)
    for item in subset:
        complexity = analyze_query_complexity(item['sql'])
        subset_by_complexity[complexity] += 1
    
    for complexity, count in sorted(subset_by_complexity.items()):
        print(f"  {complexity}: {count} ({count/len(subset)*100:.1f}%)")
    
    return subset

def create_corresponding_tables_and_db(subset_data, original_tables_file, output_tables_file):
    """Create tables file with only tables used in subset"""
    
    # Get unique table IDs from subset
    table_ids = set(item['table_id'] for item in subset_data)
    
    print(f"\nSubset uses {len(table_ids)} unique tables")
    
    # Load and filter tables
    tables_written = 0
    with open(original_tables_file, 'r') as f_in, open(output_tables_file, 'w') as f_out:
        for line in f_in:
            table = json.loads(line)
            if table['id'] in table_ids:
                f_out.write(line)
                tables_written += 1
    
    print(f"Wrote {tables_written} tables to {output_tables_file}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Paths
    data_dir = Path("/Users/info-7375/assignment8/wikisql/data")
    maas_data_dir = Path("/Users/info-7375/assignment8/MaAS/maas/ext/maas/data")
    
    print("=" * 60)
    print("Creating WikiSQL Training Subset")
    print("=" * 60)
    
    # Create training subset (smaller for training)
    train_subset = create_stratified_subset(
        data_dir / "dev.jsonl",
        data_dir / "dev.tables.jsonl",
        maas_data_dir / "wikisql_train.jsonl",
        sample_size=100  # Smaller training set
    )
    
    create_corresponding_tables_and_db(
        train_subset,
        data_dir / "dev.tables.jsonl",
        maas_data_dir / "wikisql_train.tables.jsonl"
    )
    
    print("\n" + "=" * 60)
    print("Creating WikiSQL Test Subset")
    print("=" * 60)
    
    # Create test subset (larger for evaluation)
    test_subset = create_stratified_subset(
        data_dir / "test.jsonl",
        data_dir / "test.tables.jsonl",
        maas_data_dir / "wikisql_test.jsonl",
        sample_size=200  # Larger test set
    )
    
    create_corresponding_tables_and_db(
        test_subset,
        data_dir / "test.tables.jsonl",
        maas_data_dir / "wikisql_test.tables.jsonl"
    )
    
    print("\n" + "=" * 60)
    print("Subset Creation Complete!")
    print("=" * 60)
    print(f"\nTraining set: {len(train_subset)} examples")
    print(f"Test set: {len(test_subset)} examples")
    print(f"\nEstimated runtime with sample=4:")
    print(f"  Training: ~{len(train_subset) * 4 * 45 / 3600:.1f} hours")
    print(f"  Testing: ~{len(test_subset) * 4 * 45 / 3600:.1f} hours")
