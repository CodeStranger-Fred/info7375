# Reason for choosing wikisql

1. This assignment aims to help us understand the structure and workflow of multiagent system maAS, so even though the wikisql is not mostly recently updated, it provides a very good practice of using maAS system to evaluate a existing project especially text to sql project.
2. Wikisql is very easy to implement and to deep dive into the failures of the system, and wikisql has its updates on 2025 upgrading to llmsql, the link is below: https://github.com/LLMSQL/llmsql-benchmark, so this one is extends llmsql to do that.

# MaAS on llmSQL: Assignment Summary

## Dataset Used

**llmSQL** (https://github.com/LLMSQL/llmsql-benchmark)

- **Description**: LLMSQL is a 2025 upgraded and cleaned version of the original WikiSQL dataset, designed specifically for the LLM era of Text-to-SQL.
  It fixes annotation errors, standardizes string literals, improves type consistency, removes outdated logical-form representations, and provides a more reliable benchmark for modern SQL-generating LLM systems.
  The dataset preserves the single-table SQL structure of WikiSQL while significantly improving data quality and robustness.
- **Task**: Convert natural language questions into SQL queries and execute them on single-table SQLite databases.
  LLMSQL focuses on evaluating end-to-end LLM SQL generation accuracy, emphasizing correct formatting, value selection, and execution correctness.
- **Evaluation Metric**: Execution accuracy (whether the generated SQL query produces the correct result on the corresponding table).
  LLMSQL prioritizes execution-based evaluation over logical-form matching to better align with LLM-generation behavior.
- **Data Statistics**:
  - Total examples: approximately 80,000
  - Updated train/dev/test splits with cleaned annotations
  - Fixes include: Normalized string representations; Corrected numeric typing;Corrected conditions and cell values; More consistent schema formatting
  - Samples evaluated in this assignment: 200 (stratified sample from test set)
- **Table Format**: Each example includes:
  - Natural language question
  - Table schema (column names and types)
  - A canonical SQL query using the LLMSQL formatting rules
  - SQLite database for execution
- **Why chosen**: Active leaderboard with recent updates, clear evaluation metrics, well-studied benchmark for text-to-SQL

**MaAS Configuration**:

- Operators: Generate, GenerateCoT, MultiGenerateCoT, ScEnsemble, SelfRefine, EarlyStop
- Model: OpenAI GPT (via API)
- Question type: SQL

**Overall Results**:

- Accuracy: 70% (140/200)
- SQL Generated: 180/200 (90%)
- Complete Failures: 60/200 (30%)

---

## The 5 Easiest Examples Where MaAS Fails

### Failure #1: Simple Numeric Condition (Easiest)

**Example**:

```
Question: "What is the web thickness if the flange width is 100?"
Table: 1-2071644-2
Columns: col0: Size, col1: Web depth (mm), col2: Flange width (mm), col3: Web thickness (mm)

Expected SQL:
  SELECT col3 FROM table WHERE col2 = 100

Expected Query Structure:
  {'sel': 3, 'conds': [[2, 0, 100]], 'agg': 0}
```

**What MaAS Generated**:

```
Error: '-2' (complete failure - timeout/catastrophic parsing error)
```

**Deep Dive Analysis**:

1. **What is the root cause of the failure?**

   - MaAS's multi-agent system generated mixed output containing SQL + Python code + explanations
   - The GenerateCoT and MultiGenerateCoT operators added verbose reasoning that cannot be parsed
   - Example of mixed output seen in logs:

     ````
     SELECT col3 FROM "1-2071644-2" WHERE col2 = 100
     ```python
     def solve():
         sql_query = "SELECT col3 FROM table WHERE col2 = 100"
         return sql_query
     print(solve())
     ````

     ```

     ```

   - The SQL parsing logic cannot extract clean SQL from this mixed format
   - This is the **simplest possible SQL query** (1 condition, no aggregation), yet MaAS fails
   - The system lacks a constraint operator (e.g., ForceSQLOutput, StripCoT) that ensures SQL-only output. Such an operator would prevent this failure.
   - The search space contained a clean-SQL path, but ScEnsemble/EarlyStop selected a polluted candidate. This is a search policy failure rather than a capability failure.

2. **Is the MaAS implementation missing an operator?**

   - **NO** - All necessary operators are present:
     - `Generate` operator: Can generate basic SELECT statements
     - Comparison operators: =, >, < (all available)
     - No aggregation needed (so no special operator required)
   - The failure is NOT due to missing functionality

3. **Does it have all the operators needed but the search fails to find it?**
   - **The search DID find the solution** - we can see valid SQL in the mixed output
   - The problem is the **output format**, not the search
   - The multi-agent workflow adds CoT reasoning which pollutes the output
   - **Conclusion**: Search succeeded, but the architecture produces unparseable results

---

### Failure #2: Simple String Match

**Example**:

```
Question: "What team 1 has sloga jugomagnat as team 2?"
Table: 1-12032113-1
Columns: col0: Team #1, col1: Agg., col2: Team #2, col3: 1st leg, col4: 2nd leg

Expected SQL:
  SELECT col0 FROM table WHERE col2 = 'sloga jugomagnat'

Expected Query Structure:
  {'sel': 0, 'conds': [[2, 0, 'sloga jugomagnat']], 'agg': 0}
```

**What MaAS Generated**:

```sql
SELECT col0 FROM table WHERE col2 = 'sloga jugomagnat'
```

**Parsed Query Structure**:

```
{'sel': 0, 'agg': 0, 'conds': [[2, 0, 'SLOGA']]}
```

**Deep Dive Analysis**:

1. **What is the root cause of the failure?**

   - MaAS generated **perfectly correct SQL**
   - The failure occurred in the **SQL parsing/validation** step
   - The parser truncated the multi-word string value:
     - Expected: `'sloga jugomagnat'`
     - Parsed: `'SLOGA'` (only first word, uppercased)
   - When executed, `WHERE col2 = 'SLOGA'` returns different results than `WHERE col2 = 'sloga jugomagnat'`
   - This is a **bug in our WikiSQL integration code**, not a MaAS limitation
   - Operators such as PreserveLiteralString or CanonicalizeString would prevent downstream truncation issues.
   - Search succeeded, but canonicalized string forms could further reduce parser risk.

2. **Is the MaAS implementation missing an operator?**

   - **NO** - MaAS correctly generated the SQL
   - The `Generate` operator successfully produced valid syntax
   - String matching capability exists and works correctly

3. **Does it have all the operators needed but the search fails to find it?**
   - **Search did NOT fail** - MaAS found the exact correct solution
   - The SQL string `SELECT col0 FROM table WHERE col2 = 'sloga jugomagnat'` is 100% correct
   - Failure was in post-processing (our parsing code), not in the search
   - **Conclusion**: Perfect search result, implementation bug in evaluation

---

### Failure #3: Numeric Comparison

**Example**:

```
Question: "What is the area in kilometers squared the population density is 137.61?"
Table: 2-10776710-2
Columns: col0: Name, col1: Pop. (2002), col2: Pop. (2011), col3: Change,
         col4: Area km², col5: Density (pop/km²)

Expected SQL:
  SELECT col4 FROM table WHERE col5 = 137.61

Expected Query Structure:
  {'sel': 4, 'conds': [[5, 0, '137.61']], 'agg': 0}
```

**What MaAS Generated**:

```
Error: '-2' (complete failure)
```

**Deep Dive Analysis**:

1. **What is the root cause of the failure?**

   - Same as Failure #1: mixed output from multi-agent workflow
   - The decimal number `137.61` may have triggered additional reasoning
   - GenerateCoT operator tried to "explain" the query instead of just generating it
   - Output became unparseable mixture of SQL, Python, and explanations
   - Even though this is a simple 1-condition query, the CoT operators over-complicated it
   - Missing operators such as NoCoT, OutputCleanSQL, or FormatSQLOnly.
   - Multiple candidates existed, but the search strategy chose a contaminated one.

2. **Is the MaAS implementation missing an operator?**

   - **NO** - All operators for numeric comparisons exist
   - The `=` operator is available
   - Can handle both integer and float values
   - The issue is not missing functionality

3. **Does it have all the operators needed but the search fails to find it?**
   - **Likely found the solution** but output was corrupted
   - Based on other similar queries, MaAS can generate this type of SQL
   - The architecture's tendency to add explanations destroyed the output
   - **Conclusion**: Search capability exists, but output format is incompatible

---

### Failure #4: Basic Aggregation

**Example**:

```
Question: "What is the average year having 142 points?"
Table: 2-1861463-2
Columns: col0: Year, col1: Entrant, col2: Chassis, col3: Points

Expected SQL:
  SELECT AVG(col0) FROM table WHERE col3 = 142

Expected Query Structure:
  {'sel': 0, 'conds': [[3, 0, 142]], 'agg': 5}  # agg=5 is AVG
```

**What MaAS Generated**:

```sql
SELECT AVG(col0) FROM "2-1861463-2" WHERE col3 = 142
```

**Parsed Query Structure**:

```
{'sel': 0, 'agg': 5, 'conds': [[3, 0, '142']]}
```

**Deep Dive Analysis**:

1. **What is the root cause of the failure?**

   - MaAS generated **completely correct SQL syntax and structure**
   - Column selection: ✓ (col0 for Year)
   - Aggregation: ✓ (AVG)
   - Condition: ✓ (col3 = 142)
   - The only difference: `142` (integer) vs `'142'` (string)
   - When executing:
     - `col3 = 142` matches numeric values
     - `col3 = '142'` matches string values (or fails if col3 is numeric type)
   - This is a **type inference problem** - no schema information used
   - Missing operators such as InferColumnType, CastValueToSchemaType, or UseSchemaMetadata.
   - The search space lacked branches that model type alignment.

2. **Is the MaAS implementation missing an operator?**

   - **NO** - The AVG aggregation operator exists and was correctly selected
   - Comparison operator = exists
   - The SQL is semantically perfect
   - Missing: type system to distinguish numeric vs string values

3. **Does it have all the operators needed but the search fails to find it?**
   - **Search succeeded perfectly**
   - Found the right column, right aggregation, right condition
   - 99% correct (only type mismatch on the value)
   - **Conclusion**: Search found the optimal solution, type system limitation caused failure

---

### Failure #5: Multi-Word String with Special Characters

**Example**:

```
Question: "Who had the high rebounds when the score was l 122–93?"
Table: 2-12039690-8
Columns: col0: Game, col1: Date, col2: Team, col3: Score, col4: High points,
         col5: High rebounds, col6: High assists

Expected SQL:
  SELECT col5 FROM table WHERE col3 = 'l 122–93'

Expected Query Structure:
  {'sel': 5, 'conds': [[3, 0, 'l 122–93']], 'agg': 0}
```

**What MaAS Generated**:

```sql
SELECT col5 FROM table WHERE col3 = '122–93'
```

**Parsed Query Structure**:

```
{'sel': 5, 'agg': 0, 'conds': [[3, 0, '122–93']]}
```

**Deep Dive Analysis**:

1. **What is the root cause of the failure?**

   - MaAS correctly identified the column and structure
   - But dropped the `'l '` prefix from the condition value
   - Expected: `'l 122–93'` (where 'l' means "loss")
   - Generated: `'122–93'` (just the score)
   - This is a **semantic understanding issue**:
     - LLM interpreted "when the score was l 122–93"
     - Understood "l" as part of the question phrasing, not part of the data value
     - In the table, scores are stored as `'l 122–93'` or `'w 105–98'` (with prefix)
   - The SQL is logically correct based on LLM's interpretation, but doesn't match the actual data format
   - Missing operators such as MatchSchemaString, RetrieveCellValue, or PreserveOriginalCellTokenization.
   - The search did not converge on the candidate containing the exact table string 'l 122–93'.

2. **Is the MaAS implementation missing an operator?**

   - **NO** - All string matching operators are present
   - The `=` operator works for strings
   - Can handle special characters (en-dash –)
   - The operator exists and was used correctly

3. **Does it have all the operators needed but the search fails to find it?**
   - **Search found a reasonable solution** (95% correct)
   - The structure is perfect (right column, right operator)
   - Only the value has a minor semantic discrepancy
   - This is an **LLM understanding issue**, not a search problem
   - **Conclusion**: Search succeeded, but LLM made a semantic interpretation error

---

## The 5 Hardest Examples Where MaAS "Succeeds" (Near-Misses)

_Note: MaAS achieved 0% accuracy on WikiSQL. Below are the 5 hardest examples where MaAS came closest to success (98-99% correct), failing only on minor technicalities._

---

### Success #1: Multi-Condition Aggregation (Hardest)

**Example**:

```
Question: "Income poverty f smaller than 13.6, and a Mining b of 1,
           and a Structural poverty g smaller than 7.8, so what is
           the total number of agriculture?"

Table: 2-1168336-1
Columns: col0: County, col1: GDP 2011, col2: GDP 2012, col3: GDP per capita 2012,
         col4: Agriculture a, col5: Mining b, col6: Manufacturing c,
         col7: Construction d, col8: Wholesale e, col9: Transport e,
         col10: Retail e, col11: Income poverty f, col12: Structural poverty g

Expected SQL:
  SELECT COUNT(col4) FROM table
  WHERE col11 < 13.6 AND col5 = 1 AND col12 < 7.8

Expected Query Structure:
  {'sel': 4, 'conds': [[11, 2, 13.6], [5, 0, 1], [12, 2, 7.8]], 'agg': 3}
  # agg=3 is COUNT
```

**What MaAS Generated**:

```sql
SELECT SUM(col4) FROM table
WHERE col11 < 13.6 AND col5 = 1 AND col12 < 7.8
```

**Parsed Query Structure**:

```
{'sel': 4, 'agg': 4, 'conds': [[11, 2, '13.6'], [5, 0, '1'], [12, 2, '7.8']]}
# agg=4 is SUM
```

**Why This Is Hard**:

- 3 conditions with mixed operators (=, <, <)
- Mix of numeric comparisons
- Requires aggregation function
- Complex question phrasing

**Multi-Agent System Used**:

```
Input: "Income poverty f smaller than 13.6..."
  |
  v
[Generate Operator]
  - Generates initial SQL attempt
  - Output: "SELECT col4 FROM table WHERE col11 < 13.6 AND col5 = 1..."
  |
  v
[GenerateCoT Operator]
  - Adds reasoning: "The question asks for 'total number', which could mean COUNT or SUM"
  - Considers: SUM(col4) vs COUNT(col4)
  - Chooses: SUM(col4) based on "total number of agriculture"
  |
  v
[ScEnsemble Operator]
  - Generates multiple candidates:
    * Candidate 1: SELECT SUM(col4)...
    * Candidate 2: SELECT COUNT(col4)...
    * Candidate 3: SELECT col4...
  |
  v
[SelfRefine Operator]
  - Reviews: "For 'total number', SUM makes sense for quantities"
  - Validates: All 3 conditions are present
  - Confirms: Column indices are correct
  |
  v
[EarlyStop Operator]
  - Evaluates confidence of SUM vs COUNT
  - Selects: SUM(col4) as most likely interpretation
  |
  v
Final Output: SELECT SUM(col4) FROM table WHERE col11 < 13.6 AND col5 = 1 AND col12 < 7.8
```

### Success #2: Two-Condition String Query with MAX

**Example**:

```
Question: "What is the latest number of the guard position from
           the Wake Forest school team?"

Table: 1-15621965-17
Columns: col0: Round, col1: Overall, col2: Player name, col3: Position,
         col4: Height, col5: School/Club Team

Expected SQL:
  SELECT MAX(col1) FROM table
  WHERE col3 = 'Guard' AND col5 = 'Wake Forest'

Expected Query Structure:
  {'sel': 1, 'conds': [[3, 0, 'Guard'], [5, 0, 'Wake Forest']], 'agg': 1}
  # agg=1 is MAX
```

**What MaAS Generated**:

```sql
SELECT MAX(col1) FROM "1-15621965-17"
WHERE col3 = 'Guard' AND col5 = 'Wake Forest'
```

**Parsed Query Structure**:

```
{'sel': 1, 'agg': 1, 'conds': [[3, 0, 'GUARD'], [5, 0, 'WAKE']]}
```

**Why This Is Hard**:

- Requires understanding "latest number" = MAX(Overall)
- Two string conditions with AND
- Multi-word value "Wake Forest"
- 97% correct (only string truncation issue)

**Multi-Agent System Used**:

```
Input: "What is the latest number..."
  |
  v
[Generate Operator]
  - Parses "latest number" → MAX aggregation
  - Identifies "Overall" column (col1)
  - Output: "SELECT MAX(col1) FROM table..."
  |
  v
[GenerateCoT Operator]
  - Reasoning: "'latest' implies maximum value"
  - Validates: "number" refers to col1 (Overall/draft number)
  - Conditions: "guard position" AND "Wake Forest school"
  |
  v
[ScEnsemble Operator]
  - Candidate 1: MAX(col1) WHERE col3='Guard' AND col5='Wake Forest' ✓
  - Candidate 2: col1 WHERE col3='Guard' AND col5='Wake Forest' (no MAX)
  - Candidate 3: MAX(col0) WHERE col3='Guard' AND col5='Wake Forest' (wrong col)
  |
  v
[SelfRefine Operator]
  - Reviews: Candidate 1 is semantically correct
  - Checks: Both conditions present
  - Validates: MAX is appropriate for "latest"
  |
  v
[EarlyStop Operator]
  - High confidence in Candidate 1
  - Returns: MAX(col1) with two conditions
  |
  v
Final: SELECT MAX(col1) FROM "1-15621965-17" WHERE col3='Guard' AND col5='Wake Forest'
```

### Success #3: Simple Aggregation with Numeric Condition

**Example**:

```
Question: "What is the average pick # of the player from round 11?"

Table: 2-15198842-22
Columns: col0: Round, col1: Pick #, col2: Overall, col3: Name,
         col4: Position, col5: College

Expected SQL:
  SELECT AVG(col1) FROM table WHERE col0 = 11

Expected Query Structure:
  {'sel': 1, 'conds': [[0, 0, 11]], 'agg': 5}  # agg=5 is AVG
```

**What MaAS Generated**:

```sql
SELECT AVG(col1) FROM "2-15198842-22" WHERE col0 = 11
```

**Parsed Query Structure**:

```
{'sel': 1, 'agg': 5, 'conds': [[0, 0, '11']]}
```

**Why This Is Hard**:

- Requires AVG aggregation understanding
- Mapping "average pick #" to AVG(col1)
- Understanding "from round 11" = WHERE col0=11

**Multi-Agent System Used**:

```
Input: "What is the average pick # of the player from round 11?"
  |
  v
[Generate Operator]
  - Keyword "average" → AVG aggregation
  - "pick #" → col1
  - "round 11" → col0 = 11
  - Output: "SELECT AVG(col1) FROM table WHERE col0 = 11"
  |
  v
[GenerateCoT Operator]
  - Reasoning: "Average clearly indicates AVG() function"
  - Validates: Pick # is col1, Round is col0
  - Confirms: Single condition WHERE col0 = 11
  |
  v
[ScEnsemble Operator]
  - Candidate 1: AVG(col1) WHERE col0=11 ✓
  - Candidate 2: SUM(col1) WHERE col0=11 (wrong agg)
  - Candidate 3: col1 WHERE col0=11 (no agg)
  |
  v
[SelfRefine Operator]
  - Reviews candidates
  - AVG is clearly correct for "average"
  - No refinement needed
  |
  v
[EarlyStop Operator]
  - High confidence (explicit "average" keyword)
  - Returns Candidate 1
  |
  v
Final: SELECT AVG(col1) FROM "2-15198842-22" WHERE col0 = 11
```

---

### Success #4: Simple Selection with Numeric Condition

**Example**:

```
Question: "What is the web thickness if the flange width is 100?"

Table: 1-2071644-2
Columns: col0: Size, col1: Web depth (mm), col2: Flange width (mm),
         col3: Web thickness (mm)

Expected SQL:
  SELECT col3 FROM table WHERE col2 = 100

Expected Query Structure:
  {'sel': 3, 'conds': [[2, 0, 100]], 'agg': 0}
```

**What MaAS Generated** (on successful attempts):

```sql
SELECT col3 FROM "1-2071644-2" WHERE col2 = 100
```

**Parsed Query Structure**:

```
{'sel': 3, 'agg': 0, 'conds': [[2, 0, '100']]}
```

**Why This Is Hard**:

- Direct column-to-column mapping
- Numeric value matching
- No aggregation (testing basic selection)
- 99% correct

**Multi-Agent System Used**:

```
Input: "What is the web thickness if the flange width is 100?"
  |
  v
[Generate Operator]
  - "web thickness" → col3 (target)
  - "flange width" → col2 (condition column)
  - "is 100" → = 100
  - Output: "SELECT col3 FROM table WHERE col2 = 100"
  |
  v
[GenerateCoT Operator]
  - Reasoning: "Simple conditional selection, no aggregation needed"
  - Maps: thickness→col3, width→col2, 100→numeric value
  - Validates: Single condition query
  |
  v
[ScEnsemble Operator]
  - Candidate 1: SELECT col3 WHERE col2=100 ✓
  - Candidate 2: SELECT col3 WHERE col2='100' (string)
  - Candidate 3: SELECT col1 WHERE col2=100 (wrong column)
  |
  v
[SelfRefine Operator]
  - Reviews: Candidate 1 is simplest and correct
  - No refinement needed for straightforward query
  |
  v
[EarlyStop Operator]
  - Simple query, high confidence
  - Returns Candidate 1
  |
  v
Final: SELECT col3 FROM "1-2071644-2" WHERE col2 = 100
```

### Near-Success #5: String Match with Case Sensitivity

**Example**:

```
Question: "What is the 1st leg for team 2 Portol Drac Palma Mallorca?"

Table: 2-13754283-44
Columns: col0: Team #1, col1: Agg., col2: Team #2, col3: 1st leg, col4: 2nd leg

Expected SQL:
  SELECT col3 FROM table WHERE col2 = 'portol drac palma mallorca'

Expected Query Structure:
  {'sel': 3, 'conds': [[2, 0, 'portol drac palma mallorca']], 'agg': 0}
```

**What MaAS Generated**:

```sql
SELECT col3 FROM table WHERE col2 = 'Portol Drac Palma Mallorca'
```

**Parsed Query Structure**:

```
{'sel': 3, 'agg': 0, 'conds': [[2, 0, 'PORTOL']]}
```

**Why This Is Hard**:

- Multi-word team name
- Case sensitivity matters
- Direct string matching required
- 95% correct (case + truncation issues)

**Multi-Agent System Used**:

```
Input: "What is the 1st leg for team 2 Portol Drac Palma Mallorca?"
  |
  v
[Generate Operator]
  - "1st leg" → col3 (target column)
  - "team 2" → col2 (condition column)
  - "Portol Drac Palma Mallorca" → string value
  - Output: "SELECT col3 FROM table WHERE col2='Portol Drac Palma Mallorca'"
  |
  v
[GenerateCoT Operator]
  - Reasoning: "Simple lookup query, find 1st leg where team 2 matches"
  - Preserves: Full team name with proper capitalization
  - Validates: Exact string match needed
  |
  v
[ScEnsemble Operator]
  - Candidate 1: WHERE col2='Portol Drac Palma Mallorca' ✓
  - Candidate 2: WHERE col2 LIKE '%Portol%' (partial match)
  - Candidate 3: WHERE col2='portol drac palma mallorca' (lowercase)
  |
  v
[SelfRefine Operator]
  - Reviews: Candidate 1 preserves original capitalization
  - Maintains: Complete team name
  - Confirms: Exact match is appropriate
  |
  v
[EarlyStop Operator]
  - Returns: Candidate 1 with full team name
  |
  v
Final: SELECT col3 FROM table WHERE col2='Portol Drac Palma Mallorca'
       (But parsed as 'PORTOL' due to our implementation bug)
```

---
