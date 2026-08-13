import json
import pickle
import requests
import os
from collections import defaultdict

print("="*60)
print("LEVEL 2 COMMUNITY SUMMARIZATION (GraphRAG)")
print("="*60)

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

# Test mode: set to True to process only TEST_SIZE communities
# Set to False to process all communities
TEST_MODE = True
TEST_SIZE = 10

# Input files
COMMUNITIES_FILE = "communities_3/level_2_communities.json"
GRAPH_FILE = "giant_component_directed.pkl"

# Output directory
OUTPUT_DIR = "summaries/level_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """You are a biological systems expert analyzing a Gene Ontology knowledge graph community.

Generate a comprehensive community report describing the biological mechanisms represented.


IMPORTANT RULES:
- Use ONLY provided entities and relationships
- Do NOT invent information not present in the input
- Be comprehensive - cover all significant aspects
- Each finding should have a short summary and detailed explanation (2-3 paragraphs)

Provide 5-10 findings.

Return output as valid JSON:
{{
  "title": "<short specific title representing key entities>",
  "summary": "<executive summary of community structure and entity relationships>",
  "findings": [
    {{
      "summary": "<short insight>",
      "explanation": "<detailed explanation with 2-3 paragraphs>"
    }}
  ],
  "rating": <float between 0-10 representing biological importance>,
  "rating_explanation": "<single sentence explaining the rating>"
}}

Return ONLY valid JSON, no additional text.

---Input---

Entities
id, entity
{entities_table}

Relationships
id, source, target, relationship
{relationships_table}

---Output---
"""

# ============================================================
# LOAD DATA
# ============================================================

print(f"\nLoading communities from {COMMUNITIES_FILE}...")
with open(COMMUNITIES_FILE) as f:
    communities = json.load(f)

print(f"Loading directed graph from {GRAPH_FILE}...")
with open(GRAPH_FILE, 'rb') as f:
    G = pickle.load(f)

total_communities = len(communities)
print(f"✓ {total_communities:,} communities loaded")

if TEST_MODE:
    print(f"\n⚠️  TEST MODE: Processing only {TEST_SIZE} communities")
    print(f"   Set TEST_MODE=False to process all {total_communities:,} communities")

# ============================================================
# SUMMARIZATION FUNCTION
# ============================================================

def summarize_community(comm_id, nodes):
    """
    Generate GraphRAG summary for a single community.
    
    Returns:
        dict or None: Summary JSON if successful, None if failed
    """
    
    # Build entities table
    entities_lines = []
    for i, node in enumerate(nodes):
        entities_lines.append(f"{i}, {node}")
    entities_table = "\n".join(entities_lines)
    
    # Build relationships table
    relationships_lines = []
    rel_id = 0
    for u, v, data in G.edges(data=True):
        if u in nodes and v in nodes:
            rel = data.get('relationship', 'related')
            relationships_lines.append(f"{rel_id}, {u}, {v}, {rel}")
            rel_id += 1
    
    if relationships_lines:
        relationships_table = "\n".join(relationships_lines)
    else:
        relationships_table = "(no relationships within community)"
    print(entities_table)
    print(relationships_table)
    # Build complete prompt
    prompt = PROMPT_TEMPLATE.format(
        entities_table=entities_table,
        relationships_table=relationships_table
    )
    
    # Call Ollama API
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 3000  # Allow longer responses for detailed findings
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        
        # Extract response text
        response_text = result.get('response', '').strip()
        
        # Clean markdown code blocks if present
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response_text = '\n'.join(lines).strip()
        
        # Parse JSON
        summary = json.loads(response_text)
        
        # Validate required fields
        required_fields = ['title', 'summary', 'findings', 'rating', 'rating_explanation']
        missing = [f for f in required_fields if f not in summary]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Validate findings structure
        if not isinstance(summary['findings'], list):
            raise ValueError("findings must be a list")
        
        for i, finding in enumerate(summary['findings']):
            if not isinstance(finding, dict):
                raise ValueError(f"finding {i} must be a dict")
            if 'summary' not in finding or 'explanation' not in finding:
                raise ValueError(f"finding {i} missing summary or explanation")
        
        # Validate rating range
        if not (0 <= summary['rating'] <= 10):
            raise ValueError(f"rating {summary['rating']} not in range 0-10")
        
        return summary
        
    except requests.exceptions.Timeout:
        print(f"\n  ⚠️  Timeout for {comm_id}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n  ⚠️  API error for {comm_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"\n  ⚠️  Invalid JSON for {comm_id}: {e}")
        print(f"  Response: {response_text[:200]}...")
        return None
    except ValueError as e:
        print(f"\n  ⚠️  Validation error for {comm_id}: {e}")
        return None
    except Exception as e:
        print(f"\n  ⚠️  Unexpected error for {comm_id}: {e}")
        return None

# ============================================================
# PROCESS COMMUNITIES
# ============================================================

print(f"\n{'='*60}")
print("GENERATING SUMMARIES")
print(f"{'='*60}\n")

processed = 0
successful = 0
failed = 0
skipped = 0

# Get list of communities to process
comm_items = list(communities.items())
if TEST_MODE:
    comm_items = comm_items[:TEST_SIZE]

for comm_id, nodes in comm_items:
    processed += 1
    
    # Create safe filename (replace dots with underscores)
    safe_filename = comm_id.replace('.', '_') + '.json'
    output_file = os.path.join(OUTPUT_DIR, safe_filename)
    
    # Skip if already exists
    if os.path.exists(output_file):
        skipped += 1
        print(f"[{processed}/{len(comm_items)}] SKIP {comm_id} (already exists)")
        continue
    
    print(f"[{processed}/{len(comm_items)}] {comm_id} ({len(nodes)} nodes)...", end=" ", flush=True)
    
    summary = summarize_community(comm_id, nodes)
    
    if summary:
        # Save immediately
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        successful += 1
        print("✓")
    else:
        failed += 1
        print("✗ FAILED")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"\nTotal processed  : {processed}")
print(f"Successful       : {successful}")
print(f"Failed           : {failed}")
print(f"Skipped (exists) : {skipped}")

if successful > 0:
    print(f"\nSummaries saved to: {OUTPUT_DIR}/")
    print(f"\nExample files:")
    saved_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')][:3]
    for f in saved_files:
        print(f"  - {f}")

if TEST_MODE:
    print(f"\n{'='*60}")
    print("⚠️  TEST MODE ACTIVE")
    print(f"{'='*60}")
    print(f"\nTo process all {total_communities:,} communities:")
    print(f"1. Review the test results above")
    print(f"2. Edit this script and set: TEST_MODE = False")
    print(f"3. Run again (it will skip already-completed communities)")
    print(f"\nEstimated time for full run: {total_communities * 30 / 3600:.1f} hours")

print(f"\n{'='*60}")
print("✓ LEVEL 2 SUMMARIZATION COMPLETE")
print(f"{'='*60}")