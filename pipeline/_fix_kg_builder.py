#!/usr/bin/env python3
"""Fix kg_builder.py: restore original main(), add runtime vars"""
import ast

path = "pipeline/kg_builder.py"
with open(path, "r", encoding="utf-8") as f:
    original = f.read()

# Check for syntax errors first
try:
    ast.parse(original)
    print("Already valid syntax!")
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.msg}")

# We need to:
# 1. Keep all the functions/classes before main()
# 2. Replace the broken main() with a known-good version
# 3. Add runtime override variables after the constants

# Find the exact positions
lines = original.split("\n")

# Find INPUT_DIR line (the one we need to add overrides after)
input_dir_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith("INPUT_DIR ") and "Path(__file__)" in line:
        input_dir_idx = i
        break

print(f"INPUT_DIR at line {input_dir_idx + 1}")

# Find def main():
main_idx = None
for i, line in enumerate(lines):
    if line.strip() == "def main():":
        main_idx = i
        break

print(f"main() at line {main_idx + 1}")

# Insert runtime override vars after INPUT_DIR/OUTPUT_DIR/MAX_PROJECTS block
# The block ends at input_dir_idx + 2 (MAX_PROJECTS line)
override_block = [
    "",
    "# Runtime overrides (set by batch wrapper)",
    "# These are checked by the wrapper script, not used in standalone mode",
    "",
]
insert_pos = input_dir_idx + 3  # after MAX_PROJECTS line

# Check if already inserted
if any("Runtime overrides" in l for l in lines[insert_pos:insert_pos+3]):
    print("Runtime overrides already present")
else:
    for i, line in enumerate(override_block):
        lines.insert(insert_pos + i, line)
    main_idx += len(override_block)  # adjust main position

# Build clean main() — NO f-strings with \n or nested quotes
# Use only % formatting and + concatenation for safety
new_main = [
    'def main():',
    '    """Standalone mode — process first MAX_PROJECTS files from INPUT_DIR"""',
    '    print("=" * 60)',
    '    print("KG Builder — Ontology-Aligned")',
    '    print("=" * 60)',
    '    print("Entity types: %d" % len(ENTITY_TYPES))',
    '    print("Relation types: %d" % len(RELATION_TYPES))',
    '',
    '    if not INPUT_DIR.exists():',
    '        print("[Error] Input dir not found: %s" % INPUT_DIR)',
    '        sys.exit(1)',
    '',
    '    files = sorted(INPUT_DIR.glob("*.txt"))',
    '    print("Input files: %d (processing first %d)" % (len(files), MAX_PROJECTS))',
    '    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)',
    '',
    '    results = []',
    '    for fp in files[:MAX_PROJECTS]:',
    '        proj = parse_project_file(fp)',
    '        if proj:',
    '            r = process_project(proj, OUTPUT_DIR)',
    '            results.append(r)',
    '',
    '    # Summary',
    '    print("")',
    '    print("=" * 60)',
    '    print("All done!")',
    '    for r in results:',
    '        faiss_flag = "Y" if r.get("has_vector_index") else "N"',
    '        print("  %s: %d entities %d relations -> %d datasets [FAISS=%s]" % (',
    '            r["project_name"], r["entity_count"], r["relation_count"],',
    '            r["dataset_count"], faiss_flag))',
    '',
    '    json.dump(results, open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8"),',
    '              ensure_ascii=False, indent=2)',
    '    print("Manifest: %s" % (OUTPUT_DIR / "manifest.json"))',
    '',
]

# Replace from main_idx to end
lines[main_idx:] = new_main

result = "\n".join(lines)

# Verify syntax
try:
    ast.parse(result)
    print("Syntax OK!")
    with open(path, "w", encoding="utf-8") as f:
        f.write(result)
    print("File written successfully")
except SyntaxError as e:
    print("Still has syntax error at line %d: %s" % (e.lineno, e.msg))
    error_lines = result.split("\n")
    for i in range(max(0, e.lineno - 2), min(len(error_lines), e.lineno + 3)):
        print("  %d: %s" % (i + 1, repr(error_lines[i][:120])))
