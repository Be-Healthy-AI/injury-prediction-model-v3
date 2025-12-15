# Files That Should Be Committed to GitHub

## ✅ Files That Should Be Added:

### 1. Documentation Files
- `TRANSFERMARKT_DEPENDENCY_ANALYSIS.md` - Analysis document (should be in documentation/ or root)

### 2. Scripts (if not already committed)
- `commit_and_push.ps1` - Utility script for git operations (or add to .gitignore)

## ⚠️ Files That May Still Be in Root (Should Be Moved to to_delete/):

These files should have been moved by the cleanup script. If they still exist in root, they need to be removed:

- `test_match_data_mapping.py`
- `test_raw_match_stats_mapping.py`
- `fix_bundesliga_matches.py`
- `fix_transfermarkt_score_files.py`
- `count_bundesliga_players.py`
- `count_transfermarkt_score_files.py`
- `move_reviewed_files.py`

## 📁 Folders Status:

### data_exports/transfermarkt/
- **Status**: CSV files are ignored by .gitignore (line 28: `*.csv`)
- **This is NORMAL** - CSV data files should not be committed to git
- The folder structure will be tracked, but not the CSV contents
- If you want to track the data structure, you can add empty `.gitkeep` files

### Other Important Folders:
- `scripts/` - Should be committed (all Python files)
- `documentation/` - Should be committed (all .md files)
- `config/` - Should be committed (all .json files)
- `models/` - Check .gitignore (currently commented out, so should be committed)

## 🔍 To Check What's Actually Staged:

Run these commands manually:
```powershell
git status
git status --short
git ls-files --others --exclude-standard
```

## 📝 Recommended Actions:

1. **Add TRANSFERMARKT_DEPENDENCY_ANALYSIS.md**:
   ```powershell
   git add TRANSFERMARKT_DEPENDENCY_ANALYSIS.md
   ```

2. **Decide on commit_and_push.ps1**:
   - Option A: Add it (useful utility)
   - Option B: Add to .gitignore (temporary script)

3. **Verify test/utility files are gone from root**:
   - If they still exist, manually delete them (they're already in to_delete/)

4. **Check if data_exports folder structure should be tracked**:
   - Currently CSV files are ignored (correct)
   - Folder structure might not be tracked if folders are empty
   - Consider adding `.gitkeep` files if you want to track folder structure



