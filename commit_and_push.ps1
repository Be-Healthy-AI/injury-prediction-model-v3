# PowerShell script to commit and push changes
# Run this script: .\commit_and_push.ps1

Write-Host "Checking git status..." -ForegroundColor Cyan
git status --short

Write-Host "`nStaging all changes..." -ForegroundColor Cyan
git add -A

Write-Host "`nCreating commit..." -ForegroundColor Cyan
git commit -m "Cleanup: Remove test scripts, logs, and old data exports

- Moved 39 unnecessary files to to_delete/ folder (excluded from git)
- Removed test/debug scripts (test_*.py, re_*.py, etc.)
- Removed log files (pipeline.log, etc.)
- Removed old data exports (20251109 folders)
- Removed cache files and temporary outputs
- Kept essential pipeline files and 20251203/20251205 data folders
- Updated .gitignore to exclude cleanup files"

Write-Host "`nPushing to GitHub..." -ForegroundColor Cyan
git push

Write-Host "`nDone!" -ForegroundColor Green



