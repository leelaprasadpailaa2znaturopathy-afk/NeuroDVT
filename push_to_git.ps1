# NeuroDVT Git Helper
git add .
$msg = Read-Host -Prompt 'Enter commit message (or press Enter for default)'
if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "Update NeuroDVT Project" }
git commit -m $msg
git push origin main
Write-Host "🚀 Pushed to GitHub! Now check the 'Actions' tab on your GitHub page." -ForegroundColor Green
