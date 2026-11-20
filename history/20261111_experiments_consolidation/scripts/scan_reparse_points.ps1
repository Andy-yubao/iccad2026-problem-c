$ErrorActionPreference = 'Continue'
$stack = New-Object System.Collections.Stack
$stack.Push('E:\project\ICCAD\2026\problemC\experiments')
$reparse = New-Object System.Collections.ArrayList
$dirs = 0
$files = 0
while ($stack.Count -gt 0) {
  $dir = $stack.Pop()
  try { $items = Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop } catch { Write-Output ('ACCESS_FAIL' + "`t" + $dir); continue }
  foreach ($i in $items) {
    if ($i.Attributes -band [IO.FileAttributes]::ReparsePoint) { [void]$reparse.Add($i) }
    elseif ($i.PSIsContainer) { $dirs++; $stack.Push($i.FullName) }
    else { $files++ }
  }
}
Write-Output ('TOTAL_SUBDIRS=' + $dirs)
Write-Output ('TOTAL_FILES=' + $files)
Write-Output ('REPARSE_COUNT=' + $reparse.Count)
foreach ($r in $reparse) { Write-Output ('REPARSE' + "`t" + $r.FullName + "`tTARGET=" + $r.Target) }
