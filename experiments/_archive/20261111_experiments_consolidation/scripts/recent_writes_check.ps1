$cutoff = (Get-Date).AddMinutes(-30)
$stack = New-Object System.Collections.Stack
$stack.Push('E:\project\ICCAD\2026\problemC\experiments')
$recent = New-Object System.Collections.ArrayList
$dirs = 0
$files = 0
while ($stack.Count -gt 0) {
  $dir = $stack.Pop()
  try { $items = Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop } catch { Write-Output ('ACCESS_FAIL' + "`t" + $dir); continue }
  foreach ($i in $items) {
    if ($i.PSIsContainer) { $dirs++; $stack.Push($i.FullName) }
    else {
      $files++
      if ($i.LastWriteTime -gt $cutoff) { [void]$recent.Add($i) }
    }
  }
}
Write-Output ('SUBDIRS=' + $dirs)
Write-Output ('FILES=' + $files)
Write-Output ('RECENT_30MIN=' + $recent.Count)
foreach ($r in $recent) { Write-Output ('RECENT' + "`t" + $r.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss') + "`t" + $r.FullName) }
