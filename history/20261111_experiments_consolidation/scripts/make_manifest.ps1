param(
  [string]$Root = 'E:\project\ICCAD\2026\problemC\experiments',
  [string]$OutCsv = 'E:\project\ICCAD\2026\problemC\tmp\premove_manifest.csv',
  [string]$Mode = 'pre'
)
$keep = @('E119_global_initial_generator', 'E120_node12_collision')
$hashLimit = 8MB
$sw = New-Object System.IO.StreamWriter($OutCsv, $false)
$sw.WriteLine("relpath`tsize`tmtime_utc`tsha256")
$perDir = @{}
$totalFiles = 0
$totalBytes = [long]0
$hashed = 0
$topItems = Get-ChildItem -LiteralPath $Root -Force | Sort-Object Name
foreach ($top in $topItems) {
  if ($keep -contains $top.Name) { continue }
  if ($top.Name -in @('README.md', '__init__.py')) { continue }
  if ($top.Name -eq '__pycache__') { Write-Output ('PYCACHE_PRESENT' + "`t" + $top.FullName); continue }
  $dirFiles = 0
  $dirBytes = [long]0
  if ($top.PSIsContainer) {
    $stack = New-Object System.Collections.Stack
    $stack.Push($top.FullName)
    while ($stack.Count -gt 0) {
      $dir = $stack.Pop()
      try { $items = Get-ChildItem -LiteralPath $dir -Force -ErrorAction Stop } catch { Write-Output ('ACCESS_FAIL' + "`t" + $dir); continue }
      foreach ($i in $items) {
        if ($i.PSIsContainer) { $stack.Push($i.FullName) }
        else {
          $rel = $i.FullName.Substring($Root.Length + 1)
          $sha = ''
          if ($i.Length -le $hashLimit) {
            $sha = (Get-FileHash -LiteralPath $i.FullName -Algorithm SHA256).Hash
            $hashed++
          }
          $sw.WriteLine($rel + "`t" + $i.Length + "`t" + $i.LastWriteTimeUtc.ToString('o') + "`t" + $sha)
          $dirFiles++; $dirBytes += $i.Length
        }
      }
    }
  } else {
    $rel = $top.FullName.Substring($Root.Length + 1)
    $sha = ''
    if ($top.Length -le $hashLimit) { $sha = (Get-FileHash -LiteralPath $top.FullName -Algorithm SHA256).Hash; $hashed++ }
    $sw.WriteLine($rel + "`t" + $top.Length + "`t" + $top.LastWriteTimeUtc.ToString('o') + "`t" + $sha)
    $dirFiles++; $dirBytes += $top.Length
  }
  $perDir[$top.Name] = @($dirFiles, $dirBytes)
  $totalFiles += $dirFiles
  $totalBytes += $dirBytes
}
$sw.Close()
Write-Output ('TOTAL_FILES=' + $totalFiles)
Write-Output ('TOTAL_BYTES=' + $totalBytes)
Write-Output ('HASHED_FILES=' + $hashed)
foreach ($k in ($perDir.Keys | Sort-Object)) { Write-Output ('DIRSTATS' + "`t" + $k + "`t" + $perDir[$k][0] + "`t" + $perDir[$k][1]) }
