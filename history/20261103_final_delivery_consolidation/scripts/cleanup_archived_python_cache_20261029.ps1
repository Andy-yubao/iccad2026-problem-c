$ErrorActionPreference='Stop'
$workspace=[IO.Path]::GetFullPath('E:\project\ICCAD\2026\problemC')
$archive=[IO.Path]::GetFullPath((Join-Path $workspace 'history\20261029_model_research_consolidation'))
if (-not $archive.StartsWith($workspace+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive outside workspace' }
$log=Join-Path $archive 'compiled_cache_cleanup.json'
if (Test-Path -LiteralPath $log) { throw 'Cleanup already recorded' }
$files=@()
$directories=@(Get-ChildItem -LiteralPath (Join-Path $archive 'experiments') -Directory -Filter '__pycache__' -Recurse -Force | Where-Object { -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) })
foreach ($directory in $directories) {
    $checked=[IO.Path]::GetFullPath($directory.FullName)
    if (-not $checked.StartsWith($archive+'\experiments\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Cache directory outside archive' }
    foreach ($file in (Get-ChildItem -LiteralPath $checked -File -Force)) {
        if ($file.Extension -notin @('.pyc','.nbc','.nbi')) { continue }
        $target=[IO.Path]::GetFullPath($file.FullName)
        if (-not $target.StartsWith($checked+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Cache file outside checked directory' }
        $files+=@{path=$file.FullName.Substring($archive.Length+1).Replace('\','/');bytes=$file.Length}
    }
}
@{status='verified_before_delete';scope='Only regenerable CPython/Numba files inside archived __pycache__ directories';files=$files;checkpoint_deletions=0;tensor_cache_deletions=0} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $log -Encoding utf8
foreach ($file in $files) {
    $target=[IO.Path]::GetFullPath((Join-Path $archive $file.path))
    if (-not $target.StartsWith($archive+'\experiments\',[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($target) -notin @('.pyc','.nbc','.nbi')) { throw 'Delete scope validation failed' }
    Remove-Item -LiteralPath $target -Force
}
foreach ($directory in $directories) {
    if ((Get-ChildItem -LiteralPath $directory.FullName -Force | Measure-Object).Count -eq 0) { Remove-Item -LiteralPath $directory.FullName }
}
@{status='complete';scope='Only regenerable CPython/Numba files inside archived __pycache__ directories';files=$files;file_count=$files.Count;bytes=($files | Measure-Object bytes -Sum).Sum;checkpoint_deletions=0;tensor_cache_deletions=0;historical_logs_rewritten=$false} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $log -Encoding utf8
Write-Output ('Cleared '+$files.Count+' archived compiled cache files; no checkpoint or tensor cache deleted')
