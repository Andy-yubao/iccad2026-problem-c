$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath('E:\project\ICCAD\2026\problemC')
$archive = [IO.Path]::GetFullPath((Join-Path $workspace 'history\20261029_model_research_consolidation'))
if (-not $archive.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Archive outside workspace' }
$names = @('E61_explosion_replay','E65_geometry_messages','E68_port_tokens','E70_reference_task_loss')
New-Item -ItemType Directory -Path (Join-Path $archive 'experiments') -Force | Out-Null
$manifest = @()
foreach ($name in $names) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspace ('experiments\' + $name)))
    $dest = [IO.Path]::GetFullPath((Join-Path $archive ('experiments\' + $name)))
    if (-not $source.StartsWith($workspace + '\experiments\', [StringComparison]::OrdinalIgnoreCase) -or -not $dest.StartsWith($archive + '\experiments\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Move target outside checked directories' }
    if (-not (Test-Path -LiteralPath $source)) { throw ('Source absent: ' + $source) }
    if (Test-Path -LiteralPath $dest) { throw ('Destination already exists: ' + $dest) }
    $files = @(Get-ChildItem -LiteralPath $source -File -Recurse -Force)
    $inventory = @($files | ForEach-Object { @{path=$_.FullName.Substring($source.Length+1).Replace('\','/'); bytes=$_.Length; utc_ticks=$_.LastWriteTimeUtc.Ticks} })
    $manifest += @{old_path=('experiments/'+$name);new_path=('history/20261029_model_research_consolidation/experiments/'+$name);file_count=$files.Count;bytes=($files | Measure-Object Length -Sum).Sum;files=$inventory}
}
@{status='verified_before_move';moves=$manifest;historical_files_rewritten=$false;deletions=0;active_training='E74 complete; E75 preflight complete; archived experiments all terminal';test_read=$false} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $archive 'migration_after_E74.json') -Encoding utf8
foreach ($record in $manifest) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspace $record.old_path))
    $dest = [IO.Path]::GetFullPath((Join-Path $workspace $record.new_path))
    if (-not $source.StartsWith($workspace + '\experiments\', [StringComparison]::OrdinalIgnoreCase) -or -not $dest.StartsWith($archive + '\experiments\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Move validation failed' }
    Move-Item -LiteralPath $source -Destination $dest
    $after = @(Get-ChildItem -LiteralPath $dest -File -Recurse -Force)
    if ($after.Count -ne $record.file_count) { throw 'File count changed during move' }
    foreach ($file in $record.files) {
        $actual = Get-Item -LiteralPath (Join-Path $dest $file.path)
        if ($actual.Length -ne $file.bytes -or $actual.LastWriteTimeUtc.Ticks -ne $file.utc_ticks) { throw ('Metadata changed: '+$actual.FullName) }
    }
    Write-Output ('Archived ' + $record.old_path + ' (' + $record.file_count + ' files)')
}
@{status='complete';moves=$manifest;verification='Every file exists at destination with same length and UTC modification time; same-volume directory move';historical_files_rewritten=$false;deletions=0;test_read=$false} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $archive 'migration_after_E74.json') -Encoding utf8
