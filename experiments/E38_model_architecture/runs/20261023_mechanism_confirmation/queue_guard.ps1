$ErrorActionPreference = 'Stop'
$queueLaunch = Get-Content (Join-Path $PSScriptRoot 'launch.json') -Raw | ConvertFrom-Json
$queueActive = Get-Content (Join-Path $PSScriptRoot 'active.json') -Raw | ConvertFrom-Json
if ($queueActive.kind -ne 'E39 rectangle_no_gap seed0 train') { throw 'Active stage changed; inspect before cancelling queue' }
$queueProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = '+$queueLaunch.pid)
$childProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = '+$queueActive.pid)
if ($queueProcess.ExecutablePath -ne $queueLaunch.command[0] -or $queueProcess.CommandLine -notlike ('*'+$queueLaunch.command[3]+'*')) { throw 'Coordinator identity mismatch' }
if ($childProcess.ParentProcessId -ne $queueLaunch.pid -or $childProcess.CommandLine -notlike '*rectangle_no_gap*') { throw 'Active child identity mismatch' }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class TrainingQueueGuard {
 [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
 [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GetProcessTimes(IntPtr h, out long creation, out long exit, out long kernel, out long user);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern uint WaitForSingleObject(IntPtr h, uint milliseconds);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool TerminateProcess(IntPtr h, uint exitCode);
 [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
'@
$queueHandle = [TrainingQueueGuard]::OpenProcess(0x101801,$false,[uint32]$queueLaunch.pid)
if ($queueHandle -eq [IntPtr]::Zero) { throw ('Cannot open coordinator: '+[Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
$childHandle = [TrainingQueueGuard]::OpenProcess(0x101000,$false,[uint32]$queueActive.pid)
if ($childHandle -eq [IntPtr]::Zero) { throw 'Cannot open active pipeline' }
[long]$createdTime=0; [long]$exitTime=0; [long]$kernelTime=0; [long]$userTime=0
if (-not [TrainingQueueGuard]::GetProcessTimes($queueHandle,[ref]$createdTime,[ref]$exitTime,[ref]$kernelTime,[ref]$userTime)) { throw 'Cannot validate coordinator creation time' }
if ([Math]::Abs(([DateTime]::FromFileTimeUtc($createdTime)-([DateTime]$queueLaunch.started_utc).ToUniversalTime()).TotalSeconds) -gt 10) { throw 'Coordinator PID reused' }
if ([TrainingQueueGuard]::WaitForSingleObject($childHandle,0) -ne 258) { throw 'Pipeline already exited; inspect before acting' }
$status = [TrainingQueueGuard]::NtSuspendProcess($queueHandle)
if ($status -ne 0) { throw ('Cannot suspend coordinator: '+$status) }
$record = @{status='pending_queue_cancelled_current_training_continues';user_requested_no_new_training=$true;automation_status='PAUSED';coordinator_pid=$queueLaunch.pid;coordinator_suspended=$true;current_pipeline_pid=$queueActive.pid;current_kind=$queueActive.kind;created_utc=[DateTime]::UtcNow.ToString('o');cancelled_training_jobs=@('base96x4 seed1','deep96x12 seed1','base96x4 seed2','deep96x12 seed2');cleanup='Current pipeline finishes naturally, then only the suspended coordinator is terminated; no new training or evaluation stages'}
$recordPath = Join-Path $PSScriptRoot 'queue_cancelled.json'
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 -LiteralPath $recordPath
Write-Output 'Pending queue disabled; current training continues. Cleanup waits for its natural exit.'
do { $waitResult = [TrainingQueueGuard]::WaitForSingleObject($childHandle,30000) } while ($waitResult -eq 258)
if ($waitResult -ne 0) { throw ('Unexpected wait result: '+$waitResult) }
if (-not [TrainingQueueGuard]::TerminateProcess($queueHandle,0)) { throw 'Pipeline finished; coordinator cleanup failed' }
$record.status='current_training_finished_queue_closed';$record.finished_utc=[DateTime]::UtcNow.ToString('o')
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 -LiteralPath $recordPath
[TrainingQueueGuard]::CloseHandle($childHandle) | Out-Null
[TrainingQueueGuard]::CloseHandle($queueHandle) | Out-Null
Write-Output 'Current training finished; queue closed without starting another stage.'
