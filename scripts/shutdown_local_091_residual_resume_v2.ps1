param(
    [Parameter(Mandatory=$true)][string]$ClosureReceipt,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$runId = '20261016-091-residual-resume-v2'
$repo = Split-Path -Parent $PSScriptRoot
$authorizationPath = Join-Path $repo "results/experiments/p2-shared-peft/$runId/windows-shutdown-authorization-v2.json"
$authorization = Get-Content -LiteralPath $authorizationPath -Raw | ConvertFrom-Json
if($authorization.windows_shutdown_authorized -ne $true -or
   $authorization.scope -ne "Current $runId execution only") { throw 'No current-run Windows authorization' }
$closure = Get-Content -LiteralPath $ClosureReceipt -Raw | ConvertFrom-Json
if($closure.run_id -ne $runId) { throw 'Closure receipt belongs to a different run' }
$authorizationSha = (Get-FileHash -LiteralPath $authorizationPath -Algorithm SHA256).Hash.ToLowerInvariant()
if($closure.authorization_receipt_sha256 -ne $authorizationSha) { throw 'Authorization SHA does not match current receipt' }
foreach($field in @('all_experiment_processes_stopped','gpu_idle','server_archive_verified',
    'local_archive_verified','analysis_saved','github_remote_verified','summary_sent')) {
    if($closure.$field -ne $true) { throw "Incomplete closure fact: $field" }
}
if($closure.automation_id -ne 'vla-091-vla' -or $closure.automation_status -ne 'PAUSED') {
    throw 'Current heartbeat must be paused first'
}
$remote = Get-Content -LiteralPath $closure.remote_shutdown_receipt_path -Raw | ConvertFrom-Json
if($remote.execute -ne $true -or $remote.backup -ne $closure.server_archive) {
    throw 'Missing matching native remote execute=true receipt'
}
$localRoot = (Resolve-Path -LiteralPath $closure.local_archive).Path.TrimEnd('\','/')
$sumPath = Join-Path $localRoot 'SHA256SUMS.txt'
$lines = @(Get-Content -LiteralPath $sumPath | Where-Object { $_.Trim() })
if($lines.Count -lt 3) { throw 'Complete local archive manifest required' }
foreach($line in $lines) {
    if($line -notmatch '^([a-f0-9]{64})\s+\*?(.+)$') { throw 'Invalid final SHA line' }
    $expected = $matches[1]
    $relative = $matches[2]
    $resolved = (Resolve-Path -LiteralPath (Join-Path $localRoot $relative)).Path
    if(-not $resolved.StartsWith($localRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Final archive checksum path escaped local backup'
    }
    if((Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw 'Final local file SHA changed'
    }
}
$receipt = [ordered]@{run_id=$runId;time_utc=[DateTime]::UtcNow.ToString('o');execute=[bool]$Execute;
    native_command='shutdown.exe /s /t 0';force_apps=$false;closure_receipt=(Resolve-Path -LiteralPath $ClosureReceipt).Path;
    authorization_sha256=$authorizationSha;native_remote_execute=$true;automation_paused=$true}
$json = $receipt | ConvertTo-Json -Depth 5
$json | Write-Output
if($Execute) {
    $out = Join-Path (Split-Path -Parent $ClosureReceipt) 'windows-native-shutdown-request-v2.json'
    [IO.File]::WriteAllText($out,$json,[Text.UTF8Encoding]::new($false))
    & "$env:SystemRoot/System32/shutdown.exe" /s /t 0
    if($LASTEXITCODE -ne 0) { throw 'Native Windows shutdown request failed' }
}
