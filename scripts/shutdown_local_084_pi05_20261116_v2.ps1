param([Parameter(Mandatory=$true)][string]$ClosureReceipt,[Parameter(Mandatory=$true)][string]$AuthorizationReceipt,[switch]$Execute)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$closure=Get-Content -LiteralPath $ClosureReceipt -Raw | ConvertFrom-Json
$auth=Get-Content -LiteralPath $AuthorizationReceipt -Raw | ConvertFrom-Json
$expected='20261116-084-pi05-resume-v2'
if($closure.run_id -ne $expected -or $auth.run_id -ne $expected){throw 'Wrong current round'}
if($auth.authorized -ne $true -or $auth.normal_shutdown -ne $true -or $auth.force_apps -ne $false){throw 'No current normal Windows shutdown authorization'}
if((Get-FileHash -LiteralPath $AuthorizationReceipt -Algorithm SHA256).Hash.ToLowerInvariant() -ne $closure.authorization_sha256){throw 'Authorization SHA differs'}
foreach($name in @('experiment_processes_stopped','gpu_idle','analysis_saved','private_raw_backup_verified','github_remote_verified','remote_native_shutdown_execute_true','monitoring_stopped','summary_sent')){if($closure.$name -ne $true){throw "Closure gate failed: $name"}}
$remote=Get-Content -LiteralPath $closure.remote_shutdown_receipt -Raw | ConvertFrom-Json
if($remote.execute -ne $true -or $remote.hostname -ne $closure.expected_remote_hostname){throw 'Current remote native receipt differs'}
$localProof=Get-Content -LiteralPath $closure.local_raw_verification -Raw | ConvertFrom-Json
$privateProof=Get-Content -LiteralPath $closure.private_backup_receipt -Raw | ConvertFrom-Json
if($localProof.gate -ne 'PASS_CURRENT_RAW_ARCHIVE_MEMBER_SHA' -or $localProof.members -ne $localProof.members_verified){throw 'Raw member recovery gate absent'}
if($privateProof.remote_download_sha256_verified -ne $true -or $privateProof.archive_sha256 -ne $localProof.archive_sha256){throw 'Private full readback gate absent'}
$gitProof=Get-Content -LiteralPath $closure.public_git_verification -Raw | ConvertFrom-Json
if($gitProof.remote_verified -ne $true -or $gitProof.commit -ne $closure.public_git_commit){throw 'Public Git proof differs'}
$head=& git -c "safe.directory=$($closure.repository)" -C $closure.repository rev-parse HEAD
if($LASTEXITCODE -ne 0 -or $head.Trim() -ne $closure.public_git_commit){throw 'Checkout changed since remote verification'}
if(-not $Execute){Write-Output 'PASS_CURRENT_ROUND_WINDOWS_SHUTDOWN_DRY_RUN';exit 0}
Write-Output 'REQUESTING_NORMAL_WINDOWS_SHUTDOWN_NO_FORCE'
& "$env:SystemRoot/System32/shutdown.exe" /s /t 0
if($LASTEXITCODE -ne 0){throw 'Windows shutdown request failed'}
