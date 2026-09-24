param(
    [Parameter(Mandatory=$true)][string[]]$ResultDirectory,
    [Parameter(Mandatory=$true)][string]$Label,
    [Parameter(Mandatory=$true)][int]$ExpectedVideos,
    [Parameter(Mandatory=$true)][string]$ReportRelativeRoot,
    [Parameter(Mandatory=$true)][string]$RawRelativeRoot,
    [string]$ExpectedHostname='autodl-container-8b78499521-41be183e',
    [string]$SshHost='connect.nmb1.seetacloud.com',
    [ValidateRange(1,65535)][int]$SshPort=31263,
    [string]$IdentityFile='C:/Users/zsure/.ssh/id_ed25519_vla_014',
    [string]$KnownHostsFile,
    [string]$RemoteToolDirectory='/root/autodl-tmp/qvla-repro/closure-tools',
    [string]$ResultPrefix='awq-scope-shard-',
    [string]$FallbackBackupDirectory,
    [switch]$UseVerifiedFallbackOnly,
    [switch]$Execute
)
# One explicitly approved finite closure. No boot, login, timer or deletion.
$ErrorActionPreference='Stop'
$taskRepo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if($Label -notmatch '^[A-Za-z0-9_-]+$' -or $ExpectedVideos -lt 1){throw 'Invalid closure label/video count'}
if($ExpectedHostname -notmatch '^[A-Za-z0-9-]+$' -or $SshHost -notmatch '^[A-Za-z0-9.-]+$' -or
   $RemoteToolDirectory -notmatch '^/root/autodl-tmp/qvla-repro/[A-Za-z0-9_./-]+$' -or
   $ResultPrefix -notmatch '^[A-Za-z0-9_-]+$'){throw 'Invalid closure target'}
if($FallbackBackupDirectory -and $FallbackBackupDirectory -notmatch '^/root/autodl-tmp/qvla-repro/backups/[A-Za-z0-9_-]+$'){throw 'Invalid fallback backup path'}
if($UseVerifiedFallbackOnly -and -not $FallbackBackupDirectory){throw 'Fast shutdown requires a concrete verified fallback backup'}
foreach($taskDir in $ResultDirectory){
    if($taskDir -notmatch '^/root/autodl-tmp/qvla-repro/eval/[A-Za-z0-9_-]+$'){
        throw 'Only concrete finite eval result paths are accepted'
    }
}
if($ReportRelativeRoot -notmatch '^reports/sessions/[A-Za-z0-9_-]+$' -or
   $RawRelativeRoot -notmatch '^results/[A-Za-z0-9_-]+$'){throw 'Invalid workspace report/raw paths'}
$taskReport=Join-Path $taskRepo $ReportRelativeRoot
$taskRaw=Join-Path $taskRepo $RawRelativeRoot
$taskPython='C:/Users/zsure/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$taskSsh=@('-i',$IdentityFile,'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=12')
if($KnownHostsFile){$taskSsh+=@('-o',"UserKnownHostsFile=$KnownHostsFile")}
$taskRemotePython='/root/miniconda3/envs/qvla-oft/bin/python'
$taskRemoteScripts=$RemoteToolDirectory
if(-not $Execute){
    [pscustomobject]@{execute=$false;expected_hostname=$ExpectedHostname;results=$ResultDirectory;
        expected_videos=$ExpectedVideos;steps=@('identity','remote closure tool SHA verification','paired audit','persistent archive',
        'local copy and SHA256','GitHub zsure27 account verification and push',
        'native shutdown receipt','post-shutdown receipt push');deletes_data=$false} | ConvertTo-Json -Depth 4
    return
}
function Invoke-TaskSsh([string]$command){
    $taskOutput=& ssh @taskSsh -p $SshPort "root@$SshHost" $command
    if($LASTEXITCODE -ne 0){throw 'SSH step failed; preserve all data and report server may still run'}
    return $taskOutput
}
function Invoke-TaskGitSnapshot([string]$message){
    & $taskPython (Join-Path $taskReport 'manifest.py')
    if($LASTEXITCODE -ne 0){throw 'Report manifest generation failed'}
    & $taskPython (Join-Path $taskReport 'manifest.py') --verify
    if($LASTEXITCODE -ne 0){throw 'Report manifest verification failed'}
    & git -c "safe.directory=$taskRepo" -c core.longpaths=true -C $taskRepo add -- $ReportRelativeRoot $RawRelativeRoot scripts/close_vla_session.ps1 scripts/sync_vla_remote_closure.ps1 diagnostics/audit_policy_traces.py
    if($LASTEXITCODE -ne 0){throw 'Git staging failed'}
    & git -c "safe.directory=$taskRepo" -C $taskRepo diff --cached --quiet
    if($LASTEXITCODE -eq 1){
        & git -c "safe.directory=$taskRepo" -c user.name=zsure27 -c user.email=zsure27@users.noreply.github.com -C $taskRepo commit -m $message
        if($LASTEXITCODE -ne 0){throw 'Git commit failed'}
    }elseif($LASTEXITCODE -ne 0){throw 'Git index check failed'}
    & (Join-Path $PSScriptRoot 'vla_push_local.ps1')
    if($LASTEXITCODE -ne 0){throw 'GitHub verification/push failed; original data retained'}
}
$taskHost=(Invoke-TaskSsh 'hostname' | Out-String).Trim()
if($taskHost -ne $ExpectedHostname){throw 'Instance hostname mismatch; no shutdown permitted'}
Write-Output 'CLOSURE_STEP=identity_verified'
$taskSyncArgs=@{SshHost=$SshHost;SshPort=$SshPort;IdentityFile=$IdentityFile;
    ExpectedHostname=$ExpectedHostname;RemoteDirectory=$RemoteToolDirectory}
if($KnownHostsFile){$taskSyncArgs.KnownHostsFile=$KnownHostsFile}
& (Join-Path $PSScriptRoot 'sync_vla_remote_closure.ps1') @taskSyncArgs | Write-Output
if($LASTEXITCODE -ne 0){throw 'Remote closure tool synchronization failed before backup'}
Write-Output 'CLOSURE_STEP=remote_tools_verified'
$taskBackup=$FallbackBackupDirectory
$taskPreparationError=$null
if($UseVerifiedFallbackOnly){
    $taskPreparationError='Fast shutdown used the previously verified fallback archive; no new archive or report rebuild was attempted.'
    Write-Output "CLOSURE_STEP=verified_fallback_selected; DIRECTORY=$taskBackup"
}else{
try{
foreach($taskDir in $ResultDirectory){
    Invoke-TaskSsh "$taskRemotePython $taskRemoteScripts/awq_scope_analysis.py $taskDir --reference-root /root/autodl-tmp/qvla-repro/eval" | Write-Output
}
$taskBackupCommand="$taskRemotePython $taskRemoteScripts/backup_active_diagnostics.py --label $Label"
foreach($taskDir in $ResultDirectory){$taskBackupCommand+=" --result $taskDir"}
$taskBackupLines=@(Invoke-TaskSsh $taskBackupCommand)
$taskNewBackups=@($taskBackupLines | Where-Object {$_ -match '^/root/autodl-tmp/qvla-repro/backups/[A-Za-z0-9_-]+$'})
if($taskNewBackups.Count -ne 1){throw 'Expected one concrete backup directory'}
$taskBackup=$taskNewBackups[0];$taskBase=Split-Path $taskBackup -Leaf
Write-Output "CLOSURE_STEP=persistent_backup; DIRECTORY=$taskBackup"
$taskPending=Join-Path $taskRepo ("results/pending-"+$Label)
New-Item -ItemType Directory -Force $taskPending | Out-Null
& scp -r @taskSsh -P $SshPort "root@${SshHost}:$taskBackup" $taskPending
if($LASTEXITCODE -ne 0){throw 'Local backup transfer failed; no deletion or false full-backup claim'}
$taskLocalBackup=Join-Path $taskPending $taskBase
& $taskPython (Join-Path $PSScriptRoot 'verify_awq_scope_backup_local.py') --backup $taskLocalBackup --receipts (Join-Path $taskReport "backup/$taskBase") --raw-root $taskRaw --expected-videos $ExpectedVideos --result-prefix $ResultPrefix
if($LASTEXITCODE -ne 0){throw 'Local SHA256/video/extraction verification failed'}
Write-Output 'CLOSURE_STEP=local_backup_verified'
& D:/Anaconda3-5.3.1/python.exe (Join-Path $taskReport 'reproduce_scope.py')
if($LASTEXITCODE -ne 0){throw 'Measured scope report generation failed'}
& $taskPython (Join-Path $taskRepo 'diagnostics/audit_policy_traces.py') --raw-root $taskRaw --result-prefix $ResultPrefix --output (Join-Path $taskReport 'data/policy_trace_audit.json')
if($LASTEXITCODE -ne 0){throw 'Original trace audit failed'}
Invoke-TaskGitSnapshot 'Back up completed AWQ session before native shutdown'
Write-Output 'CLOSURE_STEP=github_verified_before_shutdown'
}catch{
    $taskPreparationError=$_.Exception.Message
    if(-not $taskBackup){throw 'Preparation failed without a verified fallback archive; shutdown not executed'}
    Write-Output 'CLOSURE_STEP=preparation_incomplete; ORIGINALS_RETAINED_ON_PERSISTENT_DISK'
}
}
# The remote helper independently rechecks backup hashes, idle GPU and supervisor.
try{
    # PowerShell 5 turns expected SSH disconnect stderr into ErrorRecord objects.
    $taskSavedErrorPreference=$ErrorActionPreference
    $ErrorActionPreference='Continue'
$taskSignalOutput=& ssh @taskSsh -p $SshPort "root@$SshHost" "$taskRemotePython $taskRemoteScripts/vla_shutdown_remote.py --backup-dir $taskBackup --execute" 2>&1
$taskSignalExit=$LASTEXITCODE
}finally{$ErrorActionPreference=$taskSavedErrorPreference}
$taskReceipts=@()
foreach($taskLine in $taskSignalOutput){
    if("$taskLine".StartsWith('{')){
        try{$taskReceipt="$taskLine" | ConvertFrom-Json}catch{continue}
        if($taskReceipt.execute -eq $true -and $taskReceipt.backup -eq $taskBackup){$taskReceipts+=$taskReceipt}
    }
}
if($taskReceipts.Count -ne 1){
    throw 'No native shutdown execution receipt; server shutdown is unconfirmed and may still be billing'
}
$taskClosure=[ordered]@{time_utc=[DateTime]::UtcNow.ToString('o');hostname=$taskHost;
    native_request_receipt=$taskReceipts[0];ssh_exit_code=$taskSignalExit;
    preparation_error=$taskPreparationError;requested_results=$ResultDirectory;
    platform_off_independently_verified=$false;billing_stop_independently_verified=$false}
$taskClosure | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $taskReport 'closure.json')
Write-Output 'CLOSURE_STEP=native_request_receipt_saved; PLATFORM_OFF_NOT_INDEPENDENTLY_VERIFIED'
try{Invoke-TaskGitSnapshot 'Record native shutdown receipt without claiming platform billing verification'}
catch{Write-Output 'CLOSURE_STEP=post_shutdown_push_incomplete; EXECUTION_RECEIPT_SAVED_LOCALLY'}
if($taskPreparationError){Write-Output 'CLOSURE_STEP=native_requested_with_pending_backup'}
else{Write-Output 'CLOSURE_STEP=complete'}
