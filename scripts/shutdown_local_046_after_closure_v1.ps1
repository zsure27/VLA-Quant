param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$caseRoot = 'C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/results/experiments/p2-shared-peft/20261015-046-quant-finetuning'
$authorization = Get-Content -Raw -Encoding UTF8 -LiteralPath "$caseRoot/windows-shutdown-authorization-v1.json" | ConvertFrom-Json
$admission = Get-Content -Raw -Encoding UTF8 -LiteralPath "$caseRoot/windows-shutdown-admission.json" | ConvertFrom-Json
$backup = Get-Content -Raw -Encoding UTF8 -LiteralPath "$caseRoot/final-archive-set-local-verification.json" | ConvertFrom-Json
$native = Get-Content -Raw -Encoding UTF8 -LiteralPath "$caseRoot/native-shutdown-execute.json" | ConvertFrom-Json
if ($authorization.run_id -ne '20261015-046-quant-finetuning-v1' -or -not $authorization.windows_shutdown_authorized -or -not $authorization.current_run_only -or $admission.run_id -ne $authorization.run_id) {
    throw 'Current046 user authorization absent.'
}
if ($backup.gate -ne 'PASS_LOCAL_COMPLETE_ARCHIVE_SET_SHA_COVERAGE' -or -not $backup.all_set_files_sha_verified -or -not $backup.all_session_files_covered -or -not $backup.all_stage_archives_locally_restored_and_verified -or $backup.original_videos_verified -ne 920 -or -not $backup.root_git_bundle_verified -or -not $backup.research_git_bundle_verified) {
    throw 'Complete current046 local archive evidence absent.'
}
if (-not $native.execute -or $native.backup -ne $backup.server_directory -or -not $admission.experiments_terminated -or -not $admission.gpu_idle_verified -or -not $admission.server_archive_verified -or -not $admission.analysis_reports_synced -or -not $admission.github_remote_sha_verified -or -not $admission.heartbeat_paused -or -not $admission.summary_delivered) {
    throw 'Required current046 closure facts absent; keep Windows running.'
}
if ($ValidateOnly) { Write-Output 'PASS_CURRENT046_WINDOWS_SHUTDOWN_ADMISSION'; exit 0 }
# One-shot grace lets the complete final response finish sending; no scheduled task.
Start-Sleep -Seconds 15
$request = [ordered]@{ time_utc = [DateTime]::UtcNow.ToString('o'); run_id = $authorization.run_id; user_requested = $true; command = 'shutdown.exe /s /t 0'; force_close = $false; state = 'NATIVE_REQUEST_ABOUT_TO_EXECUTE'; independent_power_off_verified = $false }
$request | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath "$caseRoot/windows-shutdown-request.json"
& 'C:/Windows/System32/shutdown.exe' /s /t 0
$shutdownExit = $LASTEXITCODE
$request.state = if ($shutdownExit -eq 0) { 'NATIVE_REQUEST_ACCEPTED' } else { 'NATIVE_REQUEST_FAILED' }
$request.exit_code = $shutdownExit
$request | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath "$caseRoot/windows-shutdown-request.json"
exit $shutdownExit
