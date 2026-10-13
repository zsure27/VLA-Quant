$ErrorActionPreference = 'Stop'
$caseRoot = 'C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/results/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'
$admission = Get-Content -Raw -Encoding UTF8 -LiteralPath "$caseRoot/windows-shutdown-admission.json" | ConvertFrom-Json
$backup = Get-Content -Raw -Encoding UTF8 -LiteralPath "$caseRoot/final-archive-local-verification.json" | ConvertFrom-Json
$native = Get-Content -Raw -Encoding UTF8 -LiteralPath "$caseRoot/native-shutdown-execute.json" | ConvertFrom-Json
$supplement = Get-Content -Raw -Encoding UTF8 -LiteralPath "$caseRoot/final-supplement-verification.json" | ConvertFrom-Json
if ($backup.gate -ne 'PASS_LOCAL_FULL_SESSION_SHA_DATA_AND_VIDEO_COVERAGE' -or -not $native.execute -or $supplement.gate -ne 'PASS_FINAL_SUPPLEMENT_SHA_AND_GIT_BUNDLE_RECOVERY' -or -not $admission.github_remote_sha_verified -or -not $admission.heartbeat_paused -or -not $admission.summary_delivered) {
    throw 'Required complete closure evidence absent; keep Windows running.'
}
# A short one-shot grace period lets the final chat message finish sending.
# /t 0 avoids the implicit force-close behavior of positive shutdown timeouts.
Start-Sleep -Seconds 15
$request = [ordered]@{ time_utc = [DateTime]::UtcNow.ToString('o'); user_requested = $true; command = 'shutdown.exe /s /t 0'; force_close = $false; state = 'NATIVE_REQUEST_ABOUT_TO_EXECUTE'; independent_power_off_verified = $false }
$request | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath "$caseRoot/windows-shutdown-request.json"
& 'C:/Windows/System32/shutdown.exe' /s /t 0
$shutdownExit = $LASTEXITCODE
$request.state = if ($shutdownExit -eq 0) { 'NATIVE_REQUEST_ACCEPTED' } else { 'NATIVE_REQUEST_FAILED' }
$request.exit_code = $shutdownExit
$request | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath "$caseRoot/windows-shutdown-request.json"
exit $shutdownExit
