$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ((& git -C $repo remote get-url origin) -ne 'https://github.com/zsure27/VLA-Quant.git') {
    throw 'Unexpected GitHub repository'
}
& (Join-Path $PSScriptRoot 'verify_zsure27_gcm.ps1')
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'never'
$authArgs = @('-c','credential.username=zsure27','-c','credential.interactive=false')
& git -C $repo @authArgs fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'GitHub fetch failed' }
& git -C $repo merge-base --is-ancestor origin/main HEAD
if ($LASTEXITCODE -ne 0) { throw 'Remote main is not an ancestor; resolve updates before pushing' }
& git -C $repo @authArgs push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'GitHub push failed; retain all backups' }
& git -C $repo @authArgs fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'Post-push fetch failed' }
$localCommit = & git -C $repo rev-parse HEAD
$remoteCommit = & git -C $repo rev-parse origin/main
if ($localCommit -ne $remoteCommit) { throw 'Remote commit verification failed' }
Write-Output "VERIFIED_REMOTE_MAIN=$remoteCommit"
