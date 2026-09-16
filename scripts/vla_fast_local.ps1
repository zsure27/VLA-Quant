param(
    [Parameter(Mandatory=$true)][ValidateSet('check','smoke','baseline','w4','backup')][string]$Action,
    [string]$SshHost = 'connect.nmb1.seetacloud.com',
    [int]$SshPort = 19111,
    [string]$SshUser = 'root',
    [string]$IdentityFile = 'C:\Users\zsure\.ssh\id_ed25519_vla_014'
)

$ErrorActionPreference = 'Stop'
$instanceLabel = if ($SshPort -eq 19111) { '014 (3fbf46b812-2fdd6883)' } elseif ($SshPort -eq 16917) { '031 (full instance ID pending verification)' } else { "SSH port $SshPort" }
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$remoteScript = '/root/autodl-tmp/qvla-repro/scripts/vla_fast_remote.sh'
$remote = '{0}@{1}' -f $SshUser, $SshHost
$sshArgs = @('-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p',"$SshPort")
if ($IdentityFile) {
    $sshArgs += @('-i',(Resolve-Path -LiteralPath $IdentityFile).Path)
}

if ((& git -C $repo remote get-url origin) -ne 'https://github.com/zsure27/VLA-Quant.git') {
    throw 'Git remote is not zsure27/VLA-Quant.'
}

Write-Host "AutoDL $instanceLabel : $Action"
$scpArgs = @('-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-P',"$SshPort")
if ($IdentityFile) { $scpArgs += @('-i',(Resolve-Path -LiteralPath $IdentityFile).Path) }
& scp @scpArgs (Join-Path $PSScriptRoot 'vla_fast_remote.sh') "${remote}:$remoteScript"
if ($LASTEXITCODE -ne 0) { throw 'Could not upload the checked remote script' }
$remoteOutput = & ssh @sshArgs $remote "bash $remoteScript $Action"
$remoteExit = $LASTEXITCODE
$remoteOutput | ForEach-Object { Write-Host $_ }

if ($Action -eq 'backup') {
    # Copy a second recoverable copy even when the GitHub push lacks credentials.
    $backupLine = $remoteOutput | Where-Object { $_ -like 'Backup directory: *' } | Select-Object -First 1
    if (-not $backupLine) { throw 'Remote backup directory was not reported' }
    $remoteBackup = $backupLine.Substring('Backup directory: '.Length)
    if ($remoteBackup -notmatch '^/root/autodl-tmp/qvla-repro/backups/[0-9]{8}-[a-f0-9]+$') { throw 'Unexpected remote backup path' }
    $backup = Join-Path $repo ('results\pending-' + [IO.Path]::GetFileName($remoteBackup))
    New-Item -ItemType Directory -Path $backup -Force | Out-Null
    $shortCommit = ([IO.Path]::GetFileName($remoteBackup) -split '-',2)[1]
    foreach ($file in @("VLA-Quant-$shortCommit.bundle","$shortCommit.patch",'SHA256SUMS.txt','RESULTS_SHA256SUMS.txt','LARGE_FILES_NOT_IN_GIT.txt','pending-remote.txt','git-status.txt','worktree.patch','runtime-lock.txt')) {
        & scp @scpArgs "${remote}:$remoteBackup/$file" (Join-Path $backup $file)
        if ($LASTEXITCODE -ne 0) { throw "Could not copy backup file $file" }
    }
    & scp @scpArgs -r "${remote}:$remoteBackup/results" $backup
    if ($LASTEXITCODE -ne 0) { throw 'Could not copy evaluation files and videos' }
    Push-Location $backup
    try {
        foreach ($manifest in @('SHA256SUMS.txt','RESULTS_SHA256SUMS.txt')) {
            foreach ($line in (Get-Content -LiteralPath $manifest)) {
                $fields = $line -split '\s+', 2
                $file = $fields[1].TrimStart('*').Replace('/',[IO.Path]::DirectorySeparatorChar)
                $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant()
                if ($actual -ne $fields[0]) { throw "Backup hash verification failed: $file" }
            }
        }
    } finally { Pop-Location }
    Write-Host "Persistent and local backup verified: $backup"
    if ($remoteExit -eq 2) {
        Write-Warning 'GitHub push pending. Keep the server persistent disk and this local copy.'
    } elseif ($remoteExit -ne 0) {
        throw "Remote backup failed with exit code $remoteExit"
    }
} elseif ($remoteExit -ne 0) {
    throw "Remote action failed with exit code $remoteExit"
}
