param(
    [string]$SshHost='connect.nmb1.seetacloud.com',
    [ValidateRange(1,65535)][int]$SshPort=19111,
    [string]$IdentityFile='C:/Users/zsure/.ssh/id_ed25519_vla_014',
    [string]$KnownHostsFile,
    [Parameter(Mandatory=$true)][string]$ExpectedHostname,
    [string]$RemoteDirectory='/root/autodl-tmp/qvla-repro/closure-tools',
    [switch]$Plan
)

# Install the two tools needed for a bounded backup and native shutdown while
# quota is still plentiful. Uploads use temporary names and become active only
# after remote SHA256 and Python compilation checks pass.
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if($SshHost -notmatch '^[A-Za-z0-9.-]+$' -or
   $ExpectedHostname -notmatch '^[A-Za-z0-9-]+$' -or
   $RemoteDirectory -notmatch '^/root/autodl-tmp/qvla-repro/[A-Za-z0-9_./-]+$'){
    throw 'Invalid remote closure target'
}
$identity=$IdentityFile
$localTools=@(
    (Join-Path $PSScriptRoot 'backup_active_diagnostics.py'),
    (Join-Path $PSScriptRoot 'vla_shutdown_remote.py')
)
foreach($tool in $localTools){if(-not (Test-Path -LiteralPath $tool -PathType Leaf)){throw "Missing local closure tool: $tool"}}
$hashes=@{}
foreach($tool in $localTools){$hashes[[IO.Path]::GetFileName($tool)]=(Get-FileHash -Algorithm SHA256 -LiteralPath $tool).Hash.ToLowerInvariant()}
if($Plan){
    [pscustomobject]@{host=$SshHost;port=$SshPort;expected_hostname=$ExpectedHostname;
        remote_directory=$RemoteDirectory;tools=$hashes;network_actions=$false} | ConvertTo-Json -Depth 4
    return
}
$identity=(Resolve-Path -LiteralPath $IdentityFile).Path
$sshArgs=@('-i',$identity,'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=12')
$scpArgs=@('-i',$identity,'-o','StrictHostKeyChecking=yes','-o','BatchMode=yes')
if($KnownHostsFile){
    $known=(Resolve-Path -LiteralPath $KnownHostsFile).Path
    $sshArgs+=@('-o',"UserKnownHostsFile=$known")
    $scpArgs+=@('-o',"UserKnownHostsFile=$known")
}
function Invoke-ClosureSsh([string]$command){
    $output=& ssh @sshArgs -p $SshPort "root@$SshHost" $command
    if($LASTEXITCODE -ne 0){throw "Remote closure preflight failed: $command"}
    return @($output)
}
$actual=(@(Invoke-ClosureSsh 'hostname') -join '').Trim()
if($actual -ne $ExpectedHostname){throw "Hostname mismatch: expected $ExpectedHostname, received $actual"}
$nonce=[guid]::NewGuid().ToString('N')
Invoke-ClosureSsh "install -d -m 700 $RemoteDirectory" | Out-Null
foreach($tool in $localTools){
    $name=[IO.Path]::GetFileName($tool)
    $temporary="$RemoteDirectory/.$name.$nonce.tmp"
    & scp @scpArgs -P $SshPort $tool "root@${SshHost}:$temporary"
    if($LASTEXITCODE -ne 0){throw "Could not upload remote closure tool: $name"}
    $expected=$hashes[$name]
    $activate="set -eu; actual=`$(sha256sum $temporary | cut -d' ' -f1); test `"`$actual`" = $expected; /root/miniconda3/envs/qvla-oft/bin/python -m py_compile $temporary; chmod 700 $temporary; mv -f $temporary $RemoteDirectory/$name; test `"`$(sha256sum $RemoteDirectory/$name | cut -d' ' -f1)`" = $expected; echo TOOL_READY=${name}:$expected"
    Invoke-ClosureSsh $activate | Write-Output
}
[pscustomobject]@{status='REMOTE_CLOSURE_TOOLS_READY';hostname=$actual;remote_directory=$RemoteDirectory;
    backup_tool="$RemoteDirectory/backup_active_diagnostics.py";
    shutdown_tool="$RemoteDirectory/vla_shutdown_remote.py";hashes=$hashes} | ConvertTo-Json -Depth 4
