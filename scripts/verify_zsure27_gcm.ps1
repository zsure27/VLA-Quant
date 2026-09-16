# Read a stored credential only in memory; output account and push permission, never the token.
$ErrorActionPreference = 'Stop'
$env:GCM_INTERACTIVE = 'never'
$request = "protocol=https`nhost=github.com`nusername=zsure27`n`n"
$lines = $request | & git credential-manager get 2>$null
if ($LASTEXITCODE -ne 0) { throw 'No noninteractive GitHub credential is available' }
$fields = @{}
foreach ($line in $lines) {
    $separator = $line.IndexOf('=')
    if ($separator -gt 0) { $fields[$line.Substring(0,$separator)] = $line.Substring($separator + 1) }
}
if (-not $fields.ContainsKey('password')) { throw 'GitHub credential has no token' }
$headers = @{ Authorization = 'Bearer ' + $fields['password']; Accept = 'application/vnd.github+json'; 'User-Agent' = 'VLA-Quant-backup' }
try {
    $account = Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers -Method Get
    $repository = Invoke-RestMethod -Uri 'https://api.github.com/repos/zsure27/VLA-Quant' -Headers $headers -Method Get
} catch { throw 'GitHub account or repository verification failed' }
if ($account.login -cne 'zsure27' -or $repository.full_name -cne 'zsure27/VLA-Quant' -or $repository.permissions.push -ne $true) {
    throw "Authenticated account $($account.login) cannot push to zsure27/VLA-Quant"
}
Write-Output 'VERIFIED_GITHUB_LOGIN=zsure27; PUSH_PERMISSION=true'
$fields.Clear()
$headers.Clear()
