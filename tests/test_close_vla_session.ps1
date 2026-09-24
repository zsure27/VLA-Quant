# No network or real shutdown: the SSH command is shadowed by a local function.
$ErrorActionPreference='Stop'
if((Get-Command ssh).CommandType -eq 'Function' -or (Get-Command scp).CommandType -eq 'Function'){throw 'Do not replace an existing SSH/SCP function'}
$taskTestRepo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskFixtureName='closure-workflow-test-'+[guid]::NewGuid().ToString('N')
$taskFixture=Join-Path $taskTestRepo "reports/sessions/$taskFixtureName"
New-Item -ItemType Directory $taskFixture | Out-Null
[System.IO.File]::WriteAllText((Join-Path $taskFixture 'manifest.py'),'raise SystemExit(2)')
$taskDummyIdentity=Join-Path $taskFixture 'test-key'
[System.IO.File]::WriteAllText($taskDummyIdentity,'test-only')
$global:taskMockHost='wrong-instance'
$global:taskMockSignals=0
$global:taskMockAnalyses=0
$global:taskMockEmitReceipt=$true
$global:taskMockFallback='/root/autodl-tmp/qvla-repro/backups/'+$taskFixtureName
function ssh{
    $taskCommand=$args[-1]
    if($taskCommand -eq 'hostname'){$global:LASTEXITCODE=0;return $global:taskMockHost}
    if($taskCommand -like 'install -d*'){$global:LASTEXITCODE=0;return}
    if($taskCommand -like '*TOOL_READY=*'){$global:LASTEXITCODE=0;return 'TOOL_READY=mock'}
    if($taskCommand -like '*awq_scope_analysis.py*'){$global:taskMockAnalyses++;$global:LASTEXITCODE=1;return}
    if($taskCommand -like '*vla_shutdown_remote.py*'){
        $global:taskMockSignals++;$global:LASTEXITCODE=1
        if($global:taskMockEmitReceipt){
            return ([ordered]@{execute=$true;backup=$global:taskMockFallback;supervisor_pid=838} | ConvertTo-Json -Compress)
        }
        return
    }
    throw 'Unexpected mocked SSH command'
}
function scp{$global:LASTEXITCODE=0}
$taskInvoke=@{ResultDirectory='/root/autodl-tmp/qvla-repro/eval/awq-scope-shard-test';
    Label=$taskFixtureName;ExpectedVideos=50;ReportRelativeRoot="reports/sessions/$taskFixtureName";
    RawRelativeRoot='results/closure-workflow-test';IdentityFile=$taskDummyIdentity;Execute=$true}
try{
    $taskRejected=$false
    try{& (Join-Path $taskTestRepo 'scripts/close_vla_session.ps1') @taskInvoke | Out-Null}catch{$taskRejected=$true}
    if(-not $taskRejected -or $global:taskMockSignals -ne 0){throw 'Wrong host must reject before shutdown'}
    $global:taskMockHost='autodl-container-8b78499521-41be183e'
    $taskRejected=$false
    try{& (Join-Path $taskTestRepo 'scripts/close_vla_session.ps1') @taskInvoke | Out-Null}catch{$taskRejected=$true}
    if(-not $taskRejected -or $global:taskMockSignals -ne 0){throw 'Missing fallback must not send shutdown'}
    $taskInvoke.FallbackBackupDirectory=$global:taskMockFallback
    & (Join-Path $taskTestRepo 'scripts/close_vla_session.ps1') @taskInvoke | Out-Null
    $taskReceiptFile=Join-Path $taskFixture 'closure.json'
    $taskReceipt=Get-Content $taskReceiptFile -Raw | ConvertFrom-Json
    if($global:taskMockSignals -ne 1 -or $taskReceipt.ssh_exit_code -ne 1 -or
       -not $taskReceipt.native_request_receipt.execute -or
       -not $taskReceipt.preparation_error -or $taskReceipt.platform_off_independently_verified){
        throw 'Valid nonzero-disconnect receipt and pending backup must be recorded truthfully'
    }
    Remove-Item -LiteralPath $taskReceiptFile
    $taskAnalysesBeforeFast=$global:taskMockAnalyses
    $taskInvoke.UseVerifiedFallbackOnly=$true
    & (Join-Path $taskTestRepo 'scripts/close_vla_session.ps1') @taskInvoke | Out-Null
    if($global:taskMockSignals -ne 2 -or $global:taskMockAnalyses -ne $taskAnalysesBeforeFast){
        throw 'Fast verified-fallback shutdown must skip analysis and still require a receipt'
    }
    Remove-Item -LiteralPath $taskReceiptFile
    $global:taskMockEmitReceipt=$false
    $taskRejected=$false
    try{& (Join-Path $taskTestRepo 'scripts/close_vla_session.ps1') @taskInvoke | Out-Null}catch{$taskRejected=$true}
    if(-not $taskRejected -or (Test-Path $taskReceiptFile)){throw 'Missing receipt must never create a success record'}
    Write-Output '5 mocked closure and fast-fallback checks passed; no real SSH or shutdown'
}finally{
    Remove-Item Function:ssh
    Remove-Item Function:scp
    $taskReceiptFile=Join-Path $taskFixture 'closure.json'
    if(Test-Path $taskReceiptFile){Remove-Item -LiteralPath $taskReceiptFile}
    Remove-Item -LiteralPath (Join-Path $taskFixture 'manifest.py')
    Remove-Item -LiteralPath $taskDummyIdentity
    # Only this newly-created, now-empty fixture directory; no recursive deletion.
    Remove-Item -LiteralPath $taskFixture
}
