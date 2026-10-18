#requires -Version 7.2
<#
.SYNOPSIS
Checks an exact local backup deletion plan; deletes files only with -Execute.
.DESCRIPTION
No network calls, downloads, recursive removal, or directory removal. The plan
must bind the cloud receipt and optional member catalog to a verified private
Git catalog publication. Every selected file is checked before any deletion,
then checked again immediately before its individual Remove-Item operation.
#>
[CmdletBinding()]
param(
    [string]$PlanPath,
    [string]$CloudReceiptPath,
    [string]$PublishedCatalogReceiptPath,
    [string]$CoverageCatalogPath,
    [string]$ReceiptPath,
    [switch]$Execute
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Condition([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Get-ExactFullPath([string]$Path) {
    Assert-Condition (-not [string]::IsNullOrWhiteSpace($Path)) 'An absolute file path is required.'
    Assert-Condition ([IO.Path]::IsPathFullyQualified($Path)) "Path must be absolute: $Path"
    Assert-Condition (-not $Path.StartsWith('\\')) 'UNC and device paths are not allowed.'
    return [IO.Path]::GetFullPath($Path).TrimEnd([char[]]@('\', '/'))
}

function Get-AllowedBackupRoots {
    return @(
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/.codex-work/vla-quant-sync-20260915/.local/archives'),
        (Get-ExactFullPath 'D:/VLA-Backups/20261016-091-residual-resume-v2'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/backups'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/.codex-work/091-archive-range-download-v2'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/.codex-work/091-final-closing-documents-20261013-v1'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/.codex-work/091-final-closure-supplement-20261013-v1')
    )
}

function Get-AllowedBackupFiles {
    # These twelve exact tmp archive paths were audited in the 2026-10-18
    # upload-private-extra-backups plan. The surrounding tmp directory is not
    # an allowed root, and this list cannot be expanded by runtime arguments.
    return @(
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/sq-local-small.tar.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/fixed-rollout-small.tar.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/stage50-small.tar.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/p25-sync-20260927.tar'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/sq-branch-small.tar.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/fp32-pairs-small.tar.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/stage-fixed-small.tar.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/low-rank-small.tar.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/input-diag-small.tar.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/vla-p2-93cde52.bundle'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/vla-p2-3aa8f67.bundle.gz'),
        (Get-ExactFullPath 'C:/Users/zsure/Documents/Triton/tmp/vla-p2-3aa8f67.bundle')
    )
}

function Test-UnderRoot([string]$Path, [string]$Root) {
    return $Path.StartsWith($Root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-NoReparseAncestors([string]$Path) {
    $cursor = Get-ExactFullPath $Path
    while (-not [string]::IsNullOrEmpty($cursor)) {
        $attributes = [IO.File]::GetAttributes($cursor)
        Assert-Condition (($attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "Reparse path is forbidden: $cursor"
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
}

function Assert-EligibleFilePath([string]$Path, [string[]]$Roots) {
    $full = Get-ExactFullPath $Path
    $matching = @($Roots | Where-Object { Test-UnderRoot $full $_ })
    $matchingExact = @()
    if ($matching.Count -eq 0) {
        $matchingExact = @(Get-AllowedBackupFiles | Where-Object { [string]::Equals($full, $_, [StringComparison]::OrdinalIgnoreCase) })
    }
    Assert-Condition ($matching.Count -eq 1 -or $matchingExact.Count -eq 1) "Deletion target is outside the exact allowed roots and files: $full"
    Assert-NoReparseAncestors $full
    $item = Get-Item -LiteralPath $full -Force -ErrorAction Stop
    Assert-Condition (-not $item.PSIsContainer) "Directory deletion is forbidden: $full"
    Assert-Condition ($item -is [IO.FileInfo]) "Target is not a filesystem file: $full"
    $resolved = (Resolve-Path -LiteralPath $full -ErrorAction Stop).ProviderPath
    Assert-Condition ([string]::Equals((Get-ExactFullPath $resolved), $full, [StringComparison]::OrdinalIgnoreCase)) "Resolved target differs from the checked path: $full"
    return $full
}

function Get-FileSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
}

function Read-JsonObject([string]$Path) {
    $full = Get-ExactFullPath $Path
    Assert-NoReparseAncestors $full
    $value = Get-Content -LiteralPath $full -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable -Depth 100
    Assert-Condition ($value -is [Collections.IDictionary]) "JSON object required: $full"
    return $value
}

function Assert-Hash($Value, [string]$Name) {
    Assert-Condition (($Value -is [string]) -and $Value -cmatch '^[0-9a-f]{64}$') "Invalid SHA256: $Name"
}

function Assert-Length($Value, [string]$Name, [bool]$AllowZero = $false) {
    $integer = ($Value -is [long]) -or ($Value -is [int]) -or ($Value -is [System.Numerics.BigInteger])
    Assert-Condition ($integer -and ($Value -ge 0) -and ($AllowZero -or $Value -gt 0) -and $Value -le [long]::MaxValue) "Invalid byte length or ID: $Name"
}

function Assert-CloudArchive($Archive, [string]$Sha256) {
    Assert-Hash $Sha256 'cloud archive key'
    Assert-Condition ($Archive -is [Collections.IDictionary]) "Missing cloud archive: $Sha256"
    Assert-Condition ($Archive.sha256 -ceq $Sha256) 'Cloud archive key/SHA mismatch.'
    Assert-Length $Archive.bytes 'cloud archive bytes'
    Assert-Condition ($Archive.remote_download_sha256_verified -is [bool] -and $Archive.remote_download_sha256_verified) 'Cloud archive was not fully download-verified.'
    $parts = @($Archive.parts)
    Assert-Condition ($parts.Count -gt 0) 'Cloud archive has no parts.'
    [long]$offset = 0
    $ids = [Collections.Generic.HashSet[long]]::new()
    for ($i = 0; $i -lt $parts.Count; $i++) {
        $part = $parts[$i]
        Assert-Condition ($part -is [Collections.IDictionary]) 'Invalid cloud part.'
        Assert-Length $part.asset_id 'asset ID'
        Assert-Condition ($ids.Add([long]$part.asset_id)) 'Repeated asset ID in archive.'
        Assert-Length $part.bytes 'part bytes'
        Assert-Length $part.offset 'part offset' $true
        Assert-Condition ($part.offset -eq $offset) 'Cloud part offsets are not contiguous.'
        Assert-Hash $part.sha256 'part SHA'
        $name = 'sha256-' + $part.sha256 + ('.part{0:D3}-of{1:D3}' -f ($i + 1), $parts.Count)
        Assert-Condition ($part.name -ceq $name) 'Cloud part name or order mismatch.'
        Assert-Condition ($part.download_verified -is [bool] -and $part.download_verified) 'Cloud part was not download-verified.'
        $offset = [long]($offset + $part.bytes)
    }
    Assert-Condition ($offset -eq $Archive.bytes) 'Cloud part total differs from archive length.'
}

function Assert-PublishedFile($Publication, [string]$RepositoryPath, [string]$Sha256) {
    Assert-Condition (-not [string]::IsNullOrWhiteSpace($RepositoryPath)) 'Published catalog file path is required.'
    Assert-Hash $Sha256 'published file SHA'
    $matches = @($Publication.files | Where-Object { $_.path -ceq $RepositoryPath -and $_.sha256 -ceq $Sha256 })
    Assert-Condition ($matches.Count -eq 1) "Exact file SHA is absent from the published private catalog: $RepositoryPath"
}

function Get-MemberKey([string]$ArchiveSha, [string]$Container, [string]$Member) {
    Assert-Condition (-not $Container.Contains([char]0) -and -not $Member.Contains([char]0)) 'NUL in archive member proof.'
    return $ArchiveSha + [char]0 + $Container + [char]0 + $Member
}

function New-MemberIndex($Catalog) {
    Assert-Condition ($Catalog.Contains('members')) 'Coverage catalog has no members array.'
    $index = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
    foreach ($member in $Catalog.members) {
        Assert-Hash $member.sha256 'catalog member SHA'
        Assert-Hash $member.archive_sha256 'catalog archive SHA'
        Assert-Length $member.bytes 'catalog member bytes' $true
        Assert-Length $member.depth 'catalog member depth' $true
        $key = Get-MemberKey $member.archive_sha256 $member.container $member.member
        if ($index.ContainsKey($key)) {
            $old = $index[$key]
            Assert-Condition ($old.sha256 -ceq $member.sha256 -and $old.bytes -eq $member.bytes -and $old.depth -eq $member.depth) 'Ambiguous repeated archive member in coverage catalog.'
        }
        $index[$key] = $member
    }
    return ,$index
}

function Assert-RecoveryProof($Entry, $Archive, $MemberIndex) {
    Assert-Condition ($Entry.recovery -is [Collections.IDictionary]) 'Missing recovery proof.'
    if ($Entry.recovery.kind -ceq 'archive') {
        Assert-Condition ($Entry.sha256 -ceq $Entry.cloud_archive_sha256 -and $Entry.bytes -eq $Archive.bytes) 'Raw archive does not equal the remote verified archive.'
        return
    }
    Assert-Condition ($Entry.recovery.kind -ceq 'member') 'Recovery kind must be archive or member.'
    Assert-Condition ($null -ne $MemberIndex) 'Extracted file requires a published content coverage catalog.'
    $chain = @($Entry.recovery.member_chain)
    Assert-Condition ($chain.Count -gt 0) 'Extracted file has no member recovery chain.'
    $previous = $null
    for ($i = 0; $i -lt $chain.Count; $i++) {
        $proof = $chain[$i]
        Assert-Condition ($proof -is [Collections.IDictionary]) 'Invalid member chain step.'
        Assert-Hash $proof.sha256 'member chain SHA'
        Assert-Length $proof.bytes 'member chain bytes' $true
        Assert-Length $proof.depth 'member chain depth' $true
        Assert-Condition ($proof.depth -eq $i) 'Member chain must begin at depth zero and be contiguous.'
        $key = Get-MemberKey $Entry.cloud_archive_sha256 $proof.container $proof.member
        Assert-Condition ($MemberIndex.ContainsKey($key)) 'Member recovery chain is absent from the SHA-bound audit catalog.'
        $audited = $MemberIndex[$key]
        Assert-Condition ($audited.sha256 -ceq $proof.sha256 -and $audited.bytes -eq $proof.bytes -and $audited.depth -eq $proof.depth) 'Member recovery proof differs from audited content.'
        if ($null -ne $previous) {
            Assert-Condition ($proof.container -ceq ($previous.container + '!' + $previous.member)) 'Nested member container chain is broken.'
        }
        $previous = $proof
    }
    Assert-Condition ($previous.sha256 -ceq $Entry.sha256 -and $previous.bytes -eq $Entry.bytes) 'Recovered member content does not equal selected local file.'
}

function Assert-LocalFileContent($Entry, [string[]]$Roots) {
    $full = Assert-EligibleFilePath $Entry.path $Roots
    $item = Get-Item -LiteralPath $full -Force
    Assert-Condition ($item.Length -eq $Entry.bytes) "Local file length changed: $full"
    Assert-Condition ((Get-FileSha256 $full) -ceq $Entry.sha256) "Local file SHA changed: $full"
    return $full
}

function Get-DiskSnapshot([string[]]$Roots) {
    $snapshot = [ordered]@{}
    foreach ($root in $Roots) {
        $drive = [IO.Path]::GetPathRoot($root)
        if (-not $snapshot.Contains($drive)) {
            $info = [IO.DriveInfo]::new($drive)
            $snapshot[$drive] = @{ available_bytes = $info.AvailableFreeSpace; total_bytes = $info.TotalSize }
        }
    }
    return $snapshot
}

function Write-DurableEvent($Writer, $Stream, $Event) {
    $Event.utc = [DateTime]::UtcNow.ToString('o')
    $Writer.WriteLine(($Event | ConvertTo-Json -Compress -Depth 30))
    $Writer.Flush()
    $Stream.Flush($true)
}

function Invoke-VerifiedBackupRemoval {
    param([string]$PlanPath, [string]$CloudReceiptPath, [string]$PublishedCatalogReceiptPath,
          [string]$CoverageCatalogPath, [string]$ReceiptPath, [switch]$Execute)
    Assert-Condition ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) 'This exact-path removal tool is for Windows only.'
    $roots = @(Get-AllowedBackupRoots)
    $exactFiles = @(Get-AllowedBackupFiles)
    $plan = Read-JsonObject $PlanPath
    $cloud = Read-JsonObject $CloudReceiptPath
    $publication = Read-JsonObject $PublishedCatalogReceiptPath
    Assert-Condition ($plan.schema_version -eq 1 -and $plan.repository -ceq 'zsure27/VLA-Quant-Archives') 'Unexpected deletion plan schema or repository.'
    Assert-Condition ($cloud.repository -ceq 'zsure27/VLA-Quant-Archives' -and $cloud.files -is [Collections.IDictionary]) 'Unexpected cloud receipt repository or files.'
    Assert-Condition ($publication.repository -ceq 'zsure27/VLA-Quant-Archives' -and $publication.private -is [bool] -and $publication.private) 'Catalog repository is not private.'
    Assert-Condition ($publication.remote_head_verified -is [bool] -and $publication.remote_head_verified) 'Private catalog publication lacks remote HEAD verification.'
    Assert-Condition ($publication.committed_paths_verified -is [bool] -and $publication.committed_paths_verified) 'Private catalog publication lacks committed path/blob verification.'
    Assert-Condition ($plan.private_catalog_commit -cmatch '^[0-9a-f]{40}$' -and $publication.commit -ceq $plan.private_catalog_commit) 'Published private catalog commit mismatch.'
    Assert-Hash $plan.cloud_receipt_sha256 'plan cloud receipt SHA'
    Assert-Condition ((Get-FileSha256 $CloudReceiptPath) -ceq $plan.cloud_receipt_sha256) 'Cloud receipt bytes differ from the plan.'
    Assert-PublishedFile $publication $plan.cloud_receipt_repository_path $plan.cloud_receipt_sha256
    $memberIndex = $null
    if (-not [string]::IsNullOrWhiteSpace($CoverageCatalogPath)) {
        Assert-Hash $plan.coverage_catalog_sha256 'coverage catalog SHA'
        Assert-Condition ((Get-FileSha256 $CoverageCatalogPath) -ceq $plan.coverage_catalog_sha256) 'Coverage catalog bytes differ from the plan.'
        Assert-PublishedFile $publication $plan.coverage_catalog_repository_path $plan.coverage_catalog_sha256
        $memberIndex = New-MemberIndex (Read-JsonObject $CoverageCatalogPath)
    }
    Assert-Condition ($plan.files -is [array] -and $plan.files.Count -gt 0) 'Deletion plan has no explicit file array.'
    $paths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $validatedArchives = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    [long]$plannedBytes = 0
    foreach ($entry in $plan.files) {
        Assert-Condition ($entry -is [Collections.IDictionary]) 'Invalid planned file.'
        Assert-Hash $entry.sha256 'local file SHA'
        Assert-Hash $entry.cloud_archive_sha256 'recovery archive SHA'
        Assert-Length $entry.bytes 'local file bytes' $true
        Assert-Condition ($cloud.files.Contains($entry.cloud_archive_sha256)) 'Selected recovery archive is absent from cloud receipt.'
        $archive = $cloud.files[$entry.cloud_archive_sha256]
        if ($validatedArchives.Add($entry.cloud_archive_sha256)) { Assert-CloudArchive $archive $entry.cloud_archive_sha256 }
        Assert-RecoveryProof $entry $archive $memberIndex
        $full = Assert-LocalFileContent $entry $roots
        Assert-Condition ($paths.Add($full)) "Repeated local target in deletion plan: $full"
        $plannedBytes = [long]($plannedBytes + $entry.bytes)
    }
    # No file is removed if any preflight check above fails.
    $receipt = Get-ExactFullPath $ReceiptPath
    Assert-NoReparseAncestors ([IO.Path]::GetDirectoryName($receipt))
    Assert-Condition (-not (Test-Path -LiteralPath $receipt)) 'Receipt already exists; choose a new path.'
    foreach ($root in $roots) { Assert-Condition (-not (Test-UnderRoot $receipt $root)) 'Receipt must be outside the deletion roots.' }
    $before = Get-DiskSnapshot $roots
    $stream = [IO.FileStream]::new($receipt, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false), 4096, $true)
    [long]$deletedBytes = 0
    [int]$deletedCount = 0
    try {
        Write-DurableEvent $writer $stream ([ordered]@{ event = 'start'; execute = [bool]$Execute; plan_sha256 = Get-FileSha256 $PlanPath; cloud_receipt_sha256 = $plan.cloud_receipt_sha256; private_catalog_commit = $plan.private_catalog_commit; published_catalog_receipt_sha256 = Get-FileSha256 $PublishedCatalogReceiptPath; allowed_roots = $roots; allowed_exact_files = $exactFiles; files_planned = $plan.files.Count; logical_bytes_planned = $plannedBytes; disk_before = $before })
        foreach ($entry in $plan.files) {
            $full = Assert-LocalFileContent $entry $roots
            if ($Execute) {
                Write-DurableEvent $writer $stream ([ordered]@{ event = 'delete_intent'; path = $full; bytes = $entry.bytes; sha256 = $entry.sha256; cloud_archive_sha256 = $entry.cloud_archive_sha256 })
                Remove-Item -LiteralPath $full -Force -ErrorAction Stop
                Assert-Condition (-not (Test-Path -LiteralPath $full)) "Remove-Item did not remove selected file: $full"
                $deletedCount++
                $deletedBytes = [long]($deletedBytes + $entry.bytes)
                Write-DurableEvent $writer $stream ([ordered]@{ event = 'deleted'; path = $full; bytes = $entry.bytes; sha256 = $entry.sha256; cloud_archive_sha256 = $entry.cloud_archive_sha256 })
            } else {
                Write-DurableEvent $writer $stream ([ordered]@{ event = 'dry_run_checked'; path = $full; bytes = $entry.bytes; sha256 = $entry.sha256; cloud_archive_sha256 = $entry.cloud_archive_sha256 })
            }
        }
        $after = Get-DiskSnapshot $roots
        $summary = [ordered]@{ event = 'complete'; execute = [bool]$Execute; files_checked = $plan.files.Count; files_deleted = $deletedCount; logical_bytes_deleted = $deletedBytes; disk_before = $before; disk_after = $after; receipt = $receipt }
        Write-DurableEvent $writer $stream $summary
        $summary | ConvertTo-Json -Compress -Depth 10
    } catch {
        Write-DurableEvent $writer $stream ([ordered]@{ event = 'failed'; execute = [bool]$Execute; files_deleted = $deletedCount; logical_bytes_deleted = $deletedBytes; error = $_.Exception.Message; disk_after = Get-DiskSnapshot $roots })
        throw
    } finally {
        $writer.Dispose()
        $stream.Dispose()
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Invoke-VerifiedBackupRemoval -PlanPath $PlanPath -CloudReceiptPath $CloudReceiptPath -PublishedCatalogReceiptPath $PublishedCatalogReceiptPath -CoverageCatalogPath $CoverageCatalogPath -ReceiptPath $ReceiptPath -Execute:$Execute
}
