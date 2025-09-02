param([string]$input)
$start = Get-Date
python -m app.transcribe $input --out .\out --hf-token-file .env
$end = Get-Date
Write-Host ("Elapsed: " + ($end - $start).TotalSeconds + "s")
