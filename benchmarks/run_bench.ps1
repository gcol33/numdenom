$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot          # benchmarks/ -> repo root
$rs = (Get-ChildItem 'C:\Program Files\R\R-*\bin\Rscript.exe' |
       Sort-Object FullName | Select-Object -Last 1).FullName
Set-Location $repo

$waitLog = 'benchmarks\bench_wait.log'
function Log($m) { "$([DateTime]::Now.ToString('HH:mm:ss'))  $m" | Add-Content $waitLog }
Set-Content $waitLog "run_bench launcher started"

# --- wait until the machine is quiet (unrelated SDM/validation jobs done) ---
$deadline  = (Get-Date).AddHours(12)
$quietHits = 0
while ((Get-Date) -lt $deadline) {
  $sdm = @(Get-CimInstance Win32_Process -Filter "Name='Rscript.exe'" |
           Where-Object { $_.CommandLine -like '*project_all.R*' }).Count
  $cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 3
         ).CounterSamples | Measure-Object CookedValue -Average | Select-Object -Expand Average
  Log ("sdm_workers={0}  cpu_avg={1:N0}%  quietHits={2}" -f $sdm, $cpu, $quietHits)
  if ($sdm -eq 0 -and $cpu -lt 40) { $quietHits++ } else { $quietHits = 0 }
  if ($quietHits -ge 3) { break }              # ~3 consecutive quiet samples
  Start-Sleep -Seconds 30
}
Log "machine quiet -> running benchmark"

Remove-Item 'benchmarks\.bench_done' -ErrorAction SilentlyContinue
& $rs 'benchmarks\bench_stan_binomial.R' `
    > 'benchmarks\bench_stdout.log' 2> 'benchmarks\bench_stderr.log'
Log "benchmark process exited"
