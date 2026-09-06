# Entrenamiento formal: 4 variantes x 5 semillas = 20 runs
# Rutas absolutas para evitar problemas de contexto al lanzar con -File

$root   = "C:\Users\UNAL\Documents\Papers\IP Robust RL Montecarlo\P2_IG_RRL"
$python = "C:\Users\UNAL\Documents\Papers\IP Robust RL Montecarlo\.venv\Scripts\python.exe"
$script = Join-Path $root "scripts\train_residual.py"
$logdir = Join-Path $root "results\logs\formal"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

$env:PYTHONPATH        = ".;src"
$env:PYTHONUTF8        = "1"
$env:PYTHONIOENCODING  = "utf-8"

$variants = @("a1","a2","a3","p2")
$seeds    = @(1000, 2000, 3000, 4000, 5000)

foreach ($seed in $seeds) {
    Write-Output "=== Semilla $seed ==="
    $procs = @()
    foreach ($variant in $variants) {
        $tag    = "${variant}_seed${seed}"
        $stdout = Join-Path $logdir "${tag}.stdout.log"
        $stderr = Join-Path $logdir "${tag}.stderr.log"
        "" | Set-Content $stdout; "" | Set-Content $stderr
        $p = Start-Process -FilePath $python `
            -ArgumentList $script,"--variant",$variant,"--seed",$seed,"--timesteps","300000" `
            -WorkingDirectory $root `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError  $stderr `
            -NoNewWindow -PassThru
        Write-Output "  Lanzado $tag  PID=$($p.Id)"
        $procs += $p
    }
    Write-Output "  Esperando $($procs.Count) procesos..."
    foreach ($p in $procs) { $p.WaitForExit() }
    Write-Output "  Semilla $seed completada."
}
Write-Output "=== ENTRENAMIENTO FORMAL COMPLETO ==="
