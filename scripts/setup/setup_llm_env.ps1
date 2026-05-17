
param(
    [switch]$RebuildIndex
)

function Write-Ok($s){ Write-Host $s -ForegroundColor Green }
function Write-Warn($s){ Write-Host $s -ForegroundColor Yellow }
function Write-Err($s){ Write-Host $s -ForegroundColor Red }

Write-Host "Checking for Python 3.11..."
$pythonExec = $null
try {
    $pyOut = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($pyOut) { $pythonExec = $pyOut.Trim() }
} catch {}
if (-not $pythonExec) {
    try {
        $ver = (& python -c "import sys; print(sys.version)") -join ""
        if ($ver -and $ver.StartsWith("3.11")) {
            $pythonExec = (Get-Command python).Source
        }
    } catch {}
}
if (-not $pythonExec) {
    Write-Err "Python 3.11 was not found. Please install Python 3.11 and ensure 'py -3.11' or 'python' points to it."
    Write-Host "Download: https://www.python.org/downloads/release/python-311x/"
    Exit 1
}
Write-Ok "Using Python: $pythonExec"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$venvDir = Join-Path $repoRoot "ScrapperVenv311"

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating venv at $venvDir ..."
    & $pythonExec -m venv $venvDir
} else {
    Write-Ok "Venv already exists at $venvDir"
}

$pythonVenv = Join-Path $venvDir "Scripts\python.exe"

Write-Host "Upgrading pip..."
& $pythonVenv -m pip install --upgrade pip setuptools wheel

Write-Host "Installing requirements.txt..."
& $pythonVenv -m pip install -r (Join-Path $repoRoot "requirements.txt")

Write-Ok "Installed requirements."

Write-Host "Installing optional helpers (tiktoken, openai)..."
& $pythonVenv -m pip install tiktoken openai --upgrade

Write-Host "\nIf you use GROQ or HuggingFace tokens, set them now (PowerShell example):"
Write-Host "  $env:GROQ_API_KEY='YOUR_GROQ_KEY'"
Write-Host "  $env:HUGGINGFACEHUB_API_TOKEN='YOUR_HF_TOKEN'"

if ($RebuildIndex) {
    Write-Host "Running metadata_cleaner..."
    & $pythonVenv (Join-Path $repoRoot 'src\utils\metadata_cleaner.py') --apply
    Write-Host "Building vector index (this may take several minutes)..."
    & $pythonVenv (Join-Path $repoRoot 'src\utils\build_vector_index.py')
    Write-Ok "Index build complete."
}

Write-Ok "Done. To run the chat in this venv (PowerShell):"
Write-Host "  .\ScrapperVenv311\Scripts\Activate.ps1"
Write-Host "  python src\chat.py"
