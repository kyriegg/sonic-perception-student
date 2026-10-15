param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

# Isolate this run from the global user site-packages, which holds an
# unrelated editable IsaacLab install (D:\IsaacLab) that otherwise shadows
# this project's own IsaacLab_instinct in sys.meta_path.
$env:PYTHONPATH = ""
$env:PYTHONNOUSERSITE = "1"
$env:OMNI_KIT_ACCEPT_EULA = "YES"

& "D:\anconda\envs\instinct_repro\python.exe" $Script @ScriptArgs
