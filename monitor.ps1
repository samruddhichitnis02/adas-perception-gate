$Host.UI.RawUI.WindowTitle='Training monitor'
cd d:\Samruddhi\adas-perception-gate
while($true){
  Clear-Host
  Get-Date
  "--- GPU ---"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
  "--- Active run ---"
  $r = Get-ChildItem runs -Directory -ErrorAction SilentlyContinue | Where-Object Name -ne 'degraded_s0' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if($r){ $r.Name; $f="$($r.FullName)\results.csv"
    if(Test-Path $f){ Import-Csv $f | Select-Object epoch,@{n='train_box';e={[math]::Round($_.'train/box_loss',3)}},@{n='train_cls';e={[math]::Round($_.'train/cls_loss',3)}},@{n='mAP50';e={[math]::Round($_.'metrics/mAP50(B)',3)}},@{n='mAP50-95';e={[math]::Round($_.'metrics/mAP50-95(B)',3)}},@{n='min';e={[math]::Round($_.time/60,1)}} | Format-Table -AutoSize }
    else { "waiting for first epoch to finish ..." } }
  "--- finished runs ---"
  Get-ChildItem results\training -Filter *_meta.json -ErrorAction SilentlyContinue | ForEach-Object { $_.BaseName }
  Start-Sleep 5
}
