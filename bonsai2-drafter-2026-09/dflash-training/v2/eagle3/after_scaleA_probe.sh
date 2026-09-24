#!/usr/bin/env bash
# When scale-up A finishes, probe its final EMA checkpoint on the three 200-token prompts (acceptance only).
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; MD=/home/REDACTED/Bonsai-demo/models/bonsai2-eagle3
until grep -q 'SCALE-UP A DONE' $V2/logs/eagle3_scaleA_launcher.log 2>/dev/null; do sleep 120; done
CK=$MD/bonsai2_eagle3_scaleA_ema.safetensors; [ -f "$CK" ] || CK=$MD/bonsai2_eagle3_scaleA.safetensors
exec $V2/eagle3/eagle3_ckpt_probe.sh "$CK" scaleA-final
