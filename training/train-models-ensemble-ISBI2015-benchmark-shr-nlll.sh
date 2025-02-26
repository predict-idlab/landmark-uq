SIM=$1
echo "ENSEMBLE Member $SIM: ISBI2015 static heatmap regression UNet w/ NLLLoss (benchmark)"
python main.py fit --config configs/ensemble/ISBI2015_shr_unet_NLLLoss_benchmark.yaml --seed_everything $SIM

