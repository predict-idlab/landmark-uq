SIM=$1
echo "ENSEMBLE Member $SIM: ISBI2015 static heatmap regression UNet w/ NLLLoss (validtion on fold 2 and test on fold 3)"
python main.py fit --config configs/ensemble/ISBI2015_shr_unet_NLLLoss_val_fold2_test_fold3.yaml --seed_everything $SIM

