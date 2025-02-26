SIM=$1
echo "ENSEMBLE Member $SIM: MML 3D static heatmap regression UNet w/ NLLLoss"
python main.py fit --config configs/ensemble/MML_shr_unet3D_NLLLoss.yaml --seed_everything $SIM
