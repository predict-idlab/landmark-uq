# echo "Train: MML 3D static heatmap regression UNet"
# python main.py fit --config configs/MML_shr_unet3D_NLLLoss.yaml

echo "Train: MML 3D static heatmap regression UNet3D"
python main.py fit --config configs/MML_shr_unet3D_NLLLoss.yaml

echo "Train: MML 3D static heatmap regression UNet3D (benchmark)"
python main.py fit --config configs/MML_shr_unet3D_NLLLoss_benchmark.yaml