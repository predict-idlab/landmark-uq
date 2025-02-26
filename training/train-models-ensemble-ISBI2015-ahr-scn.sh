SIM=$1
echo "ENSEMBLE Member $SIM: ISBI2015 adaptive heatmap regression SCN (validtion on fold 2 and test on fold 3)"
python main.py fit --config configs/ensemble/ISBI2015_ahr_scn_val_fold2_test_fold3.yaml --seed_everything $SIM

