#python run.py --log_dir ./ckpt_lib30/ 
#python run.py --log_dir ./ckpt_base/ --checkpoint ../CFTE/checkpoint_repro_1/vox-256/00000099-checkpoint.pth.tar

#lib0: replace all conv3x3 in dense motion by dw+pw
#lib01: only replace bottleneck by dw+pw
#lib20: bottleneck DW + motion & hourglass CP
#lib21: densemotion: all DW except for hourglass, generator: all DW except for first and final
#lib30: init occ_generate



python run.py --log_dir ./test0/ --arch test0
