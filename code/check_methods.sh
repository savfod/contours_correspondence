DATA_DIR=$1

IM_MASK="B*"
GENERATED_FOLDERS=$DATA_DIR/generated/*
FIXED_IMAGES=$DATA_DIR/$IM_MASK
echo "calculating diff in files"
echo `ls $GENERATED_FOLDERS`
echo "Vs"
echo `ls $FIXED_IMAGES`


touch table.txt
rm table.txt
for diff_method in AV DYN DYN2 COV COVC AVMIN AVC; do
	for method in NO MASS TAN M1 TR CR1 CR2 CUR ACOR M1.5 M2; do
# for diff_method in `seq 3`; do
	# for method in `seq 3`; do
		echo $method;
		touch res.txt;
		rm res.txt;
		# for IMG in `seq 40`; do
		for FLD in $GENERATED_FOLDERS; do
			# pypy draw_wurfs.py -um 2 -n -s 0 -lfp 200 -dm $diff_method -wm $method -f data/PTRI_17/$IMG/{3,4,5}.txt -f2 data/RI_17_REPEATED/$IMG/{3,4,5}.txt >> res.txt;
			# pypy draw_wurfs.py -um 2 -n -s 0 -lfp 200 -dm $diff_method -wm $method -f data/PTRI_89/$IMG/{3,4,5}.txt -f2 data/RI/89/{3,4,5}.txt >> res.txt;
			# pypy draw_wurfs.py -um 3 -n -s 0 -lfp 200 -dm $diff_method -wm $method -f data/PTRI_89/$IMG/{3,4,5}.txt -f2 data/RI/89/{3,4,5}.txt >> res.txt;
			# echo $FLD/* "AND" $FIXED_IMAGES
			pypy code/draw_wurfs.py -um 3 -n -s 0 -lfp 100 -dm $diff_method -wm $method -f $FLD/* -f2 $FIXED_IMAGES >> res.txt;
			# pypy draw_wurfs.py -um 3 -n -s 0 -lfp 100 -dm $diff_method -wm $method -f openData/Lakes/generated/$IMG/O{1,2,3}_gen.txt -f2 openData/Lakes/OM_renamed/OMR{1,2,3,4,5,6}.txt >> res.txt;

		done;
		v=`python3 code/average.py res.txt`;
		echo $diff_method $method $v
		echo $diff_method $method $v >> ${DATA_DIR}/table.txt
	done;
done;
python3 code/make_table.py ${DATA_DIR}/table.txt
echo "$GENERATED_FOLDERS vs $FIXED_IMAGES" > ${DATA_DIR}/final_table.csv
python3 code/make_table.py ${DATA_DIR}/table.txt >> ${DATA_DIR}/final_table.csv
