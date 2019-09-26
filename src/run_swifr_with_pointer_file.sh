REPCOUNT=0
while read p; 
do
	if [ $REPCOUNT -ge ${3} ] && [ ${4} -gt $REPCOUNT  ]
	then
		echo "On item: $REPCOUNT";
		(head -n 1 data/def15__extra__neutral.txt && awk 'BEGIN{OFS="\t"} {$1=$1};1' $p) | sed 's/-nan/-998/g' > ${2}/tmp_input.${REPCOUNT}.txt;
		python SWIFr.py --path2trained models/swifr_model_v0.1.1.0 \
		--pi 0.999 0.001 --file ${2}/tmp_input.${REPCOUNT}.txt --outfile ${2}/tmp_output.${REPCOUNT}.txt;
		cut -f1,15 ${2}/tmp_output.${REPCOUNT}.txt > ${2}/item_${REPCOUNT}__swifr_test.txt;
		rm ${2}/tmp_input.${REPCOUNT}.txt;
		rm ${2}/tmp_output.${REPCOUNT}.txt;
	fi
	((REPCOUNT++));
done <${1}
