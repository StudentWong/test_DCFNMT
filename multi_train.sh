ls ./runs | while read line
do
	if [ "$line" != "__pycache__" ];then
#		python -u ./train_DCFNTM.py -p 20 -j 10 -c runs.${line%.*}.TrackerConfig
		python -u ./train_DCFNTM.py -p 20 -j 10 -c runs.${line%.*}.TrackerConfig > $line.log 2>&1
	fi
#	if [ -d $line -a "$line" != "result" ];then
#		echo $line
#		cd $line
#		zip_file=*.zip
#		ls $zip_file | while read file_name
#		do
#			name=`echo ${file_name%.*}`
#			echo $name
#			unzip -O CP936 $file_name -d ../result/$line/$name
#		done
#		cd ..
#	fi
done
