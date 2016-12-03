# /bin/bash

diagnoses=(AD Normal)

stdskullstrip=$1
AAL2_index_path=$2
AAL2_mask_path=$3

initialdir=$4
cd $initialdir
initialdir=`pwd`
mkdir -p results

for diagnosis in "${diagnoses[@]}"; do
    echo group: $diagnosis
    cd $diagnosis
    diagnosisdir=`pwd`
    for subject in ./*/ ; do
        echo subject: $subject        
        cd $subject
        subjectdir=`pwd`
	for time in ./*/ ; do
           echo time: $time
	   cd $time
           date=${time:2:10}
           $initialdir/preprocess_subject.sh ${date//-/''} $stdskullstrip $AAL2_index_path $AAL2_mask_path ${initialdir}/results/${subject:2:10}-${date//-/''} > /tmp/subject.txt 2>&1 
           
           cd $subjectdir
	done

        cd $diagnosisdir
    done
    
    cd $initialdir 
done 
