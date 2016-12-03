# /bin/bash

# This script groups time series files of an image into its dedicated folder.  

# $1: resultdir, folder of preprocess results
# $2: datadir, folder of paired T1 and fMRI images
# $3: outputdir, output folder

diagnoses=(AD Normal)

initialdir=`pwd`

resultdir=$1
cd $resultdir
resultdir=`pwd`

cd $initialdir
outputdir=$3
cd $outputdir
outputdir=`pwd`
mkdir ${diagnoses[0]}
mkdir ${diagnoses[1]}

cd $initialdir
datadir=$2
cd $datadir
datadir=`pwd`

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
           imageprefix=${subject:2:10}-${date//-/''}
           mkdir $outputdir/$diagnosis/$imageprefix
           cp $resultdir/$imageprefix* $outputdir/$diagnosis/$imageprefix
           cd $subjectdir
        done

        cd $diagnosisdir
    done

    cd $datadir
done
