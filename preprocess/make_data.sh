# /bin/bash

initialdir=`pwd`

invalidimageids=(229146 249406 249407 282008 297689 \
                312870 313953 316542 317121 322060 \
                336708 341918 348187 365243 373523 \
                368889 377213 424849 310441 348166 \
                257275 322438 272229 296788 303081 \
                274420 314141 642409 248516 375331)

xmlfiles=`find $1 -name "*Resting_State_fMRI*.xml"`
i=0
for xmlfile in $xmlfiles; do
    echo ''
    echo xmlfile: $xmlfile

    imageid=`xmllint --xpath '//project/subject/study/series/imagingProtocol/imageUID/text()' $xmlfile`
    invalid="false"
    for invalidid in "${invalidimageids[@]}"; do
    	if [ $invalidid -eq $imageid ]; then
            invalid="true"
            break
        fi
    done
    if [ $invalid == "true" ]; then
        echo ''
        echo ''
        echo ''
        echo invalid image id: $imageid
        echo ''
        echo ''
        echo ''	
        continue
    fi

    cd $initialdir

    label=`xmllint --xpath '//project/subject/subjectInfo[attribute::item="DX Group"]/text()' $xmlfile`
    subjectid=`xmllint --xpath '//project/subject/subjectIdentifier/text()' $xmlfile`
    
    time=`xmllint --xpath '//project/subject/study/series/dateAcquired/text()' $xmlfile`
    date=${time:0:10}    
   
    #seriesid=S`xmllint --xpath '//project/subject/study/series/seriesIdentifier/text()' $xmlfile`
    echo subject id: $subjectid
    echo date: $date    

    T1dir=$1/$subjectid/MPRAGE/$date*    
    fMRIdir=$1/$subjectid/Resting_State_fMRI/$date*
    if [ -d $T1dir ] && [ -d $fMRIdir ];then
       	i=`expr $i + 1`
        echo both files exist

        mkdir -p $label
        cd $label
    	mkdir -p $subjectid
        cd $subjectid
    	mkdir -p $date
        cd $date
    	#mkdir -p $seriesid

    	mkdir T1
    	cp $T1dir/*/*.dcm ./T1
        mkdir fMRI
        cp $fMRIdir/*/*.dcm ./fMRI
    fi
done
 
echo ''
echo valid files number: $i   
    


  


