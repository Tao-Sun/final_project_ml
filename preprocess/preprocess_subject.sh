#~/bin/bash

basename=$1
stdskullstrip=$2
AAL2_index_path=$3
AAL2_mask_path=$4
output_prefix=$5

initialdir=`pwd`

### Step 1: preprocessing of T1 images ### 
cd T1
rm -rf preprocess
mkdir preprocess

# .dcm to nii
dcm2nii -o ./preprocess *.dcm

cd ./preprocess

# remove skull
3dSkullStrip -o_ply skullstrip_mask.nii -input ${basename}*.nii
3dcalc -prefix skullstrip.nii -expr 'a*step(b)' -b skullstrip_mask.nii -a ${basename}*.nii
### Step 1: ends ###

cd $initialdir
### Step 2: preprocessing of fMRI images ###
cd fMRI
rm -rf preprocess
mkdir preprocess

# .dcm to nii
dcm2nii -o ./preprocess *.dcm

cd ./preprocess

# discard first 5 images to allow for T1 equilibration effects
i=1
while [ $i -le 5 ];
do
    rm *00$i.nii
    i=`expr $i + 1`
    
done

fslmerge -t ${basename}_4d.nii *.nii

## Correct functional image for slice-timing
3dTshift -prefix slicetiming.nii  -tpattern altplus  ${basename}_4d.nii
    
## Correct functional image for motion
3dvolreg -prefix motioncorrection.nii  -1Dfile _motion.1D  -Fourier -twopass -zpad 4  \
         -base 50 slicetiming.nii

## Despike functional image
3dDespike -prefix despike.nii -ssave _spikiness.nii motioncorrection.nii

fslsplit despike.nii despike

### Step 2: ends ###

### Step 3: Get functional-to-anatomical image registration ###

# Get functional-to-anatomical image registration
flirt \
 -omat func2anat.mat \
 -cost corratio -dof 12 -interp trilinear \
 -ref ${initialdir}/T1/preprocess/skullstrip.nii \
 -in  despike0050.nii

# Get anatomical-to-standard image registration
flirt \
 -omat anat2stnd.mat \
 -cost corratio -dof 12 -interp trilinear \
 -ref $stdskullstrip \
 -in $initialdir/T1/preprocess/skullstrip.nii

# apply MRI to standard registration
flirt -out registration_T1.nii -interp trilinear -applyxfm -init anat2stnd.mat \
        -ref $stdskullstrip -in $initialdir/T1/preprocess/skullstrip.nii

# Get functional-to-standard image transformation
convert_xfm \
 -omat func2stnd.mat \
 -concat anat2stnd.mat func2anat.mat

## Apply functional-to-standard image registration.
i=0
while [ $i -lt 135 ];
do
    echo $i
    if [ $i -lt 10 ]; then
        fMRI_index=000$i
    elif [ $i -lt 100 ]; then
        fMRI_index=00$i
    else
        fMRI_index=0$i
    fi
    echo $fMRI_index
    
    flirt -out registration_fMRI_$fMRI_index.nii -interp trilinear -applyxfm -init func2stnd.mat \
        -ref $stdskullstrip -in despike$fMRI_index.nii
    i=`expr $i + 1`
    
done


fslmerge -t registration_fMRI_4d.nii registration_fMRI*.nii
### Step 3: ends ###


### Step 4: Remove noise signal ###

## Segment anatomical image
fsl5.0-fast -o T1_segm_A -t 1 -n 3 -g -p registration_T1.nii

## Get mean signal of CSF segment
3dmaskave \
    -quiet \
    -mask T1_segm_A_seg_0.nii \
    registration_fMRI_4d.nii > fMRI_csf.1D

# motion correction in the standard space.
3dvolreg -prefix _mc_F.nii -1Dfile fMRI_motion.1D  -Fourier -twopass -zpad 4  \
   -base 50 registration_fMRI_4d.nii

## Get motion derivative
1d_tool.py \
   -write fMRI_motion_deriv.1D \
   -derivative \
   -infile fMRI_motion.1D \

## Concatenate CSF signal, motion parameters, motion
## derivative into 'noise signal'
1dcat fMRI_csf.1D fMRI_motion.1D \
    fMRI_motion_deriv.1D > fMRI_noise.1D

## Regress out the 'noise signal' from functional image
3dBandpass \
    -prefix fMRI_removenoise.nii \
    -mask registration_fMRI_0003.nii \
    -ort fMRI_noise.1D \
    0.01 0.08 \
    registration_fMRI_4d.nii

i=1
cat $AAL2_index_path | while read line; do
    roi_value=$(echo $line | tr -d '\r')

    3dmaskave \
        -quiet \
        -mrange $(echo $roi_value-0.1 | bc) $(echo $roi_value+0.1 | bc) \
        -mask $AAL2_mask_path\
        fMRI_removenoise.nii > ${output_prefix}-${i}.1D

    i=`expr $i + 1`
done;

### Step 4: ends ###


