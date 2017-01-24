"""
   Input from grouped data (e.g. ~/data/ADNI/Philips/resutls)
"""

import os
import sys


data_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])

def read_subject(file):
    time_series = []
    with open(file, 'r') as f:
        for line in f:
            time_series.append(line.strip())

    return time_series

for _, dirs, _ in os.walk(data_dir, False):
    for diagnosis in dirs:
        diagnosis_dir = data_dir + '/' + diagnosis
        print(diagnosis_dir)
        diagnosis_output_file = output_dir + '/' + diagnosis + '.txt'
        print("file:" + diagnosis_output_file)
        with open(diagnosis_output_file, 'a') as diagnosis_output:
            for _, _, subject_files in os.walk(diagnosis_dir):
                for subject_file in subject_files:
                    subject_path = diagnosis_dir + '/' + subject_file
                    print(subject_path)
                    
                    time_series = read_subject(subject_path)
                    for value in time_series:
                        diagnosis_output.write(value)
                        #print(value)
                        diagnosis_output.write("\n")