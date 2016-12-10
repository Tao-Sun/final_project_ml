import os
import sys

"""
   Input from grouped data (e.g. ~/tmp/group_results)
"""

data_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])

def read_time_series(file):
    time_series = []
    with open(file, 'r') as f:
        for line in f:
            time_series.append(line.strip())
    
    return time_series

for _, dirs, _ in os.walk(data_dir, False):
    for diagnosis in dirs:
	print diagnosis
	diagnosis_dir = data_dir + '/' + diagnosis
        diagnosis_output_file = output_dir + '/' + diagnosis + '.txt'
        #print "file:" + diagnosis_output_file	
        with open(diagnosis_output_file, 'a') as output:
            for _, image_dirs, _ in os.walk(diagnosis_dir):
                for image_id in image_dirs:
                    #print image_id
                    image_dir = diagnosis_dir + '/' + image_id
                    for _, _, time_series_files in os.walk(image_dir):
                        for time_series_file in time_series_files:
                            time_series_path = image_dir + '/' + time_series_file
                            time_series = read_time_series(time_series_path)
                            for value in time_series:
                                output.write(value + ",")
                            output.write("\n")
                    
                    output.write("\n")
                
                        
                        
