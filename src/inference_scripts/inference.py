import os
import sys
import json
import csv
import glob
import re
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mlp import MLP
from furnari2018 import viterbi, SF1
from heuristic import vsr_algorithm
from openpose_script import openpose_script
from gt_comparison import gtc_algorithm


def count_sequences(input_list):
    """
    Count the consecutive sequences of identical elements in the given list.

    Parameters:
    - input_list (list): The input list containing elements to be analyzed.

    Returns:
    - list: A list of sequences, where each sequence is represented as [element, duration].

    The function iterates through the input list and identifies consecutive sequences
    of identical elements. It calculates the duration of each sequence in seconds (assuming
    a frame rate of 24 frames per second) and returns a list of sequences with their elements
    and respective durations.
    """
    sequences = []
    frame = 1

    for i in range(1, len(input_list)):
        
        if input_list[i] == input_list[i - 1]:
            frame += 1
        else:
            
            # Convert frame to seconds (assuming 24 frames per second)
            frame = frame / 24    
            sequences.append([input_list[i - 1], frame])
            frame = 1

    # Convert the last frame to seconds and add the final sequence
    frame = frame / 24    
    sequences.append([input_list[-1], frame])

    return sequences

def natural_sort(l):
    """
    Sorts a list naturally.

    Parameters:
    - l (list): List to be sorted.

    Returns:
    - list: Sorted list.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)



environment = os.getcwd()
#Input video conversion in order to be elaborated by openpose
#Argument check
if len(sys.argv) != 2:
    print("Error in the input, please insert the video path")
    sys.exit(1)

try:
    input_video = sys.argv[1]
    video_converted_with_ext = "video_to_inference.mp4"
    video_converted = video_converted_with_ext.split('.')[0]
    os.system(f"ffmpeg -i {input_video} -r 24 -vf \"\
              scale=w=960:h=540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2\"\
               {video_converted_with_ext}")
except:
    print("Error in video conversion, check the video format or the video path")

# Elaborating the video with openpose
json_output_path = openpose_script(video_converted_with_ext, video_converted)

#Extracting the features from the json files and building the dataset
print("Creating the dataset...")

with open(environment + '/video_to_predict.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['NoseX', 'NoseY', 'NoseC',
                        'LEyeX', 'LEyeY', 'LEyeC',
                        'REyeX', 'REyeY', 'REyeC',
                        'LEarX', 'LEarY', 'LEarC',
                        'REarX', 'REarY', 'REarC',
                        'LShoulderX', 'LShoulderY', 'LShoulderC',
                        'RShoulderX', 'RShoulderY', 'RShoulderC',
                        'LElbowX', 'LElbowY', 'LElbowC',
                        'RElbowX', 'RElbowY', 'RElbowC',
                        'LWristX', 'LWristY', 'LWristC',
                        'RWristX', 'RWristY', 'RWristC',
                        'LHipX', 'LHipY', 'LHipC',
                        'RHipX', 'RHipY', 'RHipC',
                        'LKneeX', 'LKneeY', 'LKneeC',
                        'RKneeX', 'RKneeY', 'RKneeC',
                        'LAnkleX', 'LAnkleY', 'LAnkleC',
                        'RAnkleX', 'RAnkleY', 'RAnkleC',
                        'UpperNeckX', 'UpperNeckY', 'UpperNeckC',
                        'HeadTopX', 'HeadTopY', 'HeadTopC',
                        'LBigToeX', 'LBigToeY', 'LBigToeC',
                        'LSmallToeX', 'LSmallToeY', 'LSmallToeC',
                        'LHeelX', 'LHeelY', 'LHeelC',
                        'RBigToeX', 'RBigToeY', 'RBigToeC',
                        'RSmallToeX', 'RSmallToeY', 'RSmallToeC',
                        'RHeelX', 'RHeelY', 'RHeelC',
                        'video_name', 'video_frame', 'skill_id'])


folder = json_output_path
print("Reading json files from: ", folder)

folder = glob.glob(folder + "*.json")
folder = natural_sort(folder)
dataframe_local = pd.DataFrame()
total_frames = 0

for file in folder:
    total_frames += 1
    with open(file) as f:
        data = json.load(f)
    
    if data["people"] == []:
        keypoints = [0] * 75

    else:
        keypoints = data["people"][0]["pose_keypoints_2d"]
    

    #Extract the frame number from the file name
    frame_num = file.split("/")[-1]
    frame_num = frame_num.split("_")[3]
    frame_num = frame_num.lstrip("0")
    if frame_num == "":
        frame_num = 0

    frame_num = int(frame_num)

    keypoints.append(video_converted)
    keypoints.append(frame_num)
        
    dataframe_local = dataframe_local.append(pd.DataFrame([keypoints]), ignore_index=True)  

print("Dataframe created!\n", dataframe_local)


with open(environment + '/video_to_predict.csv', 'a') as f:
    dataframe_local.to_csv(f, header=False, index=False)

#Load the classes and the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)
model = MLP()
model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()
data_to_predict = pd.read_csv(environment + '/video_to_predict.csv')

X_pred = data_to_predict.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
X_pred = torch.FloatTensor(X_pred.values).to(device)

#Make the prediction
with torch.no_grad():
    outputs = model(X_pred)
    probabilities = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)


#Save the prediction in a csv file
predicted_video = pd.DataFrame()
raw_predicted = predicted.tolist()
predicted_labels = le.inverse_transform(predicted.tolist())
predicted_video['video_name'] = predicted_labels
probabilities_matrix = probabilities.cpu().numpy()
predicted_video.to_csv('predicted_video.csv', index=False)


df = pd.read_csv("predicted_video.csv")
vsr_predicted, output, output_l = vsr_algorithm(df)

seconds_length = 0
for i in range(0, len(output_l)):
    seconds_length += output_l[i][1]

#Ground truth comparison

gt_predicted, skills_frames_, skills_seconds_ = gtc_algorithm(seconds_length, total_frames)

gt_l = count_sequences(gt_predicted)
gt_l = [x for x in gt_l if x[0] != 'none']



#Video Temporal Segmentation algorithms comparison
raw_results, raw_value = SF1(gt_predicted, raw_predicted)
print("Raw_values: ", raw_value)

vsr_results, vsr_value = SF1(gt_predicted, vsr_predicted)
print("Heuristic values: ", vsr_value)

viterbi_output = viterbi(probabilities_matrix, 10e-20)
paper_results, paper_value = SF1(gt_predicted, viterbi_output)
print("Probabilistic values: ", paper_value)

f.close()
