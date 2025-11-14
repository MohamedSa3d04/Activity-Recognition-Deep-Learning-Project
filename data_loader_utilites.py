import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from model_loader_utilites import preprocess_images

def parse_track_annotation_line(line):
    '''
    Get the annotations for each player which represented as a line
    '''
    parts = line.strip().split()
    pid = int(parts[0]) #Playaer_id
    x1, y1, x2, y2 = map(int, parts[1:5]) # box info
    frame = int(parts[5])    # Frame Number
    team = int(parts[6]) # 1: right 0:left
    visible = int(parts[7]) #0: no 1: Visible
    pose = int(parts[8]) #hase pose? 1:True
    action = parts[9] # one of our 9 classes
    return {
        "player_id": pid,
        "bbox": (x1, y1, x2, y2),
        "frame": frame,
        "tgeam": team,
        "visible": bool(visible),
        "pose_index": pose,
        "action": action
    }

# Give me the video path - Return you all clips annotations in annotations.txt file
def get_video_annotations_dictionary(vid_path):
    annotations_dir = os.path.join(vid_path, 'annotations.txt')
    annotations_dictionary = {}
    with open(annotations_dir) as annotations: 
        for line in annotations: # each line represent some frames annotations
            parts = line.strip().split()
            clip_name = parts[0]
            target = parts[1]
            annotations_dictionary[clip_name] = target
    
    return annotations_dictionary

def parsing_scense_annotations(main_path):
    '''
    In follwing punch of codes, I will try to have all mid-frame ids and annotation (scene-level)
    from each clip from each video (Used for BaseLine 1)!
    # '''
    try:
        preprocessor, model, device = preprocess_images()
        videos_folders = os.listdir(main_path) # all folder in the main path folder
        for video_name in tqdm(videos_folders):

            cur_vid = os.path.join(main_path, video_name) #Having annotations.txt
            video_annotation = get_video_annotations_dictionary(cur_vid)
            clips_folders = [clip_name for clip_name in os.listdir(cur_vid) if os.path.isdir(os.path.join(cur_vid, clip_name))] # getting all the clips in the vdieo dir
            for clip_name in clips_folders: # Moving in each clip in the video
            
                cur_clip = os.path.join(cur_vid, clip_name) # cur_clip path
                clip_frames = [frame_name for frame_name in os.listdir(cur_clip) if frame_name in video_annotation] # all frames in the current clip
                
                for frame in clip_frames: # Moving in each frame (only annotated) in the clip
                    frame_path = os.path.join(cur_clip, frame)
                    img = Image.open(frame_path).convert('RGB')
                    img_tensor = preprocessor(img).unsqueeze(0)
                    ann = video_annotation[frame]

                    featrues = model(img_tensor.to(device)) #(T, 2048, 1, 1)
                    featrues = featrues.view(2048, -1).numpy()


                    save_path = os.path.join(cur_clip, 'frames_features_extracted', frame)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    np.savez(save_path, features=featrues, labels=np.array([ann]))
                    print(f"Saved: {save_path}")
                    
    except Exception as e:
            print(f"An error occurred: {e}")
       
            




    




def draw_annotations(frame_path, annotations):
    """
    Draw bounding boxes and labels for each player on a given frame.
    annotations: list of dicts from parse_track_annotation_line()
    """
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        pid = ann["player_id"]
        action = ann["action"]
        color = (0, 255, 0) if ann["tgeam"] == 0 else (255, 0, 0)

        # draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # label text: player id + action
        label = f"ID:{pid} | {action}"
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()



def get_player_annotation(clip_dir):
    all_annotations = []
    with open(clip_dir) as f:
        for line in f:
            data = parse_track_annotation_line(line)
            print(data)

            if data["frame"] == 13286:  # visualize a specific frame
                all_annotations.append(data)
        
        draw_annotations("/content/drive/MyDrive/proj_dl_data/data/videos/0/13286/13286.jpg", all_annotations)
