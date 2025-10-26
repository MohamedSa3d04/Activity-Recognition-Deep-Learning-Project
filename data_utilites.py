import argparse
import cv2
import matplotlib.pyplot as plt
def parse_track_annotation_line(line):
    parts = line.strip().split()
    pid = int(parts[0]) #Playaer_id
    x1, y1, x2, y2 = map(int, parts[1:5]) # box info
    frame = int(parts[5]) # Frame Number
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

def get_all_frames_from_clip(clip_path):
    pass



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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, color=color)

        # label text: player id + action
        label = f"ID:{pid} | {action}"
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA, color=color)

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
