import argparse
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
        "team": team,
        "visible": bool(visible),
        "pose_index": pose,
        "action": action
    }

def get_player_annotation(clip_dir):
    with open(clip_dir) as f:
        for line in f:
            data = parse_track_annotation_line(line)
            print(data)

