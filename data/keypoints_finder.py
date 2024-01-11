import os

# Main directory containing subdirectories
main_directory = 'data/NW-UCLA'

# Loop over subdirectoriesin
for folder_name in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder_name)

    # Check if the item in the main directory is a directory
    if os.path.isdir(folder_path):
        # Loop over videos in the subdirectory
        for video_file in os.listdir(folder_path):
            if video_file.endswith(".avi"):
                video_path = os.path.join(folder_path, video_file)
                video_filename = os.path.splitext(video_file)[0]

                # Generate keypoints file path
                keypoints_file = os.path.join(folder_path, f'{video_filename}_keypoints.json')
                #angles_file = os.path.join(folder_path, f'{video_filename}_angles.json')

                # Call the script with the video file
                #command = f'python keypointsfinder.py --source "{video_path}" --device "0"'
                command = f'python data/inference.py --input "{video_path}" --model data/vitpose-h-coco_25.pth --model-name h'
                os.system(command)

                # Check if keypoints.json exists
                if os.path.exists('keypoints.json'):
                    # Rename the generated keypoints file
                    os.rename('keypoints.json', keypoints_file)
                    print(f'Processed {video_file} and saved {keypoints_file}')
                else:
                    print(f'Keypoints file not found for {video_file}')
                
