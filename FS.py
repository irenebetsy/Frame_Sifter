import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip

def sift_match(input_image, frame, ratio_threshold=0.75):
    # Convert both images to grayscale
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors with SIFT in both images
    kp1, des1 = sift.detectAndCompute(gray_img, None)
    kp2, des2 = sift.detectAndCompute(gray_frame, None)

    # If there are no descriptors in either image, return false
    if des1 is None or des2 is None:
        return False, [], [], []

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Apply the ratio test to filter matches
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # Log number of good matches
    print(f"Good matches found: {len(good_matches)}")

    # If we have enough good matches, return True along with the keypoints
    return len(good_matches) > 5, good_matches, kp1, kp2

def extract_segments(video_path, image_path, output_path, match_threshold=5):
    # Load the input image (object to detect)
    input_image = cv2.imread(image_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0
    matched_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform SIFT matching between the input image and the current frame
        match_found, good_matches, kp1, kp2 = sift_match(input_image, frame)
        if match_found:
            matched_frame_count += 1
            frames.append(frame)
            print(f"Match found in frame {frame_count}")
            
            # Visualize the matches only if we have good keypoints
            if good_matches:
                img_matches = cv2.drawMatches(
                    input_image, kp1, frame, kp2, good_matches[:10], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imshow('Matches', img_matches)

                # Check if 'q' is pressed to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

    # Check if we found any matches and create output video
    if frames:
        print(f"Total matched frames: {matched_frame_count}")
        clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=30)
        clip.write_videofile(output_path, codec='libx264')
        print(f"Extracted video saved as: {output_path}")
    else:
        print("No segments found containing the image.")


# Paths to the input video
input_video_path = "D:\projects\svn\FrameSifter\Data\MN_Promo1.mp4"
input_video=input_video_path.replace('\\','/')
input_image_path = "D:\projects\svn\FrameSifter\Data\WhatsApp Image 2024-09-27 at 5.42.57 PM.jpeg"
input_image=input_image_path.replace('\\','/')
output_video_path = "D:/projects/svn/FrameSifter/Data/output_video.mp4"


# Run the function
extract_segments(input_video, input_image_path, output_video_path, match_threshold=5)

