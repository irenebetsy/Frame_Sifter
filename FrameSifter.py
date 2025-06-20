# Paths to the input video
input_video_path = "D:\svn\FrameSifter\Data2\Boomerang UK Doraemon New Show Promo.mp4"
input_video=input_video_path.replace('\\','/')
input_image_path = "D:\svn\FrameSifter\Data2\download.png"
input_image=input_image_path.replace('\\','/')
output_video_path = "D:\svn\FrameSifter\Output\output_video.mp4"
output_video=output_video_path.replace('\\','/')


import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip

def preprocess_image(image):
    """ Preprocess the image for better feature detection. """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

def detect_and_compute_features(image):
    """ Detect and compute SIFT features. """
    gray_image = preprocess_image(image)
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray_image, None)

def sift_match(input_image, frame, ratio_threshold=0.75):
    # Detect and compute features using SIFT
    kp1, des1 = detect_and_compute_features(input_image)
    kp2, des2 = detect_and_compute_features(frame)

    print(f"Input image keypoints: {len(kp1)}, Frame keypoints: {len(kp2)}")

    if des1 is None or des2 is None:
        return False, [], [], []

    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in raw_matches if m.distance < ratio_threshold * n.distance]

    print(f"Good matches found: {len(good_matches)}")

    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        inliers = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
        print(f"Inlier matches after RANSAC: {len(inliers)}")
        
        return len(inliers) >= 4, inliers, kp1, kp2  

    return False, good_matches, kp1, kp2

def extract_segments(video_path, image_path, output_path):
    input_image = cv2.imread(image_path)
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0
    matched_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform SIFT matching for the current frame
        match_found, good_matches, kp1, kp2 = sift_match(input_image, frame)
        if match_found:
            matched_frame_count += 1
            frames.append(frame)
            print(f"Match found in frame {frame_count}")

            if good_matches:
                img_matches = cv2.drawMatches(
                    input_image, kp1, frame, kp2, good_matches[:10], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imshow('Matches', img_matches)
                cv2.waitKey(1)
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

    if frames:
        print(f"Total matched frames: {matched_frame_count}")
        clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=30)
        clip.write_videofile(output_path, codec='libx264')
        print(f"Extracted video saved as: {output_path}")
    else:
        print("No segments found containing the image.")


# Run the function
extract_segments(input_video_path, input_image_path, output_video)
