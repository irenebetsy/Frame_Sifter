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

def sift_match(input_image, frame, ratio_threshold=0.75):
    # Convert both images to grayscale
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors with SIFT in both images
    kp1, des1 = sift.detectAndCompute(gray_img, None)
    kp2, des2 = sift.detectAndCompute(gray_frame, None)

    # Debug: Print number of keypoints
    print(f"Input image keypoints: {len(kp1)}, Frame keypoints: {len(kp2)}")

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

    # Extract location of good matches
    if len(good_matches) >= 4:  # Need at least 4 matches for RANSAC
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Use RANSAC to find homography and filter matches
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Count inlier matches
        inliers = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
        print(f"Inlier matches after RANSAC: {len(inliers)}")
        
        return len(inliers) >= 4, inliers, kp1, kp2  # Need at least 4 inliers

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
        
        match_found, good_matches, kp1, kp2 = sift_match(input_image, frame)
        if match_found:
            matched_frame_count += 1
            frames.append(frame)
            print(f"Match found in frame {frame_count}")
            
            # Visualize the matches
            if good_matches:
                img_matches = cv2.drawMatches(
                    input_image, kp1, frame, kp2, good_matches[:10], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imshow('Matches', img_matches)
                cv2.waitKey(1)  # Show for 1 ms
        
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
