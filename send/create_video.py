import os
import cv2
from tqdm import tqdm
import glob


def make_video(image_paths, vid_name, video_fps=20):
    print("############################")
    print("\nGenerating video ...")
    img = cv2.imread(image_paths[0])
    print("img.shape ", img.shape[:2])
    h, w = img.shape[:2]

    # The following line defines the video codec, set by default to avc1
    # If openCV is not compiled with codec support, it is possible that the video
    # is then not encoded properly and then your browser complains that file media is not supported
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    video = cv2.VideoWriter(vid_name, fourcc, video_fps, (w, h))
    i = 1
    max_state = len(image_paths)
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        # cv2.putText(img,os.path.basename(img_path), (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        video.write(img)
        progress_state = int(100*i/max_state)
        yield "data:" + str(progress_state) + "\n\n"
        i += 1
    video.release()
    print('finished ' + vid_name)


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def analyze_video(file):
    vid = cv2.VideoCapture(file)
    coding = vid.get(cv2.CAP_PROP_FOURCC)
    print(f"coding {decode_fourcc(coding)}")


if __name__ == "__main__":

    out_path = './static/video_summary/'
    os.makedirs(out_path, exist_ok=True)
    out_vid_name = out_path + 'summary.mp4'
    image_folder = "./static/images"
    image_paths = [x for x in glob.glob(f"{image_folder}/*.jpg")]
    image_paths.sort(key=lambda x: int(os.path.basename(x)[:os.path.basename(x).find("_")]))
    make_video(image_paths[:1000], out_vid_name)

    analyze_video(out_vid_name)
