from argparse import ArgumentParser
from process import Processor

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("interval", metavar="i", type=int, help="no. of frames per second.")
    parser.add_argument("directory_name", metavar="d", type=str, help="Video directory name.")
    parser.add_argument("output_directory_name",metavar="o", type=str, help="CLIP feature output directory name.")

    args = parser.parse_args()

    return args

def main():
    inputs = parse_args()
    fps = inputs.interval
    video_directory_path = inputs.directory_name
    clip_vector_output_directory_path = inputs.output_directory_name

    processor = Processor()
    processor.process_video(video_directory_path, fps, clip_vector_output_directory_path)
    


if __name__=='__main__':
    main()