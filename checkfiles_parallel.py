from multiprocessing import Pool
import subprocess
from glob import glob
import os
from skimage import io
from tqdm import tqdm
from argparse import ArgumentParser


def process_files(path_to_pngs, debug=False):
    '''
    Args:
        path_to_pngs (str): Path to directory for a given video that was processed to produces pngs of frames.
    Returns:
        damaged (list): List containing full path (str) to damaged files.
    '''
    frames = []
    damaged = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        frames.extend(glob(os.path.join(path_to_pngs, ext)))
    
    if debug: print("\tprocess N: ", path_to_pngs)
    
    for frame in frames:
        try:
            _ = io.imread(frame)
        except:
            print("\t!!! Damaged file: ", frame)
            damaged.append(frame)
            
    return damaged
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_workers", default=4, type=int, help='Number of parallel workers')
    parser.add_argument("--dir_to_process", 
                        default='/home/jupyter/vox_png_512px/train',
                        help='Path to top level directory to check images for damage (i.e. cannot load).')
    parser.add_argument("--output_file", 
                        default='/home/jupyter/vox_png_512px/damaged.txt',
                        help='Path to file that will contain all damaged images that could not be loaded.')
    
    
    
    args = parser.parse_args()
    
    print("Number of parallel workers: ", args.num_workers)
    print("Processing directories in: ", args.dir_to_process)
    print("Writing damaged file list to: ", args.output_file)
    
    # Get all directory paths (top level) for each exported movie.
    # 
    subdirs = [x[0] for x in os.walk(args.dir_to_process)][1:] 
    pool = Pool(processes=args.num_workers)
    
    # Spawn processes to iterate through directories and check for damaged images.
    # Returns a list of damaged files that we write to disk for inspection/removal
    #
    f = open(args.output_file, 'w')
    for frames in tqdm(pool.imap_unordered(process_files, subdirs)):
        for frame in frames:
            print(frame, file=f)
            f.flush()
            
    f.close()
    pool.close()
    pool.join()

