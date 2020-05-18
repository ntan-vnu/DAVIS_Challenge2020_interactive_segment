from davisinteractive.session import DavisInteractiveSession
from pipeline_chiet import run_interactive

import time
import numpy as np
import glob
import cv2

# Configuration used in the challenges
max_nb_interactions = 8 # Maximum number of interactions 
max_time_per_interaction = 30 # Maximum time per interaction per object

# Total time available to interact with a sequence and an initial set of scribbles
max_time = max_nb_interactions * max_time_per_interaction # Maximum time per object

def gen_black_masks(sequence):
    files = glob.glob('/home/ntan/DAVIS2020/data/test-dev/JPEGImages/480p/%s/*.jpg'%(sequence))
    img_height, img_width = cv2.imread(files[0]).shape[:2]
    return np.zeros([len(files), img_height, img_width]), len(files)
    pass

def compute_req_indices(scribbles, nb_frame):
    index = 0
    for i, l_obj in enumerate(scribbles['scribbles']):
        if len(l_obj) > 0:
            index = i
            break
    return [(index + int(np.ceil(nb_frame/8)) * i) % nb_frame for i in range(8)]
    pass

with DavisInteractiveSession(host='https://server.davischallenge.org', 
                          user_key='ed3db2783547b034398e89f6163b955dc1a0135b33d5ae346ce61aa8dcf3d356', # ntan
                        #   user_key='fae1b878c12cec0857a64b13154a45cd9dd4be1f450a0064e4a2fde32020f144', # nhdang
                            # user_key='d028a4c4bc13459359b63c9d948e91e95c1998c1a474cac07cdbfd4be811dadc', # qckan
                             davis_root='/home/ntan/DAVIS2020/data/test-dev', 
                             subset='test-dev',
                             max_nb_interactions=max_nb_interactions, 
                             max_time=max_time) as sess:
    while sess.next():
        st = time.time()
        # Get the current interaction's scribbles
        sequence, scribbles, _ = sess.get_scribbles()
        
        saved_masks = glob.glob('output/masks_test-dev_chiet/%s_*'%(sequence))
        saved_masks.sort()
        if len(saved_masks) > 0:
            pred_masks = np.load(saved_masks[-1])
            sess.submit_masks(pred_masks)
            print('submitted...')
            continue

        save_scribbles = [scribbles]
        print(sequence)

        pred_masks, nb_frame = gen_black_masks(sequence)
        req_indices = compute_req_indices(scribbles, nb_frame)
        print(req_indices)
        print('request', req_indices[0])
        for i in req_indices[1:]:
            # submit black masks
            sess.submit_masks(pred_masks, [i])
            _, next_scribbles, _ = sess.get_scribbles(True)
            save_scribbles.append(next_scribbles)
            print('request', i)

        # Your model predicts the segmentation masks from the scribbles
        pred_masks = run_interactive(sequence, save_scribbles, req_indices)
        print('predicted...')
        
        np.save('output/masks_test-dev_chiet/%s_frame_%03d'%(sequence, req_indices[0]), pred_masks)
        print('saved pred_masked')

        # Submit your prediction
        sess.submit_masks(pred_masks)
        print('submitted...')
        
        print(time.time()-st)

    # Get the DataFrame report
    report = sess.get_report()

    # Get the global summary
    summary = sess.get_global_summary(save_file='summary.json')

