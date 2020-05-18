from davisinteractive.session import DavisInteractiveSession
from pipeline import run_interactive

import time
import numpy as np

# Configuration used in the challenges
max_nb_interactions = 8 # Maximum number of interactions 
max_time_per_interaction = 30 # Maximum time per interaction per object

# Total time available to interact with a sequence and an initial set of scribbles
max_time = max_nb_interactions * max_time_per_interaction # Maximum time per object

with DavisInteractiveSession(host='https://server.davischallenge.org', 
                          user_key='ed3db2783547b034398e89f6163b955dc1a0135b33d5ae346ce61aa8dcf3d356', # ntan
                        # user_key='fae1b878c12cec0857a64b13154a45cd9dd4be1f450a0064e4a2fde32020f144', # nhdang
                             davis_root='/home/ntan/DAVIS2020/data/test-dev', 
                             subset='test-dev',
                             max_nb_interactions=max_nb_interactions, 
                             max_time=max_time) as sess:
    while sess.next():
        st = time.time()
        # Get the current interaction's scribbles
        sequence, scribbles, _ = sess.get_scribbles()

        # Your model predicts the segmentation masks from the scribbles
        pred_masks, index = run_interactive(scribbles)

        # Submit your prediction
        sess.submit_masks(pred_masks)
        np.save('output/masks_test-dev/%s_frame_%03d'%(sequence, index), pred_masks)
        print(time.time()-st)
    # Get the DataFrame report
    report = sess.get_report()

    # Get the global summary
    summary = sess.get_global_summary(save_file='summary.json')

