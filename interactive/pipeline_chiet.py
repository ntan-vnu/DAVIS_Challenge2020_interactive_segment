import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

import cv2
import sys
import numpy as np
import glob
import json


sys.path.append('/home/ntan/fbrs_interactive_segmentation')
from fbrs_nogui import create_controller

sys.path.append('/home/ntan/SiamMask')
from siammask_nogui import create_siammask, siammask_process

sys.path.append('/home/ntan/CascadePSP')
from cascadepsp import cascade_psp_finetune


def mask_to_bbox(mask):
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax-xmin, ymax-ymin]


_fbrs_seg = create_controller() # RGB
_siammask = create_siammask()   # BGR
DAVIS_ROOT = '/home/ntan/DAVIS2020/data/test-dev/'
MAX_NB_POINT = 5

def finetune_cascade_single_mask(mask, img):
    return cascade_psp_finetune(img, mask.astype(np.float)*255)
    pass

def finetune_cascade(masks, file_images):
    res = []
    for i, mm in enumerate(masks):
        img = cv2.imread(file_images[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ft_mask = cascade_psp_finetune(img, mm.astype(np.float)*255)
        res.append(ft_mask)
        print('cascade finetune', i, end='\r')
    return res
    pass


def fbrs_scribble_to_mask(image, dict_ob_paths, cur_obj_id, image_width, image_height):
    global _fbrs_seg
    _fbrs_seg.set_image(image)
    
    for ob_id, ob_paths in dict_ob_paths.items():
        if cur_obj_id != ob_id:
            continue
        for pth in ob_paths:
            nb_point = len(pth)
            step = int(nb_point/MAX_NB_POINT)
            step = step if step > 0 else MAX_NB_POINT
            selected_indices = np.arange(0, nb_point, step)
            pth = np.array(pth)[selected_indices] * np.array([image_width, image_height])
            pth = pth.astype(np.int32)
            for (x, y) in pth:
                _fbrs_seg.add_click(x, y, True)
        
    # mask = _fbrs_seg.get_result_mask()*255
    # cv2.imwrite('output/mask.jpg', mask)
    # blended_image = _fbrs_seg.get_visualization()
    # cv2.imwrite('output/vis.jpg', cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))

    mask = _fbrs_seg.get_result_mask() #*255
    _fbrs_seg.finish_object()
    _fbrs_seg.reset_mask()
    bbox = mask_to_bbox(mask)
    return mask, bbox
    pass

def siammask_track(bbox, file_images):
    global _siammask
    images = [cv2.imread(it) for it in file_images]
    masks = siammask_process(_siammask, images, bbox)
    return finetune_cascade(masks, file_images)
    pass

def add_masks(submit_masks, masks, ob_id, st_ind, en_ind):
    for i in range(len(masks)):
        submit_masks[i+st_ind][masks[i].astype(bool)] = int(ob_id)
    return submit_masks
    pass

def scribbles_to_dict_paths(list_objects):
    dict_obj_paths = {}
    for obj in list_objects:
        pth = obj['path']
        ob_id = obj['object_id']
        if ob_id == 0:
            continue
        if ob_id not in dict_obj_paths:
            dict_obj_paths[ob_id] = []
        dict_obj_paths[ob_id].append(pth)
    return dict_obj_paths
    pass

def run_interactive(sequence, scribbles, indices):
    # get images
    files = glob.glob(DAVIS_ROOT+'/JPEGImages/480p/'+sequence+'/*.jpg')
    files.sort()
    print(len(files), ' images found...')
    
    nb_frames = len(files)
    image_height, image_width, _ = cv2.imread(files[0]).shape
    pred_masks = np.zeros([len(files), image_height, image_width], dtype=np.int)
    min_ind, max_ind = min(indices), max(indices)

    try:
	    for ii, scrib in enumerate(scribbles):
	        i_frame = indices[ii]
	        print('handling frame', i_frame)
	        list_objects = scrib['scribbles'][i_frame]
	        dict_obj_paths = scribbles_to_dict_paths(list_objects)
	        print(dict_obj_paths.keys())

	        image = cv2.imread(files[i_frame])
	        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	        if min_ind < i_frame and i_frame < max_ind:
	            for obj_id, obj_paths in sorted(dict_obj_paths.items()):
	                obj_mask, obj_bbox = fbrs_scribble_to_mask(image, dict_obj_paths, obj_id,
	                                                    image_width, image_height)
	                tracked_masks = siammask_track(obj_bbox, files[i_frame:i_frame+nb_frames//16+1])
	                pred_masks = add_masks(pred_masks, tracked_masks,
	                                            obj_id, i_frame, i_frame+nb_frames//16)
	                pass
	            
	            for obj_id, obj_paths in sorted(dict_obj_paths.items()):
	                obj_mask, obj_bbox = fbrs_scribble_to_mask(image, dict_obj_paths, obj_id,
	                                                    image_width, image_height)
	                tracked_masks = siammask_track(obj_bbox, files[i_frame-nb_frames//16:i_frame+1][::-1])
	                tracked_masks = tracked_masks[1:][::-1]
	                pred_masks = add_masks(pred_masks, tracked_masks, obj_id,
	                                        i_frame-nb_frames//16, i_frame-1)
	                pass
	        
	        if i_frame == min_ind:
	            for obj_id, obj_paths in sorted(dict_obj_paths.items()):
	                obj_mask, obj_bbox = fbrs_scribble_to_mask(image, dict_obj_paths, obj_id,
	                                                    image_width, image_height)
	                tracked_masks = siammask_track(obj_bbox, files[i_frame:i_frame+nb_frames//16+1])
	                pred_masks = add_masks(pred_masks, tracked_masks,
	                                            obj_id, i_frame, i_frame+nb_frames//16)
	                pass

	            for obj_id, obj_paths in sorted(dict_obj_paths.items()):
	                obj_mask, obj_bbox = fbrs_scribble_to_mask(image, dict_obj_paths, obj_id,
	                                                    image_width, image_height)
	                tracked_masks = siammask_track(obj_bbox, files[0:i_frame+1][::-1])
	                tracked_masks = tracked_masks[1:][::-1]
	                pred_masks = add_masks(pred_masks, tracked_masks, obj_id,
	                                        0, None)
	                pass
	            pass

	        if i_frame == max_ind:
	            for obj_id, obj_paths in sorted(dict_obj_paths.items()):
	                obj_mask, obj_bbox = fbrs_scribble_to_mask(image, dict_obj_paths, obj_id,
	                                                    image_width, image_height)
	                tracked_masks = siammask_track(obj_bbox, files[i_frame:])
	                pred_masks = add_masks(pred_masks, tracked_masks,
	                                            obj_id, i_frame, None)
	                pass

	            for obj_id, obj_paths in sorted(dict_obj_paths.items()):
	                obj_mask, obj_bbox = fbrs_scribble_to_mask(image, dict_obj_paths, obj_id,
	                                                    image_width, image_height)
	                tracked_masks = siammask_track(obj_bbox, files[i_frame-nb_frames//16:i_frame+1][::-1])
	                tracked_masks = tracked_masks[1:][::-1]
	                pred_masks = add_masks(pred_masks, tracked_masks, obj_id,
	                                        i_frame-nb_frames//16, i_frame-1)
	                pass
	            pass
    except Exception as e:
    	print(e)
    	pass

    return pred_masks
    pass

