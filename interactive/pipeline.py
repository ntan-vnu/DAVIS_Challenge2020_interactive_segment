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

# sys.path.append('/home/ntan/CascadePSP')
# from cascadepsp import cascade_psp_finetune


def fbrs_seg_test():
    fbrs_seg = create_controller()
    img = cv2.imread('images/sheep.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fbrs_seg.set_image(img)
    fbrs_seg.add_click(100, 200, True)
    fbrs_seg.add_click(150, 200, True)
    fbrs_seg.add_click(200, 200, False)

    mask = fbrs_seg.get_result_mask()*255
    cv2.imwrite('output/mask.jpg', mask)
    blended_image = fbrs_seg.get_visualization()
    cv2.imwrite('output/vis.jpg', cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
    xmin, ymin, xmax, ymax = mask_to_bbox(mask)
    img_bbox = cv2.rectangle(cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR),
                             (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.imwrite('output/bbox.jpg', img_bbox)

    fbrs_seg.finish_object()


def mask_to_bbox(mask):
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax-xmin, ymax-ymin]


def siammask_test():
    siammask = create_siammask()
    bbox = (286, 116, 190, 250) # tennis (x, y, w, h)
    files = glob.glob('/home/ntan/SiamMask/data/tennis/*.jpg')
    files.sort()
    print(len(files))
    images = [cv2.imread(it) for it in files]
    masks = siammask_process(siammask, images, bbox)
    print (len(masks))
    pass


_fbrs_seg = create_controller() # RGB
_siammask = create_siammask()   # BGR
DAVIS_ROOT = '/home/ntan/DAVIS2020/data/test-dev/'
MAX_NB_POINT = 5


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
    return masks
    pass

def add_masks(submit_masks, masks, ob_id):
    for i, _ in enumerate(submit_masks):
        submit_masks[i][masks[i].astype(bool)] = int(ob_id)
    return submit_masks
    pass

def fbrs_finetune_single_obj(image, mask, obj_id):
    global _fbrs_seg
    _fbrs_seg.set_image(image)

    mask = (mask==obj_id).astype(np.int)
    xmin, ymin, w, h = mask_to_bbox(mask)
    xmax, ymax = xmin+w, ymin+h

    count = 0
    flg_cont = True
    for ii in range(xmin+w//4, xmax-w//4-1, w//4):
        for jj in range(ymin+h//4, ymax-h//4-1, h//4):
            if mask[jj, ii] > 0:
                _fbrs_seg.add_click(ii, jj, True)
                count += 1
                if count == 4:
                    flg_cont = False
                    break
        if not flg_cont:
            break
    
    if count == 0:
        _fbrs_seg.finish_object()
        _fbrs_seg.reset_mask()
        return mask

    res_mask = _fbrs_seg.get_result_mask() #*255
    _fbrs_seg.finish_object()
    _fbrs_seg.reset_mask()
    return res_mask
    pass

def fbrs_finetune_single_mask(frame, pred_mask):
    image = cv2.imread(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nb_obj = len(np.unique(pred_mask)) - 1

    res_mask = np.zeros(pred_mask.shape)
    for obj_id in range(1, nb_obj+1, 1):
        ft_mm = fbrs_finetune_single_obj(image, pred_mask, obj_id)
        res_mask[ft_mm.astype(bool)] = obj_id
    return res_mask
    pass

def finetune_pred_masks(sequence_name, pred_masks):
    nb_frames = len(pred_masks)
    frames = glob.glob(DAVIS_ROOT+'/JPEGImages/480p/'+sequence_name+'/*.jpg')
    frames.sort()

    res_masks = []
    for i, frm in enumerate(frames):
            ft_mask = fbrs_finetune_single_mask(frames[i], pred_masks[i])
            res_masks.append(ft_mask)
            print(i, end='\r')

    return res_masks
    pass


def finetune_cascade_single_mask(seq_name, mask, i):
    nb_obj = len(np.unique(mask)) - 1

    res = np.zeros(mask.shape)
    for obj_id in range(1, nb_obj+1, 1):
        img = cv2.imread(DAVIS_ROOT+'/JPEGImages/480p/'+seq_name+'/%05d.jpg'%(i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tmp_mask = (mask==obj_id).astype(np.float)*255

        ft_mask = cascade_psp_finetune(img, tmp_mask)
        res[ft_mask.astype(bool)] = obj_id

    return res
    pass

def finetune_cascade(seq_name, pred_masks):
    res = []
    for i, it in enumerate(pred_masks):
        res.append(finetune_cascade_single_mask(seq_name, it, i))
        print(i, end='\r')
    return res
    pass


def run_interactive(d_scrib):
    # read scribble info
    scribbles = d_scrib['scribbles']
    sequence_name = d_scrib['sequence']

    index = 0
    list_objects = None
    for i, l_obj in enumerate(scribbles):
        if len(l_obj) > 0:
            index = i
            list_objects = l_obj
            break
    print(sequence_name, 'frame', index)
    
    # collect paths by object id
    dict_ob_paths = {}
    for obj in list_objects:
        pth = obj['path']
        ob_id = obj['object_id']
        if ob_id == 0:
            continue
        if ob_id not in dict_ob_paths:
            dict_ob_paths[ob_id] = []
        dict_ob_paths[ob_id].append(pth)
    print(dict_ob_paths.keys())

    # load image filenames
    file_images = glob.glob(DAVIS_ROOT+'/JPEGImages/480p/'+sequence_name+'/*.jpg')
    file_images.sort()
    image_height, image_width, _ = cv2.imread(file_images[0]).shape
    
    # draw demo scribble
    # image = cv2.imread(file_images[index])
    # for ob_id, ob_paths in dict_ob_paths.items():
    #     color = list(np.random.random(size=3) * 256)
    #     # print(color)
    #     for points in ob_paths:
    #         points = np.array(points) * np.array([image_width, image_height])
    #         points = points.astype(np.int32)
    #         # print(points[:10])
    #         image = cv2.polylines(image, [points], False, color, thickness=3)
    # cv2.imwrite('output/scriblle.jpg', image)

    # fbrs RGB
    image = cv2.imread(file_images[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    submit_masks = [np.zeros([image_height, image_width], dtype=np.int)
                                 for it in range(len(file_images))]
    try:
        for ob_id, ob_paths in sorted(dict_ob_paths.items()):
            obj_mask, obj_bbox = fbrs_scribble_to_mask(image, dict_ob_paths, ob_id,
                                                image_width, image_height)
            files = glob.glob(DAVIS_ROOT+'/JPEGImages/480p/'+sequence_name+'/*.jpg')
            files.sort()
            masks_tail = siammask_track(obj_bbox, files[index:])
            masks_head = siammask_track(obj_bbox, files[:index+1][::-1])[1:][::-1]
            masks = masks_head + masks_tail
            print('obj_id', ob_id, '#path', len(ob_paths), '#frame', len(files))
            submit_masks = add_masks(submit_masks, masks, ob_id)
    except Exception as exp:
        print('exception track', sequence_name, 'frame', index)
        print(exp)
        return np.array(submit_masks, dtype=np.int), index
    
    
    try:
        # finetune fbrs
        submit_masks = finetune_pred_masks(sequence_name, submit_masks)
        # submit_masks = finetune_cascade(sequence_name, submit_masks)
    except Exception as e:
        print('exception ft', sequence_name, 'frame', index)
        print(e)
    
    return np.array(submit_masks, dtype=np.int), index
    pass


if __name__ == "__main__":
    # fbrs_seg_test()
    # siammask_test()
    # run_interactive(json.load(open('../../data/trainval/Scribbles/rhino/001.json')))
    pass
