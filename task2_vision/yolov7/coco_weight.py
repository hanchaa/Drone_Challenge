import torch 
from models.yolo import Model

if __name__ == '__main__':

    weights = 'yolov7_training.pt'
    device = 'cuda:1'
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    model = ckpt['model']
    #import pdb; pdb.set_trace()
    for key in model.state_dict().keys():
        if 'model.105.im' in key: 
            print(f'{key}')
            print(f'{model.state_dict()[key]}') #1,255,1,1    255 = 85 (object 수 + 1 + 4) * 3 (anchor 수 3개)
    import pdb; pdb.set_trace()

    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=17, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
