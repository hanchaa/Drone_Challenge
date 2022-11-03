
import json



if __name__ == '__main__':
    with open('objects.json', 'r') as f:
        json_data = json.load(f)

    #기존 label : man,child,safe,person
    #추가 : 
    # key_lists = ['man','woman','child','monitor','cabinet','basket','box','trash bin','computer','laptop','bookshelf',
    # 'chair', 'printer','desk','whiteboard']
    
    key_lists = ['banner','mirror','stairs','toy','fire hydrant','poster','sink','sports equipment','speaker']
    
    key_lists = ['','','','','','','','','','']
    key2num = {key:i for i, key in enumerate(key_lists)}
    count_dict = dict()
    for key in key_lists:
        count_dict[key] = 0

    #import pdb; pdb.set_trace()
    object_list = []
    for data in json_data:
        img_id = data['image_id']
        #import pdb; pdb.set_trace()
        for x in data['objects']:
            #import pdb; pdb.set_trace()
            if x['names'][0] in key_lists:
                #import pdb; pdb.set_trace()
                object_dict = dict()
                count_dict[x['names'][0]] += 1 
                object_dict['img_id'] = img_id
                object_dict['label'] = key2num[x['names'][0]]
                object_dict['w'], object_dict['h'], object_dict['x'], object_dict['y'] = x['w'], x['h'], x['x'], x['y']
                #import pdb; pdb.set_trace()
                object_list.append(object_dict)
                print(f'{object_dict} added!')
                #object_dict['w'] = 
    
    
    #import pdb; pdb.set_trace()
    with open('vg_label.json', 'w', encoding='utf-8') as f:
        json.dump(object_list, f)


        