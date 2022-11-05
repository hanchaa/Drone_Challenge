
import json



if __name__ == '__main__':
    with open('attributes.json', 'r') as f:
        json_data = json.load(f)

    #기존 label : man,child,safe,person
    #추가 : 
    object_list = ['shirt','pants','skirt'] #2개 
    attribute_list = ['red','orange','yellow','green','blue','purple','white','black','gray'] #9개 
    #key_lists = ['man','woman','child','monitor','cabinet','basket','box','trash bin','computer','laptop','bookshelf',
    #'chair', 'printer','desk','whiteboard']
    
    #import pdb; pdb.set_trace()
    # key_lists = ['','','','','','','','','','']
    # key2num = {key:i for i, key in enumerate(key_lists)}
    # count_dict = dict()
    # for obj in object_list:
    #     for attr in attribute_list:
    #         key = obj+'_'+attr
    #         count_dict[key] = 0

    #import pdb; pdb.set_trace()
    #attribute_dict_keys : (['image_id', 'attributes'])
    
    
    #object_list = []
    # count_dict = dict()
    # cnt = 0
    # for data in json_data:
    #     img_id = data['image_id']
    #     #import pdb; pdb.set_trace()
    #     for x in data['attributes']:
    #         import pdb; pdb.set_trace()
    #         if 'attributes' in x.keys():
    #             for attribute in x['attributes']:
    #                 cnt += 1
    #                 for attr in attribute_list:
    #                     if attr in attribute:
    #                         if attr in count_dict.keys():
    #                             count_dict[attr] += 1
    #                         else:
    #                             count_dict[attr] = 1
    count_obj = dict()
    for data in json_data:
        img_id = data['image_id']
        #import pdb; pdb.set_trace()
        for x in data['attributes']:
            #import pdb; pdb.set_trace()
            for obj in object_list:
                if obj in x['names'][0]:
                    if obj in count_obj.keys():
                        count_obj[obj] += 1
                    else:
                        count_obj[obj] = 1

    #import pdb; pdb.set_trace()
    
    count_comb = dict()
    for data in json_data:
        img_id = data['image_id']
        #import pdb; pdb.set_trace()
        for x in data['attributes']:

            for obj in object_list:
                for attr in attribute_list:
                    if (obj in (x['names'][0])) and (attr in (x['names'][0])):
                        key = obj + '_' + attr
                        if key in count_comb.keys():
                            count_comb[key] += 1
                        else:
                            count_comb[key] = 1

                        if 'attributes' in x.keys():
                            if obj in (x['names'][0] or x['attributes']) and attr in (x['names'][0] or x['attributes']):
                                key = obj + '_' + attr
                                if key in count_comb.keys():
                                    count_comb[key] += 1
                                else:
                                    count_comb[key] = 1

    import pdb; pdb.set_trace()
                        
                   
    
    #import pdb; pdb.set_trace()
    with open('vg_label.json', 'w', encoding='utf-8') as f:
        json.dump(object_list, f)


        