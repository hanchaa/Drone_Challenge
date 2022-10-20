import json
import os
import cv2

def json_preprocess(data_folder = './data_toy/'):
    file_list = os.listdir(data_folder)
    text_list = [f for f in file_list if 'text' in f]
    img_list = [f for f in file_list if 'image' in f]

    objects_dict = {
        '책상': ['desk', 'table'],
        '칠판': ['whiteboard'],
        '의자': ['chair'],
        '캐비닛': ['cabinet'],
        '모니터': ['monitor'],
        '상자': ['box'],
        '쓰레기통': ['trash can', 'garbage bin']
    }

    people_dict = {
        '아이': 'child',
        '아내': 'woman',
        '남편': 'man',
        '엄마': 'woman',
        '아빠': 'man'
    }

    colors_dict = {
        '빨강': 'red',
        '주황': 'orange',
        '노랑': 'yellow',
        '초록': 'green',
        '파랑': 'blue',
        '보라': 'purple',
        '흰색': 'white',
        '검정': 'black',
        '회색': 'gray'
    }

    query = {}
    images = {}

    for textfile in text_list:
        filepath = os.path.join(data_folder, textfile)
        with open(filepath, 'r') as f:
            data = json.load(f)     # text data

            # (1) json data parsing
            num = data.get('no')    # later used when making answer sheet file (json)
            objects = data.get('주변사물')
            people = data.get('일행')
            top = data.get('상의')
            bottom = data.get('하의')
            
            # (2) making query
            query[num] = []
            
            ## i. objects
            for obj in objects:
                for obj_query in objects_dict[obj]:
                    query[num].append(obj_query)
            
            ## ii. people & clothes
            ### rule-based female/male/child classification
            if people is not None:
                if '아내' in people:
                    # 요구조자 = male
                    shirt = colors_dict[top] + ' shirt'
                    pants = colors_dict[bottom] + ' pants'
                    query[num].append(shirt)
                    query[num].append(pants)
                    # 요구조자 본인
                    query[num].append('man')
                elif '남편' in people: 
                    # 요구조자 = female
                    shirt = colors_dict[top] + ' shirt'
                    skirt = colors_dict[bottom] + ' skirt'
                    query[num].append(shirt)
                    query[num].append(skirt)
                    # 요구조자 본인
                    query[num].append('woman')
                elif ('엄마' in people) or ('아빠' in people):
                    # 요구조자 = child
                    # female or male
                    shirt = colors_dict[top] + ' shirt'
                    pants = colors_dict[bottom] + ' pants'
                    query[num].append(shirt)
                    query[num].append(pants)
                    # 요구조자 본인
                    query[num].append('child')
                for person in people:
                    query[num].append(people_dict[person])
                
    for imgfile in img_list:
        filepath = os.path.join(data_folder, imgfile)
        img = cv2.imread(filepath)      # img data
        num = filepath.split('.png')
        num = num[0][-2:]
        images[num] = img
        
    return query, images

def json_postprocess(num_clues, data):
    # path, 1, ['01'',99]
    # json skeleton
    print('here')
    json_object = {
        'team_id': 'rony2',
        'secret': 'h8pnwElZ3FBnCwA4',
        'answer_sheet': {
            'room_id': None,
            'mission': 1,
            'answer': {
                'person_id': {
                }
            }
        }
    }
    # for i in range(num_clues):    # TODO: consider multiple clues
    for i in range(0, len(data[1])):
        if data[1][i] >= 500:
            json_object['answer_sheet']['room_id'] = data[1][i]
        else:
            json_object['answer_sheet']['answer']['person_id'][data[0]] = data[1][i]    # TODO: make it pretty

    return json_object