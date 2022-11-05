import json

def json_preprocess(data_folder = './data_toy/'):
    text_list = data_folder

    objects_dict = {
        '책상': ['desk'],
        '칠판': ['whiteboard'],
        # '의자': ['chair'],
        '캐비닛': ['cabinet'],
        '모니터': ['monitor'],
        '상자': ['box'],
        '쓰레기통': ['trash-bin']
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
        with open(textfile, 'r') as f:
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
                
    # for imgfile in img_list:
    #     filepath = os.path.join(data_folder, imgfile)
    #     img = cv2.imread(filepath)      # img data
    #     num = filepath.split('.png')
    #     num = num[0][-2:]
    #     images[num] = img
        
    return query#, images

def json_postprocess(clues_num, data, room_id, unclear=False):
    # json skeleton
    json_object = {
        'answer_sheet': {
            'room_id': None,
            'mission': "1",
            'answer': {
                'person_id': {
                }
            }
        }
    }

    if room_id != None:
        json_object['answer_sheet']['room_id'] = str(room_id)

    if unclear == True:
        for i in range(0, len(clues_num)):
            json_object['answer_sheet']['answer']['person_id'].update({clues_num[i]:["UNCLEAR"]})
    else:
        for i in range(0, len(clues_num)):
            person_id_list = []
            for j in range(0, len(data[1])):
                if data[1][j] < 500:
                    person_id_list.append(str(data[1][j]))
            json_object['answer_sheet']['answer']['person_id'].update({clues_num[i]:person_id_list})

    return json_object
