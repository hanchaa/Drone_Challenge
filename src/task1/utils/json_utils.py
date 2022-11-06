import json

def json_preprocess(data_folder = './data_toy/'):
    text_list = data_folder

    objects_dict = {
        '책상': ['desk'],
        '칠판': ['whiteboard'],
        '의자': ['chair'],
        '캐비닛': ['cabinet'],
        '모니터': ['monitor'],
        '상자': ['box'],
        '쓰레기통': ['trash bin'],
        '바구니': ['bakset'],
        '컴퓨터': ['laptop'],
        '책장': ['bookshelf'],
        '프린터': ['printer'],
        '현수막': ['banner'],
        '거울': ['mirror'],
        '계단': ['stairs'],
        '장난감': ['toy'],
        '소화기': ['fire extinguisher'],
        '포스터': ['poster'],
        '세면대': ['sink'],
        '운동기구': ['exercise tool'],
        '스피커': ['speaker'],
    }

    people_dict = {
        '아이': 'person_child',
        '아내': 'person_woman',
        '남편': 'person_man',
        '엄마': 'person_woman',
        '아빠': 'person_man'
    }

    top_dict = {
        '빨강': 'up_red',
        '주황': 'up_orange',
        '노랑': 'up_yellow',
        '초록': 'up_green',
        '파랑': 'up_blue',
        '보라': 'up_purple',
        '흰색': 'up_white',
        '검정': 'up_black',
        '회색': 'up_gray'
    }

    low_dic = {
        '빨강': 'low_red',
        '주황': 'low_orange',
        '노랑': 'low_yellow',
        '초록': 'low_green',
        '파랑': 'low_blue',
        '보라': 'low_purple',
        '흰색': 'low_white',
        '검정': 'low_black',
        '회색': 'low_gray'

    }

    query = {}

    with open(text_list, 'r') as f:
        data = json.load(f)     # text data

        # (1) json data parsing
        num = data.get('no')    # later used when making answer sheet file (json)
        objects = data.get('주변사물')
        people = data.get('일행')
        top = data.get('상의')
        low = data.get('하의')
        
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
                if top is not None:
                    shirt = top_dict[top]
                    query[num].append(shirt)
                if low is not None:
                    pants = low_dic[low]
                    query[num].append(pants)
                # 요구조자 본인
                query[num].append('person_man')
            elif '남편' in people: 
                # 요구조자 = female
                if top is not None:
                    shirt = top_dict[top]
                    query[num].append(shirt)
                if low is not None:
                    skirt = low_dic[low]
                    query[num].append(skirt)
                # 요구조자 본인
                query[num].append('person_woman')
            elif ('엄마' in people) or ('아빠' in people):
                # 요구조자 = child
                # female or male
                if top is not None:
                    shirt = top_dict[top]
                    query[num].append(shirt)
                if low is not None:
                    pants = low_dic[low]
                    query[num].append(pants)
                # 요구조자 본인
                query[num].append('person_child')
            for person in people:
                query[num].append(people_dict[person])
        
    return query

def json_postprocess(clues_num, data):
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

    person_id_list = []
    for i in range(0, len(data)):
        if data[i] < 500:
            person_id_list.append(str(data[i]))
    json_object['answer_sheet']['answer']['person_id'].update({clues_num:person_id_list})

    return json_object
