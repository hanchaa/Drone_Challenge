import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # task 1
    parser.add_argument('--clue_path', default=None, help='clue(img, txt) path')
    parser.add_argument('--json_output_path', default='.', help='json output path')

    parser.add_argument('--task1_debug', action="store_true", help='(optional)debug mode')
    parser.add_argument('--debug_input_path', default=None, help='debugging input image path')
    parser.add_argument('--debug_output_path', default=None, help='debugging output image path')
    
    parser.add_argument('--yolo_path', default='.', help='yolo task1 checkpoint path')
    parser.add_argument('--img_conf_th', type=float, default=0.6, help='img threshold')   # NOTE: determine best confidence threshold value
    parser.add_argument('--img_kp_th', type=float, default=50, help='img threshold')      # NOTE: determine best keypoint threshold value
    parser.add_argument('--txt_th', type=float, default=0.8, help='txt threshold')        # NOTE: determine value
    parser.add_argument('--od_th', type=float, default=0.5, help='OD threshold')          # NOTE: determine value
    parser.add_argument('--total_th', type=float, default=0.9, help='img+txt threshold')  # NOTE: determine value

    # task 2 vision
    parser.add_argument('--yolo_weights', type=str, help='model.pt path(s)')
    parser.add_argument('--strong_sort_weights', type=str)
    parser.add_argument('--config_strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--imgsz', '--img', '--img_size', type=int, default=(640, 640), help='inference size h,w')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show_vid', action='store_true', default=False, help='display tracking video results')
    parser.add_argument('--classes', type=int, default=None, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--half', action='store_true', default=False, help='use FP16 half-precision inference')
    parser.add_argument('--unclear_thres', type=int, default=10)

    # task 2 audio
    parser.add_argument('--checkpoint', type=str, default='./task2_audio/ckpts/2022_11_03_17_47_47.0100.pt')
    parser.add_argument('--filename', type=str, default='wavwavwavwav.wav')
    parser.add_argument('--mel', dest='mel', action='store_true')
    parser.set_defaults(mel=False)
    parser.add_argument('--ipd', dest='ipd', action='store_true')
    parser.set_defaults(ipd=False)
    parser.add_argument('--lightweight', dest='lightweight', action='store_true')
    parser.set_defaults(lightweight=False)
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--debug', type=bool, default=False)

    # task 3
    ## Craft (Detection)
    parser.add_argument('--craft_weight', default='task3/trained_model/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.5, type=float, help='text confidence threshold')
    parser.add_argument('--max_confidence', default=0.2, type=float, help='outputlist confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')

    ## WIW (Recognition)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--wiw_wieght', default='task3/trained_model/best_accuracy_new.pth', help="path to saved_model to evaluation")
    
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', default='True', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘?!', help='character label')
    parser.add_argument('--sensitive', default='True', help='for sensitive character mode')
    parser.add_argument('--PAD', default='True', help='whether to keep ratio then pad for image resize')
    
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=8, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # for debugging
    parser.add_argument('--video_path', type=str, default='/hub_data2/video_sample/set03_drone01.mp4', help='video path')

    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    return args
