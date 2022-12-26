import os
import shutil



if __name__ == '__main__':
    val_path = '/home/shkwon129/project/yolov7/dataset/visual_genome/valid/labels'
    img_path = '/home/shkwon129/project/yolov7/dataset/visual_genome/train/images'
    save_path = '/home/shkwon129/project/yolov7/dataset/visual_genome/valid/images'
    val_label = os.listdir(val_path)
    val_img = []
    for data in val_label:
        val_img.append(data.replace('.txt','.jpg'))
    #import pdb; pdb.set_trace()

    for name in val_img:
        shutil.move(img_path+'/'+name, save_path)
        #import pdb; pdb.set_trace()
    
    # shutil.move("/tmp/my_test.txt", "/tmp/my_test_moved.txt")

    # if os.path.exists("/tmp/my_test_moved.txt"):
    #     print("exists")
